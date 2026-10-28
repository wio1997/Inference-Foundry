"""Bounded reads for explicit 910C module integration, no full-model allocation."""
from __future__ import annotations

import json
import math
import mmap
import struct
from pathlib import Path

import torch

from deepinfer.npu_ops.compat910c import decode_block_weight


class ByteCheckpoint:
    def __init__(self,root,max_read_bytes=32*1024**2,index_file='model.safetensors.index.json'):
        self.root=Path(root);self.max_read_bytes=max_read_bytes
        self.mapping=json.loads((self.root/index_file).read_text())['weight_map'];self.headers={}
        self._row_mmaps={}
        self._row_views={}

    def descriptor(self,name):
        shard=self.mapping[name]
        if shard not in self.headers:
            path=self.root/shard
            with path.open('rb') as f:
                prefix=f.read(8)
                if len(prefix)!=8:raise ValueError('incomplete shard header')
                size=struct.unpack('<Q',prefix)[0]
                if not 0<size<=min(100_000_000,path.stat().st_size-8):raise ValueError('invalid shard header size')
                self.headers[shard]=(json.loads(f.read(size)),8+size,path.stat().st_size)
        header,start,size=self.headers[shard];entry=header[name]
        begin,end=entry['data_offsets']
        if not 0<=begin<=end<=size-start:raise ValueError('invalid tensor range')
        return self.root/shard,start,entry

    def tensor(self,name):
        path,start,entry=self.descriptor(name);begin,end=entry['data_offsets']
        if end-begin>self.max_read_bytes:raise ValueError('bounded reader refuses large full-tensor allocation')
        dtype={'BF16':torch.bfloat16,'F32':torch.float32,'F8_E4M3':torch.uint8,'F8_E8M0':torch.uint8,'I8':torch.uint8,'U8':torch.uint8}.get(entry['dtype'])
        if dtype is None:raise ValueError('unsupported storage dtype')
        if math.prod(entry['shape'])*dtype.itemsize!=end-begin:raise ValueError('tensor shape/range mismatch')
        with path.open('rb') as f:f.seek(start+begin);data=f.read(end-begin)
        if len(data)!=end-begin:raise EOFError(name)
        return torch.frombuffer(bytearray(data),dtype=dtype).reshape(entry['shape']),entry['dtype']

    def rows(self,name,indices):
        """Read selected unquantized table rows without materializing the table."""
        if self.descriptor(name)[2]['dtype'] not in ('BF16','F32'):
            raise ValueError('rows requires an unquantized table; use explicit table_rows for Engram')
        return self.raw_rows(name,indices)[0]

    def raw_rows(self,name,indices):
        """Read byte-preserving table rows; callers must interpret the dtype."""
        path,start,entry=self.descriptor(name)
        dtype={'BF16':torch.bfloat16,'F32':torch.float32,'F8_E4M3':torch.uint8,'F8_E8M0':torch.uint8,'I8':torch.int8}.get(entry['dtype'])
        if dtype is None or len(entry['shape'])!=2:
            raise ValueError('row reads require a supported two-dimensional table')
        indices=torch.as_tensor(indices)
        if indices.dtype not in (torch.int32,torch.int64):
            raise ValueError('row indices must be integers')
        indices=indices.cpu();count,width=entry['shape'];begin,end=entry['data_offsets']
        row_bytes=width*dtype.itemsize
        if count*row_bytes!=end-begin:raise ValueError('tensor shape/range mismatch')
        if indices.numel()*row_bytes>self.max_read_bytes:
            raise ValueError('bounded reader refuses large row allocation')
        if bool(((indices<0)|(indices>=count)).any()):raise ValueError('row index out of bounds')
        if '.engram.embed.' in name:
            # Selected-row reads are the CPU offload boundary. Reopening and
            # seeking once per row serializes thousands of tiny reads across
            # the16 workers. A read-only map plus one indexed gather preserves
            # exact stored bytes and touches only the requested filesystem pages.
            import numpy as np
            if name not in self._row_views:
                key=str(path)
                if key not in self._row_mmaps:
                    with path.open('rb') as handle:
                        self._row_mmaps[key]=mmap.mmap(handle.fileno(),0,access=mmap.ACCESS_READ)
                storage_dtype={torch.bfloat16:np.uint16,torch.float32:np.float32,
                               torch.uint8:np.uint8,torch.int8:np.int8}[dtype]
                self._row_views[name]=np.ndarray((count,width),dtype=storage_dtype,
                    buffer=self._row_mmaps[key],offset=start+begin)
            selected=self._row_views[name][indices.numpy()]
            result=torch.from_numpy(selected)
            if dtype==torch.bfloat16:
                result=result.view(torch.bfloat16)
            return result,entry['dtype']
        data=bytearray()
        with path.open('rb') as f:
            for index in indices.flatten().tolist():
                f.seek(start+begin+index*row_bytes);row=f.read(row_bytes)
                if len(row)!=row_bytes:raise EOFError(name)
                data.extend(row)
        if not data:return torch.empty((*indices.shape,width),dtype=dtype),entry['dtype']
        return torch.frombuffer(data,dtype=dtype).reshape(*indices.shape,width),entry['dtype']

    def row_slice(self,name,start_row,stop_row):
        """Read a contiguous row shard, bounding host staging independently."""
        path,start,entry=self.descriptor(name)
        dtype={'BF16':torch.bfloat16,'F32':torch.float32,'I8':torch.int8,'F8_E8M0':torch.uint8}.get(entry['dtype'])
        if dtype is None or len(entry['shape'])!=2:raise ValueError('expected supported matrix storage')
        rows,width=entry['shape'];begin,end=entry['data_offsets']
        if not 0<=start_row<=stop_row<=rows:raise ValueError('invalid row slice')
        row_bytes=width*dtype.itemsize;size=(stop_row-start_row)*row_bytes
        if rows*row_bytes!=end-begin:raise ValueError('tensor shape/range mismatch')
        if size>self.max_read_bytes:raise ValueError('bounded reader refuses large row allocation')
        if size==0:return torch.empty((0,width),dtype=dtype),entry['dtype']
        with path.open('rb') as f:
            f.seek(start+begin+start_row*row_bytes);data=f.read(size)
        if len(data)!=size:raise EOFError(name)
        return torch.frombuffer(bytearray(data),dtype=dtype).reshape(stop_row-start_row,width),entry['dtype']

    def table_rows(self,name,indices,device):
        """Decode selected Engram E4M3/UE8M0 rows with explicit 1x32 geometry."""
        if not name.endswith('.engram.embed.weight'):
            raise ValueError('quantized row reads are restricted to explicit Engram tables')
        values,dtype=self.raw_rows(name,indices)
        if dtype!='F8_E4M3':raise ValueError('expected E4M3 Engram table')
        scales,sfdtype=self.raw_rows(name.removesuffix('.weight')+'.scale',indices)
        if sfdtype!='F8_E8M0' or scales.shape[:-1]!=values.shape[:-1]:
            raise ValueError('expected row-aligned UE8M0 Engram scales')
        width=values.shape[-1]
        if values.numel()==0:return values.to(device=device,dtype=torch.bfloat16)
        decoded=decode_block_weight(values.reshape(-1,width).to(device),scales.reshape(-1,scales.shape[-1]).to(device),fmt='e4m3',block_shape=(1,32))
        return decoded.reshape(values.shape)

    def matrix(self,name,device):
        data,dtype=self.tensor(name)
        if dtype in ('BF16','F32'):return data.to(device)
        scale_name=name.removesuffix('.weight')+'.scale';sf,sfdtype=self.tensor(scale_name)
        if sfdtype!='F8_E8M0':raise ValueError('expected explicit unpacked UE8M0 scales')
        if dtype=='F8_E4M3':fmt,block='e4m3',(32,32)
        elif dtype=='I8' and '.experts.' in name:fmt,block='e2m1',(1,32)
        else:raise ValueError('unsupported or ambiguous quantized matrix format')
        return decode_block_weight(data.to(device),sf.to(device),fmt=fmt,block_shape=block)
