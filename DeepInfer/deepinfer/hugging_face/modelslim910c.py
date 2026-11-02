# SPDX-License-Identifier: Apache-2.0
"""Ascend ModelSlim W8A8 checkpoint reads; preserve INT8 resident weights.

Layout and scale conversion follow the installed vLLM-Ascend W8A8_DYNAMIC
scheme. This loader never expands quantized weight matrices to BF16.
"""
import json

import torch

from deepinfer.npu_ops.checkpoint910c import ByteCheckpoint


class ModelSlimCheckpoint(ByteCheckpoint):
    def __init__(self,root,max_read_bytes=512*1024**2):
        super().__init__(root,max_read_bytes,'quant_model_weights.safetensors.index.json')
        self.config=json.loads((self.root/'config.json').read_text())
        self.quant=json.loads((self.root/'quant_model_description.json').read_text())
        declaration=self.config.get('quantization_config',{})
        method=declaration.get('quant_method') if isinstance(declaration,dict) else None
        if method=='ascend':
            pass
        elif 'quantization_config' not in self.config and self.config.get('model_type')=='glm5_next':
            # This GLM checkpoint declares quantization in its ModelSlim sidecar.
            # Validate the explicit architecture/layer/recipe contract first;
            # missing metadata on arbitrary models is never an Ascend default.
            from deepinfer.hugging_face.glm5_next_config import Glm5NextConfig
            contract=Glm5NextConfig.from_checkpoint(self.root)
            if contract.quantization_method!='W8A8_DYNAMIC':
                raise ValueError('expected explicit GLM W8A8_DYNAMIC metadata')
        else:
            raise ValueError('expected Ascend ModelSlim checkpoint or validated GLM W8A8_DYNAMIC')

    def matrix(self,name,device):
        # ByteCheckpoint.matrix also supports legacy FP4 expert storage. Never
        # interpret a ModelSlim signed INT8 expert as packed FP4 or dequantize it.
        if self.quant.get(name)!='FLOAT' or self.descriptor(name)[2]['dtype'] not in ('BF16','F32'):
            raise ValueError('ModelSlim matrix requires explicit FLOAT storage; use linear_payload for W8A8')
        return self.materialize(name,device)

    def tensor(self,name):
        value,dtype=super().tensor(name)
        return (value.view(torch.int8) if dtype=='I8' else value),dtype

    def linear_payload(self,prefix,*,scale_dtype=torch.bfloat16,output_slice=None):
        name=prefix+'.weight';recipe=self.quant.get(name)
        if output_slice is None:value,dtype=self.tensor(name)
        else:
            rows=self.descriptor(name)[2]['shape'][0]
            if not isinstance(output_slice,slice) or output_slice.step not in (None,1):
                raise ValueError('output sharding requires a contiguous row slice')
            first,last,_=output_slice.indices(rows)
            value,dtype=self.row_slice(name,first,last)
        if recipe=='FLOAT':
            if dtype not in ('BF16','F32'):raise ValueError(f'{name}: FLOAT storage mismatch')
            return value,None
        if recipe!='W8A8_DYNAMIC' or dtype!='I8' or value.ndim!=2:
            raise ValueError(f'{name}: unsupported ModelSlim recipe/storage {recipe}/{dtype}')
        scale,sd=self.tensor(prefix+'.weight_scale');offset,od=self.tensor(prefix+'.weight_offset')
        rows=self.descriptor(name)[2]['shape'][0]
        if sd!='F32' or od!='F32' or scale.shape!=(rows,1) or offset.shape!=scale.shape:
            raise ValueError(f'{name}: invalid per-channel scale/offset layout')
        if not bool(torch.isfinite(scale).all()) or not bool((scale>0).all()):
            raise ValueError(f'{name}: nonpositive/nonfinite scale')
        if bool((offset!=0).any()):raise ValueError(f'{name}: asymmetric weight offset is unsupported by the dynamic INT8 scheme')
        if output_slice is not None:scale=scale[output_slice]
        # vLLM creates these parameters using model params_dtype (BF16), even
        # when the checkpoint records F32; preserve that rounding explicitly.
        if scale_dtype not in (torch.bfloat16,torch.float32):raise ValueError('expected BF16/F32 runtime scale dtype')
        rounded=scale.flatten().to(scale_dtype)
        if not bool(torch.isfinite(rounded).all()) or not bool((rounded>0).all()):
            raise ValueError(f'{name}: scale invalid after runtime dtype conversion')
        return value,rounded

    def materialize(self,name,device,*,chunk_bytes=32*1024**2):
        """Load a dense matrix in bounded host chunks, preserving its dtype."""
        desc=self.descriptor(name)[2];size=desc['data_offsets'][1]-desc['data_offsets'][0]
        if size<=min(chunk_bytes,self.max_read_bytes):return self.tensor(name)[0].to(device)
        dtype={'BF16':torch.bfloat16,'F32':torch.float32,'I8':torch.int8}.get(desc['dtype'])
        if dtype is None or len(desc['shape'])!=2:raise ValueError('chunk loading requires a BF16/F32/INT8 matrix')
        rows,width=desc['shape'];step=min(chunk_bytes,self.max_read_bytes)//(width*dtype.itemsize)
        if step<1:raise ValueError('staging limit is smaller than one row')
        output=torch.empty((rows,width),dtype=dtype,device=device)
        for start in range(0,rows,step):
            end=min(start+step,rows);output[start:end].copy_(self.row_slice(name,start,end)[0])
        return output

    def load_linear(self,prefix,device,*,nz=True,output_slice=None):
        from deepinfer.npu_ops.w8a8 import AscendW8A8Linear
        weight,scale=self.linear_payload(prefix,output_slice=output_slice)
        if scale is None:return FloatLinear(weight.to(device))
        return AscendW8A8Linear(weight.to(device),scale.to(device),nz=nz)

    def engram_rows(self,layer,indices,device):
        """CPU-offloaded signed INT8 rows plus the checkpoint's E8M0 groups.

        Only selected rows and their scales are staged. The global table stays
        in CPU/filesystem storage; neither the table nor its scales occupy NPU
        memory in full. Negative image sentinels produce zero rows.
        """
        indices=torch.as_tensor(indices)
        if indices.dtype not in (torch.int32,torch.int64):raise ValueError('Engram indices must be integers')
        indices=indices.long().cpu()
        name=f'layers.{layer}.engram.embed.weight'
        desc=self.descriptor(name)[2]
        if desc['dtype']!='I8' or desc['shape'][1]!=256:raise ValueError('expected signed INT8 Engram width256')
        if bool((indices < -1).any()):raise ValueError('only -1 is a valid image sentinel')
        valid=indices>=0
        safe=indices.clamp_min(0)
        codes,dtype=self.raw_rows(name,safe);scales,sfdtype=self.raw_rows(name.removesuffix('.weight')+'.scale',safe)
        if dtype!='I8' or sfdtype!='F8_E8M0' or scales.shape!=(*codes.shape[:-1],codes.shape[-1]//32):
            raise ValueError('expected Engram group32 E8M0 scales')
        # Decode E8M0 on CPU, matching vLLM-Ascend's offload loader. Only the
        # small selected BF16 rows are transferred, not expanded matrices.
        sf=torch.ldexp(torch.ones_like(scales,dtype=torch.float32),scales.int()-127)
        sf=torch.where(scales==255,float('nan'),sf)
        values=(codes.float().unflatten(-1,(-1,32))*sf[...,None]).flatten(-2).bfloat16()
        return torch.where(valid[...,None],values,0.).to(device)


class FloatLinear(torch.nn.Module):
    """Unquantized projection selected by explicit ModelSlim FLOAT metadata."""
    def __init__(self,weight):
        super().__init__();self.weight=torch.nn.Parameter(weight,requires_grad=False)

    def forward(self,x):return torch.nn.functional.linear(x,self.weight)
