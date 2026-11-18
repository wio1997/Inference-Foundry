"""NPU INT8 Engram integration preserving SGLang's native hash/history/TP path.

Uses the fixed-image Host-UVA kernel already verified on CANN9.1. Full-model
readiness and performance remain separate gates.
"""
from pathlib import Path
import ctypes
import hashlib
import json
import re
import threading

import torch
from torch import nn
from safetensors import safe_open

import source_uva


def configure_host_abi():
    candidates=[Path('/usr/local/Ascend/ascend-toolkit/latest/include/acl/acl_rt.h'),
                Path('/usr/local/Ascend/cann-9.1.0/aarch64-linux/include/acl/acl_rt.h')]
    header=next(p for p in candidates if p.is_file())
    declaration=re.search(r'aclrtMallocHost\s*\(([^;]+?)\)\s*;',header.read_text())
    if declaration is None:raise RuntimeError('Cannot verify the installed HostMalloc ABI')
    arity=declaration[1].count(',')+1
    if arity!=2:raise RuntimeError(f'This pinned CANN9.1 adapter requires the verified 2-argument ABI, got {arity}')
    lib=source_uva._host_library()
    if not getattr(lib,'_sg_int8_abi_configured',False):
        raw=lib.aclrtMallocHost
        raw.argtypes=[ctypes.POINTER(ctypes.c_void_p),ctypes.c_size_t]
        def malloc(pointer,size,reserved):
            if reserved!=0:raise ValueError('Nonzero reserved HostMalloc argument')
            return raw(pointer,size)
        lib.aclrtMallocHost=malloc
        lib._sg_int8_abi_configured=True
    return arity


def load_rotation(model_root,hidden_size):
    path=Path(model_root)/'optional/quarot.safetensors'
    if not path.is_file():raise ValueError('The rotated checkpoint requires optional/quarot.safetensors')
    with safe_open(path,framework='pt',device='cpu') as f:
        rotation=f.get_tensor('global_rotation')
    if tuple(rotation.shape)!=(hidden_size,hidden_size) or hidden_size%32:
        raise ValueError('Unexpected global rotation shape')
    block=rotation[:32,:32].contiguous()
    if not torch.equal(rotation,torch.block_diag(*[block]*(hidden_size//32))):
        raise ValueError('Global rotation is not repeated block32')
    return block.float()


def rotated_gate(x,kv,q_weight,k_weight,eps,clamp_value,rotation_block,
                 input_ids=None,image_token_id=None):
    """Original-basis gate; the saved value projection is already rotated."""
    hc,dim=x.shape[-2:]
    key,value=kv.split([hc*dim,dim],dim=-1)
    key=key.float().unflatten(-1,(hc,dim))
    original=(x.float().unflatten(-1,(-1,32))@rotation_block.T).flatten(-2)
    rstd=torch.rsqrt(original.square().mean(-1)+eps)*torch.rsqrt(key.square().mean(-1)+eps)
    dot=(original*(q_weight.float()*k_weight.float())*key).sum(-1)*rstd*dim**-0.5
    # Exact FP32 sign bits, including negative zero. copysign uses a host
    # fallback in the pinned CANN9.1 stack and cannot execute during capture.
    magnitude=dot.abs().clamp_min(clamp_value).sqrt()
    negative=dot.view(torch.int32)<0
    gate=torch.sigmoid(torch.where(negative,-magnitude,magnitude))
    del key, original
    _record_lifetime_large_path(x)
    # Keep the residual in the rotated model basis; only gate evaluation unrotates.
    out=(x.float()+gate.unsqueeze(-1)*value.float().unsqueeze(-2)).to(x.dtype)
    if input_ids is not None:
        out=torch.where((input_ids==image_token_id)[:,None,None],x,out)
    return out


_LIFETIME_LARGE_PATH_RECORDED = False

def _record_lifetime_large_path(x):
    """One host-only path receipt per actual worker; no tensor operations."""
    global _LIFETIME_LARGE_PATH_RECORDED
    if _LIFETIME_LARGE_PATH_RECORDED or x.shape[0] < 8191:
        return
    import os, datetime
    directory=Path(__file__).resolve().parent/'gate_lifetime_hits'
    directory.mkdir(exist_ok=True)
    evidence={'pid':os.getpid(),'device_index':x.device.index,'rows':x.shape[0],
              'shape':list(x.shape),'dtype':str(x.dtype),
              'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'adapter_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'path':'actual rotated_gate after del key,original and before final unchanged output expression',
              'scope':'Executed model path only; not output correctness or request completion'}
    (directory/f'worker{os.getpid()}.json').write_text(json.dumps(evidence,indent=2)+'\n')
    _LIFETIME_LARGE_PATH_RECORDED=True


def install(model_root):
    """Install before native Engram model construction; exact checkpoint contract."""
    import sglang.srt.layers.engram as native
    root=Path(model_root).resolve(strict=True)
    if hasattr(native,'_int8_npu_adapter'):
        if native._int8_npu_adapter['model_root']!=str(root):
            raise ValueError('An Engram adapter for another checkpoint is installed')
        return native._int8_npu_adapter
    config=json.loads((root/'config.json').read_text())
    text=config.get('text_config',config)
    if config.get('model_type')!='deepseek_v41':raise ValueError('Requires DeepSeek V4.1 checkpoint')
    expected={'value_projection_rotated':True,'value_basis':'quarot_global',
              'key_and_gate_basis':'original','runtime_delta_rotation':False}
    if config.get('engram_rotation_config')!=expected:raise ValueError('Unknown Engram rotation contract')
    quant=json.loads((root/'quant_model_description.json').read_text())
    index=json.loads((root/'quant_model_weights.safetensors.index.json').read_text())['weight_map']
    for layer in text['engram_layer_ids']:
        if quant.get(f'layers.{layer}.engram.embed.quant_type')!='INT8_SYM':
            raise ValueError('Expected INT8_SYM Engram weights')
    rotation_cpu=load_rotation(root,text['hidden_size'])
    arity=configure_host_abi()
    NativeEmbedding=native.EngramEmbedding
    native_init=native.Engram.__init__
    rotations={}

    class NPUINT8EngramEmbedding(NativeEmbedding):
        def __init__(self,num_embeddings,dim,layer_id,column_bounds):
            nn.Module.__init__(self)
            parallel=native.get_parallel()
            self.dim=dim;self.tp_size=parallel.tp_size
            self.row_start,end=native.shard_range(num_embeddings,column_bounds,parallel.tp_rank,self.tp_size)
            self.rows=end-self.row_start
            self.host_table=None;self.peer=None;self._sort_host_reads=False
            self.staged_by_scheduler=False
            self.lookup=native.EngramLookup.ALL_REDUCE
            self._cols_per_rank=native.cols_per_rank(len(column_bounds)-1,self.tp_size)
            if dim!=256:raise ValueError('Only checkpoint Engram width256 is verified')
            self._weight_key=f'layers.{layer_id}.engram.embed.weight'
            self._scale_key=f'layers.{layer_id}.engram.embed.scale'
            self._expected_shapes={self._weight_key:(num_embeddings,dim),self._scale_key:(num_embeddings,dim//32)}
            for key,dtype in [(self._weight_key,('I8','INT8')),(self._scale_key,('F8_E8M0','F32'))]:
                filename=Path(index[key])
                if filename.is_absolute() or '..' in filename.parts:raise ValueError('Unsafe indexed shard path')
                with safe_open(root/filename,framework='pt',device='cpu') as f:
                    view=f.get_slice(key)
                    if tuple(view.get_shape())!=self._expected_shapes[key] or view.get_dtype() not in dtype:
                        raise ValueError(f'Unexpected Engram source shape/dtype for {key}')
            device=torch.device('npu',torch.npu.current_device())
            # Empty geometry retains valid pointer storage while native lookup zero-fills.
            self._codes=source_uva.HostUvaBuffer((max(self.rows,1),dim),torch.int8,device)
            try:self._scales=source_uva.HostUvaBuffer((max(self.rows,1),dim//32),torch.float32,device)
            except BaseException:self._codes.close();raise
            self._codes.tensor.zero_();self._scales.tensor.zero_()
            self.weight=nn.Parameter(self._codes.tensor[:self.rows],requires_grad=False)
            self.scale=nn.Parameter(self._scales.tensor[:self.rows],requires_grad=False)
            self.weight.weight_loader=self._load_rows
            self.scale.weight_loader=self._load_rows
            self._load_lock=threading.Lock();self._loaded=False;self._callbacks=set()
            self.load_receipt=None

        def _load_rows(self,param,loaded_weight):
            if param is self.weight:key=self._weight_key
            elif param is self.scale:key=self._scale_key
            else:raise ValueError('Unknown Engram parameter callback')
            if tuple(loaded_weight.shape)!=self._expected_shapes[key]:
                raise ValueError('Checkpoint callback shape mismatch')
            with self._load_lock:
                if not self._loaded:
                    start=self.row_start;end=start+self.rows
                    with safe_open(root/index[self._weight_key],framework='pt',device='cpu') as wf, \
                         safe_open(root/index[self._scale_key],framework='pt',device='cpu') as sf:
                        weights=wf.get_slice(self._weight_key);scales=sf.get_slice(self._scale_key)
                        for first in range(start,end,65536):
                            stop=min(first+65536,end);a=first-start;b=stop-start
                            self.weight.data[a:b].copy_(weights[first:stop])
                            # Numerically cast E8M0 scales; never reinterpret codes as FP8.
                            self.scale.data[a:b].copy_(scales[first:stop].float())
                    self._loaded=True
                    self.load_receipt={'row_start':start,'rows':self.rows,'weight_dtype':str(self.weight.dtype),
                                       'scale_dtype':str(self.scale.dtype),'weight_key':self._weight_key,
                                       'scale_key':self._scale_key,'chunk_rows':65536,'codes_host':self.weight.device.type=='cpu',
                                       'scales_host':self.scale.device.type=='cpu'}
                self._callbacks.add(key)

        def finish_load(self,label):
            if not self._loaded or self._callbacks!={self._weight_key,self._scale_key}:
                raise ValueError(f'Engram {label} codes/scale callbacks did not both complete')

        def _owned_rows(self,indices):
            if self.rows==0:return self.new_rows(indices).zero_()
            flat=indices.reshape(-1,1)
            output=torch.empty((flat.shape[0],self.dim),device=indices.device,dtype=torch.bfloat16)
            if flat.shape[0]:
                source_uva._engram_host_uva_gather_dequant_kernel[(flat.shape[0],)](
                    self._codes.ptrs,self._scales.ptrs,flat,output,flat.shape[0],
                    self.row_start,self.row_start+self.rows,flat.stride(0),
                    CHUNK=source_uva.CHUNK_ROWS,WIDTH=self.dim,GROUP=32,HEAD_START=0,
                    LOCAL_HEADS=1,PAD_HEADS=1,QUANTIZED=True,MXFP8=False,num_warps=4)
            return output.view(*indices.shape,self.dim)

        def close_host_offload(self):
            for name,alias in [('_codes','weight'),('_scales','scale')]:
                buffer=getattr(self,name,None)
                if buffer is None:continue
                buffer.close();setattr(self,name,None)
                old=getattr(self,alias)
                setattr(self,alias,nn.Parameter(torch.empty(0,dtype=old.dtype,device='cpu'),requires_grad=False))

    def init_engram(self,*args,**kwargs):
        native_init(self,*args,**kwargs)
        device=torch.npu.current_device()
        if device not in rotations:rotations[device]=rotation_cpu.to(torch.device('npu',device))

    def gate(x,kv,q_weight,k_weight,eps,clamp_value,input_ids=None,image_token_id=None):
        return rotated_gate(x,kv,q_weight,k_weight,eps,clamp_value,rotations[x.device.index],input_ids,image_token_id)

    native.EngramEmbedding=NPUINT8EngramEmbedding
    native.Engram.__init__=init_engram
    native.engram_gate=gate
    receipt={'model_root':str(root),'HostMalloc_arity':arity,'register_flags':2,
             'native_lookup_and_TP_path_preserved':True,'native_hasher_and_history_preserved':True,
             'source_uva_sha256':hashlib.sha256(Path(source_uva.__file__).read_bytes()).hexdigest(),
             'full_model_readiness_verified':False}
    native._int8_npu_adapter=receipt
    return receipt
