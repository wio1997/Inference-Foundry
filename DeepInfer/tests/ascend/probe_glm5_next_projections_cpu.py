# SPDX-License-Identifier: Apache-2.0
"""Bounded actual FLOAT indexer projections; no INT8 MLA or attention execution."""
import argparse
import ast
import __future__
import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from deepinfer.hugging_face.glm5_next_config import Glm5NextConfig
from deepinfer.hugging_face.modelslim910c import ModelSlimCheckpoint
from deepinfer.model_executor.models.glm5_next.cache import state_from_aligned_prefix
from deepinfer.model_executor.models.glm5_next.projections import GlmMLAProjections,GlmIndexerProjections

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--reference',type=Path,required=True)
p.add_argument('--cache-evidence',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
assert os.environ.get('TORCH_DEVICE_BACKEND_AUTOLOAD')=='0' and not list(Path('/dev').glob('davinci[0-9]*'))
torch.set_num_threads(2);torch.manual_seed(5318)
config=Glm5NextConfig.from_checkpoint(a.checkpoint)
reader=ModelSlimCheckpoint(a.checkpoint,max_read_bytes=16*1024*1024)
previous=json.loads(a.cache_evidence.read_text());assert previous['status']=='passed' and previous['NPU_execution'] is False
src=Path(__file__).resolve().parents[2]
assert previous['config_sha256']==hashlib.sha256((a.checkpoint/'config.json').read_bytes()).hexdigest()
for name in ('deepinfer/model_executor/models/glm5_next/cache.py','deepinfer/model_executor/models/glm5_next/request_state.py'):
    assert hashlib.sha256((src/name).read_bytes()).hexdigest()==previous['source_sha256'][name]
report={'status':'in_progress','native_round':18,'NPU_execution':False,'attention_executed':False,'INT8_MLA_executed':False,
        'architecture':platform.machine(),'torch':torch.__version__,'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'config_sha256':previous['config_sha256'],'reused_cache_evidence_sha256':hashlib.sha256(a.cache_evidence.read_bytes()).hexdigest(),
        'reference_sha256':hashlib.sha256(a.reference.read_bytes()).hexdigest(),'per_read_bound_bytes':reader.max_read_bytes,'CPU_FLOAT_accumulation':'FP32 GEMM -> BF16; native BF16 path unverified',
        'payload_bytes':0,'layers':{},'checks':{},'source_sha256':{name:hashlib.sha256((src/name).read_bytes()).hexdigest() for name in [
            'deepinfer/model_executor/models/glm5_next/projections.py','tests/ascend/test_glm5_next_projections.py','tests/ascend/probe_glm5_next_projections_cpu.py']}}


def check(name,x,y,*,atol=0.,rtol=0.):
    torch.testing.assert_close(x,y,atol=atol,rtol=rtol);delta=(x.float()-y.float()).abs()
    report['checks'][name]={'passed':True,'shape':list(x.shape),'max_abs':float(delta.max()) if delta.numel() else 0.,'atol':atol,'rtol':rtol}


def rounded_oracle(name,actual,expected):
    # CPU BF16 GEMM vs independent FP64 dot: allow one BF16 storage ULP plus
    # 1e-5 absolute floor for FP32 cancellation. Initial1e-7 failed for one
    # near-zero layer43 query: FP32 dot error3.632e-6 at value3.724e-4.
    # Diagnostic retained; relativeRMSE<=0.001 is unchanged. Not an NPU bound.
    expected=torch.tensor(expected,dtype=torch.float64).bfloat16()
    actual=actual.bfloat16();f=expected.float()
    spacing=torch.maximum((torch.nextafter(expected,torch.full_like(expected,float('inf'))).float()-f).abs(),
                           (torch.nextafter(expected,torch.full_like(expected,float('-inf'))).float()-f).abs())
    error=(actual.float()-f).abs();assert bool((error<=spacing+1e-5).all()),name
    rmse=float(torch.linalg.vector_norm(error)/torch.linalg.vector_norm(f).clamp_min(1e-20));assert rmse<=.001,name
    report['checks'][name]={'passed':True,'shape':list(actual.shape),'max_abs':float(error.max()) if error.numel() else 0.,
        'relative_RMSE':rmse,'max_BF16_ULP':float((error/spacing.clamp_min(1e-30)).max()) if error.numel() else 0.,
        'guard':'<=1 BF16 ULP+1e-5 and relativeRMSE<=0.001; CPU projection only'}


def np64(x):return x.float().numpy().astype(np.float64)


# Execute the installed forward method with metadata/cache dispatch stubs. Only
# real query/head-weight torch expressions run; no vLLM/device/runtime imports.
class Metadata:
    pass
class TailMetadata:
    pass
class TupleLinear:
    def __init__(self,module):self.module=module
    def __call__(self,x):return (self.module(x),None)


def installed_query_head(module,hidden,qr,*,compute_topk):
    tree=ast.parse(a.reference.read_bytes());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Glm5NextKPoolIndexerBackend')
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='forward')
    namespace={'torch':torch,'F':F,'AscendIndexerKPoolMetadata':Metadata,'AscendIndexerKPoolTailMetadata':TailMetadata,
               'get_forward_context':lambda:SimpleNamespace(attn_metadata={'tail':TailMetadata()},cudagraph_runtime_mode='none'),
               'CUDAGraphMode':SimpleNamespace(FULL='full')}
    exec(compile(ast.Module(body=[method],type_ignores=[]),str(a.reference),'exec',flags=__future__.annotations.compiler_flag),namespace)
    dummy=SimpleNamespace(_key_weight=module.key_weight,_head_weight_f32=module.head_weight,wq_b=TupleLinear(module.wq_b),
        n_head=module.num_heads,head_dim=module.layout.index_dim,softmax_scale=module.layout.index_dim**-.5,
        k_cache=object(),tail_cache=SimpleNamespace(prefix='tail'),_bound_cache=lambda unused:None,
        _forward_cann=lambda hidden,q,w,*args,**kwargs:(q,w))
    meta=Metadata();meta.query_metadata=None;meta.num_actual_tokens=len(hidden)
    return namespace['forward'](dummy,hidden,qr,hidden,meta,compute_topk=compute_topk)


try:
    class NoAccess:
        def __getattr__(self,name):raise AssertionError('MLA CPU guard accessed reader')
    try:GlmMLAProjections.from_checkpoint(NoAccess(),3,'cpu')
    except ValueError:report['checks']['native_mla_cpu_guard']={'passed':True}
    else:raise AssertionError('MLA CPU INT8 request accepted')
    for layer in (3,43):
        module=GlmIndexerProjections.from_checkpoint(reader,config,layer,'cpu')
        root=f'model.language_model.layers.{layer}.self_attn.indexer.';payloads={}
        for suffix in ('wq_b.weight','wk.weight','weights_proj.weight','index_kpool_compress_gate','k_norm.weight','k_norm.bias','index_kpool_compress_ape'):
            path,start,desc=reader.descriptor(root+suffix);begin,end=desc['data_offsets'];assert desc['dtype']=='BF16' and end-begin<=reader.max_read_bytes
            with path.open('rb') as f:f.seek(start+begin);raw=f.read(end-begin)
            assert len(raw)==end-begin;report['payload_bytes']+=len(raw);payloads[suffix]={'shape':desc['shape'],'dtype':desc['dtype'],'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
            owned={'wq_b.weight':module.wq_b.weight,'wk.weight':module.key_weight,'weights_proj.weight':module.head_weight,
                   'index_kpool_compress_gate':module.gate_weight,'k_norm.weight':module.norm_weight,'k_norm.bias':module.norm_bias,'index_kpool_compress_ape':module.ape}[suffix]
            assert owned.detach().bfloat16().contiguous().view(torch.uint8).numpy().tobytes()==raw
        n=7;hidden=torch.randn(n,4096).bfloat16();qr=torch.randn(n,1536).bfloat16()
        # Seven tokens exercise one completed pool and a trailing partial pool.
        result=module(hidden,qr);reference_q,reference_weights=installed_query_head(module,hidden,qr,compute_topk=True)
        check(f'layer{layer}_query_installed',result.query,reference_q)
        check(f'layer{layer}_weights_installed',result.head_weights,reference_weights)
        q_oracle=np64(qr)@np64(module.wq_b.weight.detach()).T
        rounded_oracle(f'layer{layer}_query_numpy',result.query.reshape(n,-1),q_oracle)
        raw=F.linear(hidden.float(),module.key_weight.float()).bfloat16()
        rounded_oracle(f'layer{layer}_key_dot_numpy',raw,np64(hidden)@np64(module.key_weight).T)
        rounded_oracle(f'layer{layer}_gate_dot_numpy',result.gate_scores,np64(hidden)@np64(module.gate_weight).T)
        head=np64(hidden)@np64(module.head_weight).T
        head_rounded=torch.tensor(head).bfloat16().float().numpy().astype(np.float64)
        rounded_oracle(f'layer{layer}_head_numpy',result.head_weights,head_rounded*module.weight_scale)
        centered=np64(raw)-np64(raw).mean(-1,keepdims=True)
        keys=centered/np.sqrt((centered**2).mean(-1,keepdims=True)+1e-6)*np64(module.norm_weight)+np64(module.norm_bias)
        check(f'layer{layer}_layernorm_numpy',result.raw_keys,torch.tensor(keys,dtype=torch.float32),atol=3e-6,rtol=3e-6)
        check(f'layer{layer}_retained_gate_boundary',result.gate_scores,F.linear(hidden.float(),module.gate_weight.float()).bfloat16().float())
        assert result.raw_keys.dtype==torch.float32 and not torch.equal(result.raw_keys,result.raw_keys.bfloat16().float())
        cache_only=module(hidden,None,compute_topk=False);rq,rw=installed_query_head(module,hidden,None,compute_topk=False)
        assert rq is rw is cache_only.query is cache_only.head_weights is None
        check(f'layer{layer}_cache_only_keys',cache_only.raw_keys,result.raw_keys)
        check(f'layer{layer}_cache_only_gates',cache_only.gate_scores,result.gate_scores)
        # Normalized MLA latents are synthetic; no INT8 projection is executed.
        latent=torch.randn(n,512).bfloat16();initial=state_from_aligned_prefix(latent[:0],torch.empty(0,128,dtype=torch.bfloat16),module.layout)
        whole=module.append_cache(latent,result,initial,start_position=0);state=initial;start=0
        for end in (1,3,4,7):
            part=module(hidden[start:end],qr[start:end]);state=module.append_cache(latent[start:end],part,state,start_position=start);start=end
        for name in ('latent','compressed','tail','tail_positions'):check(f'layer{layer}_projection_cache_segmented_{name}',getattr(state,name),getattr(whole,name))
        expected=[]
        for start in range(0,4,4):
            scores=np64(result.gate_scores[start:start+4])+np64(module.ape);weights=np.exp(scores-scores.max(0));weights/=weights.sum(0)
            expected.append((weights*np64(result.raw_keys[start:start+4])).sum(0))
        rounded_oracle(f'layer{layer}_pooled_numpy',whole.compressed,np.array(expected))
        for pos in (4,5,6):check(f'layer{layer}_raw_tail_{pos}',whole.tail[0,pos%4],result.raw_keys[pos])
        empty=module(hidden[:0],qr[:0]);assert empty.query.shape==(0,32,128)
        report['layers'][str(layer)]={'hidden_tokens':n,'hidden_size':4096,'query_rank':1536,'index_heads':32,'index_dim':128,
            'latent_source':'synthetic; no MLA INT8 projection','payloads':payloads,'runtime_APE':'FP32','raw_keys':'CPU FP32 GEMM -> BF16 -> FP32 LayerNorm; native BF16 path unverified',
            'gate_scores':'CPU FP32 GEMM -> BF16 result -> FP32; native BF16 path unverified','head_weights':'FP32 GEMM -> BF16 -> 1/sqrt(128*32) scale in BF16',
            'installed_reference_scope':'forward query/head expressions with dispatch stubs using explicit CPU FP32 query projection; retained-tail equation with CPU FP32 GEMM; no CANN operator'}
        del module,result,state,whole
    assert 'torch_npu' not in sys.modules
    report['status']='passed'
except Exception as exc:
    report['status']='failed';report['failure']=repr(exc);raise
finally:
    report['ended_at']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    (a.output_dir/'cpu-projection-result.json').write_text(json.dumps(report,indent=2)+'\n')
