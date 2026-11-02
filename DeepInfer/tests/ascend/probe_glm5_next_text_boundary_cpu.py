# SPDX-License-Identifier: Apache-2.0
"""Actual bounded text-table rows/norm; no backbone, full logits or generation."""
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
from torch.nn import functional as F
from deepinfer.hugging_face.modelslim910c import ModelSlimCheckpoint
from deepinfer.model_executor.models.glm5_next.text_boundary import GlmTextBoundary

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--reference',type=Path,required=True)
p.add_argument('--logits-reference',type=Path,required=True);p.add_argument('--prior-evidence',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
assert os.environ.get('TORCH_DEVICE_BACKEND_AUTOLOAD')=='0' and not list(Path('/dev').glob('davinci[0-9]*'))
torch.set_num_threads(2);torch.manual_seed(5324)
src=Path(__file__).resolve().parents[2];old=json.loads(a.prior_evidence.read_text());assert old['status']=='passed' and old['NPU_execution'] is False
assert old['config_sha256']==hashlib.sha256((a.checkpoint/'config.json').read_bytes()).hexdigest()
assert old['source_sha256']['deepinfer/hugging_face/glm5_next_plan.py']==hashlib.sha256((src/'deepinfer/hugging_face/glm5_next_plan.py').read_bytes()).hexdigest()


class Reader(ModelSlimCheckpoint):
    def __init__(self,*args,**kwargs):super().__init__(*args,**kwargs);self.row_calls=[];self.shard_calls=[]
    def raw_rows(self,name,ids):self.row_calls.append(ids.tolist());return super().raw_rows(name,ids)
    def row_slice(self,name,start,stop):self.shard_calls.append((start,stop));return super().row_slice(name,start,stop)


reader=Reader(a.checkpoint,max_read_bytes=256*1024);boundary=GlmTextBoundary.from_checkpoint(reader,max_unique_tokens=32,max_tokens=128)
report={'status':'in_progress','native_round':24,'NPU_execution':False,'backbone_executed':False,'full_vocabulary_projection':False,'model_generation':False,
        'architecture':platform.machine(),'torch':torch.__version__,'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'config_sha256':old['config_sha256'],'reused_evidence_sha256':hashlib.sha256(a.prior_evidence.read_bytes()).hexdigest(),
        'per_read_bound_bytes':reader.max_read_bytes,'payloads':{},'checks':{},'reference_sha256':{},
        'source_sha256':{n:hashlib.sha256((src/n).read_bytes()).hexdigest() for n in [
            'deepinfer/model_executor/models/glm5_next/text_boundary.py','tests/ascend/test_glm5_next_text_boundary.py','tests/ascend/probe_glm5_next_text_boundary_cpu.py']}}


def check(name,x,y,*,atol=0.,rtol=0.):
    torch.testing.assert_close(x,y,atol=atol,rtol=rtol);error=(x.float()-y.float()).abs()
    report['checks'][name]={'passed':True,'shape':list(x.shape),'max_abs':float(error.max()) if error.numel() else 0.,'atol':atol,'rtol':rtol}


def np64(x):return x.float().numpy().astype(np.float64)
def rounded_check(name,x,expected):
    ref=torch.tensor(expected,dtype=torch.float64).bfloat16();f=ref.float()
    spacing=torch.maximum((torch.nextafter(ref,torch.full_like(ref,float('inf'))).float()-f).abs(),(torch.nextafter(ref,torch.full_like(ref,float('-inf'))).float()-f).abs())
    error=(x.float()-f).abs();assert bool((error<=spacing+1e-5).all())
    relative=float(torch.linalg.vector_norm(error)/torch.linalg.vector_norm(f).clamp_min(1e-20));assert relative<=.001
    report['checks'][name]={'passed':True,'shape':list(x.shape),'max_abs':float(error.max()) if error.numel() else 0.,'relative_RMSE':relative,'CPU_guard':'1BF16ULP+1e-5,relativeRMSE<=0.001; not native acceptance'}


def raw_rows(name,ids):
    path,start,desc=reader.descriptor(name);begin,end=desc['data_offsets'];width=desc['shape'][-1];size=width*2
    assert desc['dtype']=='BF16' and len(ids)*size<=reader.max_read_bytes
    with path.open('rb') as f:
        chunks=[]
        for row in ids:f.seek(start+begin+row*size);chunks.append(f.read(size))
    raw=b''.join(chunks);assert len(raw)==len(ids)*size
    result=torch.from_numpy(np.frombuffer(raw,dtype='<u2').copy()).view(torch.bfloat16).reshape(len(ids),width)
    return result,raw


try:
    m=boundary.manifest;assert (m.vocab_size,m.hidden_size,m.streams)==(154880,4096,4)
    ids=torch.tensor([0,1,42,m.vocab_size-1,42]);embedding=boundary.embed(ids)
    expected,raw=raw_rows(m.embedding_name,ids.tolist());check('embedding_independent_raw_rows',embedding,expected)
    assert reader.row_calls==[[0,1,42,m.vocab_size-1]]
    report['payloads']['embedding']={'unique_rows':[0,1,42,m.vocab_size-1],'unique_bytes':4*4096*2,'ordered_output_sha256':hashlib.sha256(raw).hexdigest()}
    # Extract only the pure hc_contract function; no reference-package imports.
    refs=json.loads(a.reference.read_text());f=next(f for f in refs['files'] if f['path'].endswith('/mhc.py'))
    assert hashlib.sha256(f['content'].encode()).hexdigest()==f['sha256'];report['reference_sha256'][f['path']]=f['sha256']
    tree=ast.parse(f['content']);node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='hc_contract')
    namespace={'torch':torch};exec(compile(ast.Module(body=[node],type_ignores=[]),'hc_contract_reference','exec',flags=__future__.annotations.compiler_flag),namespace)
    streams=(torch.randn(5,4,4096)*.125).bfloat16();collapsed=namespace['hc_contract'](streams,4)
    check('final_contract_mean_reference',streams.float().mean(1).bfloat16(),collapsed)
    hidden=boundary.final_hidden(streams)
    norm=boundary.norm_weight;collapsed64=np64(collapsed)
    expected=collapsed64/np.sqrt((collapsed64**2).mean(-1,keepdims=True)+m.norm_eps)*np64(norm)
    rounded_check('final_RMSNorm_numpy',hidden,expected)
    path,start,desc=reader.descriptor(m.norm_name);begin,end=desc['data_offsets']
    with path.open('rb') as f:f.seek(start+begin);norm_raw=f.read(end-begin)
    assert norm.contiguous().view(torch.uint8).numpy().tobytes()==norm_raw
    report['payloads']['norm']={'bytes':len(norm_raw),'sha256':hashlib.sha256(norm_raw).hexdigest()}
    bf16=boundary.head_shard(100,108,max_rows=8);fp32=boundary.head_shard(100,108,max_rows=8,head_dtype=torch.float32)
    weight,raw=raw_rows(m.head_name,list(range(100,108)));check('head_shard_independent_bytes',bf16.weight,weight)
    assert bf16.vocab_start==100 and bf16.vocab_end==108
    dot=np64(hidden)@np64(weight).T
    expected_bf16=torch.tensor(dot).bfloat16().float().numpy().astype(np.float64)*m.logit_scale
    rounded_check('BF16_head_numpy_boundary_scale',bf16(hidden),expected_bf16)
    check('FP32_head_numpy',fp32(hidden),torch.tensor(dot*m.logit_scale,dtype=torch.float32),atol=1e-5,rtol=5e-5)
    report['payloads']['head']={'row_range':[100,108],'unique_bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'head_precisions':['BF16','FP32']}
    # Actual FP32 head_dtype branch and processor scale/shaping run as extracted
    # Python with no vLLM/native imports; TP gather and bias are excluded.
    raw_ref=a.logits_reference.read_bytes();report['reference_sha256']['logits_processor.py']=hashlib.sha256(raw_ref).hexdigest()
    tree=ast.parse(raw_ref);cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='LogitsProcessor')
    nodes=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in ('_apply_head','_get_logits','forward')]
    class Unquant:pass
    ns={'torch':torch,'F':F,'UnquantizedEmbeddingMethod':Unquant,'UnquantizedLinearMethod':Unquant,'current_platform':SimpleNamespace(is_cuda=lambda:False,is_rocm=lambda:False)}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'logits_reference','exec',flags=__future__.annotations.compiler_flag),ns)
    processor=SimpleNamespace(head_dtype=torch.float32,logits_as_input=False,soft_cap=None,scale=m.logit_scale,org_vocab_size=m.vocab_size)
    processor._apply_head=lambda head,x,b:ns['_apply_head'](processor,head,x,b)
    processor._get_logits=lambda x,head,b,skip:ns['_get_logits'](processor,x,head,b,skip)
    head=SimpleNamespace(weight=fp32.weight,quant_method=Unquant(),tp_size=1)
    reference_logits=ns['forward'](processor,head,hidden)
    check('FP32_head_extracted_logits_processor',fp32(hidden),reference_logits)
    before=(len(reader.row_calls),len(reader.shard_calls))
    for action in (lambda:boundary.embed(torch.arange(33)),lambda:boundary.embed(torch.zeros(129,dtype=torch.int64)),lambda:boundary.head_shard(0,m.vocab_size,max_rows=8)):
        try:action()
        except ValueError:pass
        else:raise AssertionError('read/allocation bound silently bypassed')
    assert before==(len(reader.row_calls),len(reader.shard_calls));report['checks']['bounds_reject_before_payload']={'passed':True}
    check('empty_embedding_shape',torch.tensor(boundary.embed(torch.empty(0,dtype=torch.int64)).shape),torch.tensor([0,4096]))
    check('empty_final_shape',torch.tensor(boundary.final_hidden(streams[:0]).shape),torch.tensor([0,4096]))
    check('empty_head_shape',torch.tensor(fp32(hidden[:0]).shape),torch.tensor([0,8]))
    report['unique_weight_payload_bytes']=4*4096*2+len(norm_raw)+len(raw)
    report['table_shapes']={'embedding':[154880,4096],'head':[154880,4096],'full_tables_loaded':False}
    assert 'torch_npu' not in sys.modules
    report['status']='passed'
except Exception as exc:
    report['status']='failed';report['failure']=repr(exc);raise
finally:
    report['ended_at']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    (a.output_dir/'cpu-text-boundary-result.json').write_text(json.dumps(report,indent=2)+'\n')
