# SPDX-License-Identifier: Apache-2.0
"""Bounded CPU math checks at actual GLM dimensions; all Q/K/V are synthetic."""
import argparse
import ast
import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import sys

import numpy as np
import torch
from deepinfer.hugging_face.glm5_next_config import Glm5NextConfig
from deepinfer.model_executor.models.glm5_next.cache import DSACacheLayout
from deepinfer.model_executor.models.glm5_next.sparse_reference import select_pools_reference,sparse_mla_reference

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--kernel-reference',type=Path,required=True)
p.add_argument('--prior-evidence',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
assert os.environ.get('TORCH_DEVICE_BACKEND_AUTOLOAD')=='0' and not list(Path('/dev').glob('davinci[0-9]*'))
torch.set_num_threads(2);torch.manual_seed(5320)
c=Glm5NextConfig.from_checkpoint(a.checkpoint);layout=DSACacheLayout.from_config(c,logical_block_size=128)
old=json.loads(a.prior_evidence.read_text());assert old['status']=='passed' and old['NPU_execution'] is False
assert old['config_sha256']==hashlib.sha256((a.checkpoint/'config.json').read_bytes()).hexdigest()
src=Path(__file__).resolve().parents[2]
assert old['source_sha256']['deepinfer/model_executor/models/glm5_next/cann_cache.py']==hashlib.sha256((src/'deepinfer/model_executor/models/glm5_next/cann_cache.py').read_bytes()).hexdigest()
j=json.loads(a.kernel_reference.read_text());f=next(f for f in j['files'] if f['path'].endswith('glm5_next_lightning_indexer.py'))
assert hashlib.sha256(f['content'].encode()).hexdigest()==f['sha256']
report={'status':'in_progress','native_round':20,'NPU_execution':False,'native_attention_executed':False,'model_generation':False,'model_weights_read':0,
        'input_source':'synthetic BF16 query/head weights/keys/values; actual checkpoint config only',
        'architecture':platform.machine(),'torch':torch.__version__,'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'config_sha256':old['config_sha256'],'reused_evidence_sha256':hashlib.sha256(a.prior_evidence.read_bytes()).hexdigest(),
        'reference_sha256':{f['path']:f['sha256']},'checks':{},'workloads':{},'source_sha256':{n:hashlib.sha256((src/n).read_bytes()).hexdigest() for n in [
            'deepinfer/model_executor/models/glm5_next/sparse_reference.py','tests/ascend/test_glm5_next_sparse_reference.py','tests/ascend/probe_glm5_next_sparse_reference_cpu.py']}}


def np64(x):return x.float().numpy().astype(np.float64)
def check(name,x,y,*,atol=0.,rtol=0.):
    torch.testing.assert_close(x,y,atol=atol,rtol=rtol);delta=(x.float()-y.float()).abs()
    report['checks'][name]={'passed':True,'shape':list(x.shape),'max_abs':float(delta.max()) if delta.numel() else 0.,'atol':atol,'rtol':rtol}


def output_check(name,x,expected):
    rounded=torch.tensor(expected,dtype=torch.float64).bfloat16();r=rounded.float()
    spacing=torch.maximum((torch.nextafter(rounded,torch.full_like(rounded,float('inf'))).float()-r).abs(),(torch.nextafter(rounded,torch.full_like(rounded,float('-inf'))).float()-r).abs())
    error=(x.float()-r).abs();assert bool((error<=spacing+1e-5).all())
    relative=float(torch.linalg.vector_norm(error)/torch.linalg.vector_norm(r).clamp_min(1e-20));assert relative<=.001
    report['checks'][name]={'passed':True,'shape':list(x.shape),'max_abs':float(error.max()),'relative_RMSE':relative,'CPU_guard':'1BF16ULP+1e-5,relativeRMSE<=0.001; not native acceptance'}


try:
    # Execute exactly the extracted pure torch weighted-query expression.
    tree=ast.parse(f['content']);node=next(n for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='qbar' for t in n.targets))
    args=ast.arguments(posonlyargs=[],args=[ast.arg(arg=n) for n in ('query','weights','start','rows')],kwonlyargs=[],kw_defaults=[],defaults=[])
    fn=ast.FunctionDef(name='qbar_reference',args=args,body=[node,ast.Return(value=ast.Name(id='qbar',ctx=ast.Load()))],decorator_list=[])
    module=ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[]));namespace={};exec(compile(module,'extracted_qbar','exec'),namespace)
    pools=600;heads=c.indexer_config['index_n_heads'];width=layout.index_dim
    query=(torch.randn(2,heads,width)*.125).bfloat16();weights=(torch.randn(2,heads)*.015625).bfloat16();keys=(torch.randn(pools,width)*.125).bfloat16();positions=torch.tensor([2047,2399])
    result=select_pools_reference(query,weights,keys,positions,layout)
    extracted=namespace['qbar_reference'](query,weights,0,2)
    qbar=(query.float()*weights.float()[...,None]).sum(1)
    check('weighted_query_extracted_torch',qbar,extracted)
    scores=np.einsum('thd,th->td',np64(query),np64(weights))@np64(keys).T
    for row,n in enumerate((512,600)):
        ids=sorted(range(n),key=lambda i:(-scores[row,i],i))[:512]
        check(f'row{row}_ranked_pool_ids_numpy',result.pool_ids[row],torch.tensor(ids,dtype=torch.int32))
        check(f'row{row}_ranked_scores_numpy',result.pool_scores[row],torch.tensor(scores[row,ids],dtype=torch.float32),atol=1e-7,rtol=2e-5)
        expanded=[p for pool in ids for p in range(pool*4,pool*4+4)]
        check(f'row{row}_expanded_causal_tokens',result.token_indices[row,:2048],torch.tensor(expanded,dtype=torch.int32))
        assert bool((result.token_indices[row,2048:]==-1).all())
        if n>512:report['workloads']['selection_boundary_margin']=float(scores[row,ids[-1]]-max(scores[row,i] for i in set(range(n))-set(ids)))
    # Exact ties are a defined CPU ordering only; native ties remain unverified.
    zeros=select_pools_reference(query[:1],weights[:1],torch.zeros_like(keys),torch.tensor([2399]),layout)
    check('CPU_ties_choose_lower_pool_ID',zeros.pool_ids[0],torch.arange(512,dtype=torch.int32))
    try:select_pools_reference(query,weights,keys,positions,layout,max_score_elements=10)
    except ValueError:report['checks']['score_budget_rejection']={'passed':True}
    else:raise AssertionError('score budget was ignored')
    report['workloads']['selector']={'queries':2,'heads':heads,'index_dim':width,'compressed_pools':600,'selected_pools':512,'score_elements':1200,'positions':positions.tolist(),'tie_scope':'deterministic CPU only'}
    # Tiny actual MLA geometry and selected context, no model/INT8 projection.
    positions=torch.tensor([7,10]);index_query=(torch.randn(2,heads,width)*.125).bfloat16();head_weights=(torch.randn(2,heads)*.015625).bfloat16()
    short=select_pools_reference(index_query,head_weights,keys[:2],positions,layout)
    h=c.mla_dimensions['num_attention_heads'];qk=c.mla_dimensions['qk_nope_head_dim'];v=c.mla_dimensions['v_head_dim']
    q=(torch.randn(2,h,qk)*.125).bfloat16();k=(torch.randn(11,h,qk)*.125).bfloat16();values=(torch.randn(11,h,v)*.125).bfloat16()
    output,lse=sparse_mla_reference(q,k,values,short.token_indices,positions)
    expected=np.zeros((2,h,v));expected_lse=np.zeros((2,h));serial=np.zeros((2,h,v))
    for row in range(2):
        chosen=short.token_indices[row][short.token_indices[row]>=0].tolist()
        # Python per-head serial selected-token softmax/value oracle.
        for head in range(h):
            logits=np.array([float(np.dot(np64(q[row,head]),np64(k[token,head])))*(qk**-.5) for token in chosen])
            exp=np.exp(logits-logits.max());prob=exp/exp.sum()
            expected[row,head]=sum(p*np64(values[token,head]) for p,token in zip(prob,chosen))
            expected_lse[row,head]=logits.max()+np.log(exp.sum())
        # These short histories select every causal token, so dense causal
        # attention is an independent decomposition despite pool-rank order.
        for head in range(h):
            logits=np.array([float(np.dot(np64(q[row,head]),np64(k[token,head])))*(qk**-.5) for token in range(int(positions[row])+1)])
            exp=np.exp(logits-logits.max());prob=exp/exp.sum()
            serial[row,head]=sum(p*np64(values[token,head]) for p,token in zip(prob,range(int(positions[row])+1)))
    output_check('MLA_selected_softmax_value_numpy',output,expected)
    output_check('MLA_all_selected_matches_dense_causal_numpy',output,serial)
    check('MLA_LSE_numpy',lse,torch.tensor(expected_lse,dtype=torch.float32),atol=2e-6,rtol=2e-6)
    # Selected token order does not change the mathematical attention result.
    permuted=short.token_indices.clone()
    for row in range(2):
        n=int((permuted[row]>=0).sum());permuted[row,:n]=permuted[row,:n].flip(0)
    reordered,_=sparse_mla_reference(q,k,values,permuted,positions)
    output_check('MLA_pool_order_invariance',reordered,expected)
    try:sparse_mla_reference(q,k,values,short.token_indices,positions,max_work_elements=1)
    except ValueError:report['checks']['MLA_work_budget_rejection']={'passed':True}
    else:raise AssertionError('MLA work budget ignored')
    report['workloads']['MLA']={'queries':2,'context_tokens':11,'heads':h,'qk_dim':qk,'value_dim':v,'scale':qk**-.5,'selected_tokens':[8,11],'selected_work_elements':19*h*(qk+v),'INT8_projections':False}
    assert 'torch_npu' not in sys.modules
    report['status']='passed'
except Exception as exc:
    report['status']='failed';report['failure']=repr(exc);raise
finally:
    report['ended_at']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    (a.output_dir/'cpu-sparse-reference-result.json').write_text(json.dumps(report,indent=2)+'\n')
