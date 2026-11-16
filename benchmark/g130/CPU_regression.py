"""CPU seam regression using the actual producer/handoff/DSPARK consumer bodies.

This does not import the serving framework or execute NPU/model inference.
Only the orchestrator/request context is replaced with a small deterministic DTO.
"""
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import ast,dataclasses,hashlib,json,torch
R=Path(__file__).resolve().parents[2];D=R/'benchmark/g130'
B=D/'source/sampling_batch_info.py'
P=D/'source/min_new_tokens.py'
N=D/'source/sampling_params.py'
torch.set_num_threads(2)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def function(path,name):
 t=ast.parse(path.read_text());f=next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name==name)
 env={'torch':torch,'Any':Any,'apply_custom_logit_processor':lambda *a,**k:(_ for _ in ()).throw(AssertionError('No custom processor in this task'))}
 exec('from __future__ import annotations\n'+ast.unparse(f),env);return env[name]
plan=json.loads((D/'candidate_plan.json').read_text())
for name,item in plan['changes'].items():
 assert sha(D/'original'/name)==item['original_sha256'] and sha(R/'compat/runtime/g130'/name)==item['candidate_sha256']
 assert (R/'compat/runtime/g130'/name).read_text()==(D/'original'/name).read_text().replace('acc_linear_penalties','acc_additive_penalties')
btree=ast.parse(B.read_text());bclass=next(n for n in btree.body if isinstance(n,ast.ClassDef) and n.name=='SamplingBatchInfo')
fields=[n.target.id for n in bclass.body if isinstance(n,ast.AnnAssign) and isinstance(n.target,ast.Name)]
assert 'acc_additive_penalties' in fields and 'acc_linear_penalties' not in fields
assert not any(isinstance(n,ast.FunctionDef) and n.name=='acc_linear_penalties' for n in bclass.body)
ptree=ast.parse(P.read_text());pclass=next(n for n in ptree.body if isinstance(n,ast.ClassDef));pclass.bases=[ast.Name(id='_BatchedPenalizer',ctx=ast.Load())]
penv={'torch':torch,'_BatchedPenalizer':object,'is_pin_memory_available':lambda device:False};exec('from __future__ import annotations\n'+ast.unparse(pclass),penv);Penalizer=penv[pclass.name]
@dataclasses.dataclass
class Forward:
 temperatures:Any
 vocab_size:int
 penalizer_orchestrator:Any
 acc_additive_penalties:Any=None
 acc_scaling_penalties:Any=None
 has_custom_logit_processor:bool=False
 grammar_mask:Any=None
 logit_bias:Any=None
 def __len__(self):return len(self.temperatures)
env={'torch':torch,'dataclasses':dataclasses,'SamplingBatchInfo':Forward,'_fast_copy_enabled':lambda:False}
for name in ['update_penalties','copy_for_forward','_apply_pre_grammar_logits_transforms','_apply_post_grammar_logits_transforms','apply_logits_bias']:
 method=next(n for n in bclass.body if isinstance(n,ast.FunctionDef) and n.name==name);exec('from __future__ import annotations\n'+ast.unparse(method),env);setattr(Forward,name,env[name])
old_pred=function(D/'original/dspark_verify.py','verify_logits_adjustments_are_noop');new_pred=function(R/'compat/runtime/g130/dspark_verify.py','verify_logits_adjustments_are_noop')
old_apply=function(D/'original/dflash_utils.py','apply_dflash_verify_logits_adjustments');new_apply=function(R/'compat/runtime/g130/dflash_utils.py','apply_dflash_verify_logits_adjustments')
# Native original method proves ignore_eos does not remove EOS from min_tokens mask.
ntree=ast.parse(N.read_text());nclass=next(n for n in ntree.body if isinstance(n,ast.ClassDef) and n.name=='SamplingParams');nf=next(n for n in nclass.body if isinstance(n,ast.FunctionDef) and n.name=='update_from_generation_config');nenv={};exec('from __future__ import annotations\n'+ast.unparse(nf),nenv)
ns=SimpleNamespace(ignore_eos=True,_all_stop_token_ids=set(),stop_token_ids=[]);nenv[nf.name](ns,{},eos_token_id=1);assert ns._all_stop_token_ids=={1}
cases=[]
for bs in [1,2,4]:
 for width in [1,6]:
  for dtype in [torch.float32,torch.bfloat16]:
   for emitted in [0,4]:
    reqs=[SimpleNamespace(sampling_params=SimpleNamespace(min_new_tokens=4 if i%2==0 else 0,stop_token_ids=None),eos_token_ids={1},tokenizer=SimpleNamespace(additional_stop_token_ids=None,eos_token_id=1)) for i in range(bs)]
    penalty=Penalizer();penalty.orchestrator=SimpleNamespace(device='cpu',vocab_size=8,reqs=lambda:reqs);assert penalty._is_required();penalty._prepare()
    for _ in range(emitted):penalty._cumulate_output_tokens(torch.zeros(bs,dtype=torch.int64))
    class Orch:
     is_required=True
     def accumulate_additive_penalties(self,logits):penalty._apply(logits)
     def accumulate_scaling_penalties(self):return None
     def apply(self,logits):penalty._apply(logits)
    info=Forward(torch.ones(bs,1),8,Orch());copied=info.copy_for_forward()
    assert copied.penalizer_orchestrator is None and copied.acc_additive_penalties is not None and copied.acc_scaling_penalties is None
    assert old_pred(copied) is True and new_pred(copied) is False
    logits=torch.zeros(bs*width,8,dtype=dtype);logits[:,1]=10;logits[:,0]=9
    expected=logits.view(bs,width,8).clone();expected.add_(copied.acc_additive_penalties.to(dtype)[:,None,:]);expected=expected.reshape(bs*width,8)
    before=logits.clone();after=logits.clone();old_apply(next_token_logits=before,sampling_info=copied,draft_token_num=width);new_apply(next_token_logits=after,sampling_info=copied,draft_token_num=width)
    assert torch.equal(after,expected)
    masked_rows=[i for i in range(bs) if i%2==0 and emitted<4]
    for i in range(bs):
     expected_id=0 if i in masked_rows else 1
     assert after.view(bs,width,8)[i].argmax(-1).tolist()==[expected_id]*width
    if masked_rows:assert not torch.equal(before,expected) and before.argmax(-1).tolist()==[1]*(bs*width)
    else:assert torch.equal(before,after)
    # The live-orchestrator path already worked; compatibility fix preserves it.
    eager=logits.clone();old_apply(next_token_logits=eager,sampling_info=info,draft_token_num=width);assert torch.equal(eager,expected)
    cases.append({'bs':bs,'verify_width':width,'dtype':str(dtype),'already_emitted':emitted,'masked_request_rows':masked_rows,'new_matches_actual_producer_penalty':True,'old_drops_penalty':bool(masked_rows),'live_orchestrator_path_preserved':True})
# A genuinely unadjusted forward remains eligible for its existing folded path.
plain=Forward(torch.ones(2,1),8,None);assert old_pred(plain) and new_pred(plain)
raw=torch.randn(12,8);a=raw.clone();b=raw.clone();old_apply(next_token_logits=a,sampling_info=plain,draft_token_num=6);new_apply(next_token_logits=b,sampling_info=plain,draft_token_num=6);assert torch.equal(a,b) and torch.equal(a,raw)
result={'status':'CPU_seam_regression_passed_requires_NPU_and_model_validation','torch':torch.__version__,'actual_producer_fields':{'acc_additive_penalties':True,'acc_linear_penalties':False},'actual_copy_drops_live_orchestrator':True,'original_noop_predicate_wrong_with_precomputed_additive':True,'fixed_consumers_apply_exact_actual_producer_penalty':True,'native_ignore_eos_still_masks_EOS_for_min_tokens':True,'cases':cases,'unadjusted_fastpath_and_logits_unchanged':True,'NPU_executed':False,'full_model_executed':False,'performance_gain_verified':False,'source_sha256':{str(p.relative_to(R)):sha(p) for p in [B,P,N,D/'CPU_regression.py',D/'original/dspark_verify.py',D/'original/dflash_utils.py',R/'compat/runtime/g130/dspark_verify.py',R/'compat/runtime/g130/dflash_utils.py']},'scope':'Actual producer/method/consumer bodies executed on CPU. Minimal request/orchestrator DTO, not fullframework import. Verifies overlap handoff/stop-mask broadcast and route predicate; not CUDA/NPU graph, position-dependent speculation-boundary, arbitrary scaling penalty, or whole-model feature certification.'}
out=R/'.work/sampling-contract/CPU_regression.json';out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['cases','source_sha256']}))
