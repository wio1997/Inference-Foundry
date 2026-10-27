"""Execute exact saved-source branches with CPU stdlib sentinels, never torch/NPU.
Numeric NaN fixtures are Python floats, not tensor/backend equivalence tests.
"""
from pathlib import Path
import ast,copy,csv,hashlib,itertools,json,math,time,types
D=Path(__file__).resolve().parent;S=D/'installed-source';P=D.parent/'executor-run-gen38';facts=[];checks=[]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def node(rel,owner,name,base=S):
 p=base/rel;t=ast.parse(p.read_text());scope=t if owner is None else next(n for n in t.body if isinstance(n,ast.ClassDef)and n.name==owner)
 n=next(n for n in scope.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))and n.name==name)
 facts.append(dict(path=str(p.relative_to(D))if p.is_relative_to(D)else '../executor-run-gen38/'+str(p.relative_to(P)),sha256=sha(p),symbol=(owner+'.'if owner else '')+name,line=n.lineno,end_line=n.end_lineno));return copy.deepcopy(n)
def run(n,env=None,base=None):
 n.decorator_list=[];body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]
 if base:
  body.append(ast.ClassDef(name='SavedSubclass',bases=[ast.Name(id='SavedBase',ctx=ast.Load())],keywords=[],body=[n],decorator_list=[]))
 else:body.append(n)
 g=dict(env or {});g['SavedBase']=base
 exec(compile(ast.fix_missing_locations(ast.Module(body=body,type_ignores=[])),'<exact-saved-source>','exec'),g)
 return g['SavedSubclass']if base else g[n.name]
def check(name,condition):
 assert condition,name;checks.append({'name':name,'passed':True})
# All downloaded source bytes are SHA-bound; additional sources retain exact git comparison.
receipts=[]
for root,names in [(P/'installed-source',['download.json','ascend-download.json']),(S,['download.json','extra-download.json','ascend-runner-download.json'])]:
 for name in names:
  j=json.loads((root/name).read_text())
  for row in j['files']:
   p=root/row['relative_path'];assert sha(p)==row['sha256'];assert p.stat().st_size==row['bytes']
   receipts.append({'path':str(p.relative_to(D.parent)),'sha256':row['sha256'],'installed_equals_git_HEAD':row.get('installed_equals_git_HEAD'),'git_HEAD':row.get('git_HEAD')})
# build_logitsprocs original normal, pooling and speculative gates.
n=node('vllm/v1/sample/logits_processor/__init__.py',None,'build_logitsprocs')
tree=ast.parse((S/'vllm/v1/sample/logits_processor/__init__.py').read_text())
builtin=[n.id for a in tree.body if isinstance(a,ast.AnnAssign)and isinstance(a.target,ast.Name)and a.target.id=='BUILTIN_LOGITS_PROCESSORS' for n in a.value.elts];ctor=lambda name:(lambda *a:name)
env={'LogitsProcessors':list,'itertools':itertools,'BUILTIN_LOGITS_PROCESSORS':[ctor(k)for k in builtin], '_load_custom_logitsprocs':lambda custom:[ctor(k)for k in custom], 'MinTokensLogitsProcessor':ctor('MinTokens'),'logger':types.SimpleNamespace(debug=lambda *a:None,warning=lambda *a:None),'STR_POOLING_REJECTS_LOGITSPROCS':'pool','STR_SPEC_DEC_REJECTS_LOGITSPROCS':'spec'}
f=run(n,env);cfg=types.SimpleNamespace(speculative_config=None)
check('normal builder includes builtins plus explicit/plugin loader output',f(cfg,None,False,False,['Custom'])==builtin+['Custom'])
check('pooling builder empty',f(cfg,None,False,True)==[])
check('speculative builder min-tokens only',f(types.SimpleNamespace(speculative_config=True),None,False,False)==['MinTokens'])
# Exact watermark _sample_random: zero temperature and per-request opt-out return base.
class Vec(list):
 def __getitem__(self,i):return Vec(list.__getitem__(self,j)for j in i)if isinstance(i,list)else super().__getitem__(i)
 def __ne__(self,x):return Vec(z!=x for z in self)
 def __and__(self,x):return Vec(a and b for a,b in zip(self,x))
class Base:
 def _sample_random(self,*args):return 'ordinary'
class WatermarkReached(Exception):pass
def reach(*a):raise WatermarkReached()
n=node('vllm/v1/watermarking/gpu_sampler.py','GPUWatermarkSampler','_sample_random')
C=run(n,{'np':types.SimpleNamespace(any=any),'apply_top_k_top_p':reach},base=Base)
for enabled,temp,expected in [(True,0,'ordinary'),(False,1,'ordinary'),(True,1,'watermark'),(False,0,'ordinary')]:
 obj=C();obj.watermarking=types.SimpleNamespace(np=Vec([enabled]));obj.sampling_states=types.SimpleNamespace(temperature=types.SimpleNamespace(np=Vec([temp])))
 try:result=obj._sample_random(None,None,[0],None,None,None,False)
 except WatermarkReached:result='watermark'
 check('watermark gate enabled=%s temperature=%s'%(enabled,temp),result==expected)
# SG custom preprocessor invokes sanitize after optional custom action.
n=node('sglang/srt/layers/sampler.py','Sampler','_preprocess_logits')
for enabled in [False,True]:
 events=[];f=run(n,{'apply_custom_logit_processor':lambda *a:events.append('custom'),'sanitize_nan_logits':lambda *a:events.append('sanitize')})
 value=object();check('SG preprocess custom gate %s'%enabled,f(None,value,types.SimpleNamespace(has_custom_logit_processor=enabled))is value and events==(['custom','sanitize']if enabled else ['sanitize']))
# SG sanitizer conditional and exact nonfinite replacement constants.
n=node('sglang/srt/utils/async_probe.py',None,'sanitize_nan_logits')
for enabled in [False,True]:
 events=[];values=[float('nan'),float('inf'),-float('inf'),2.0]
 def nan_to_num_(v,**kw):
  events.append(dict(kw))
  for i,x in enumerate(v):
   if math.isnan(x):v[i]=kw['nan']
   elif x==math.inf:v[i]=kw['posinf']
   elif x==-math.inf:v[i]=kw['neginf']
 f=run(n,{'envs':types.SimpleNamespace(SGLANG_SANITIZE_NAN_LOGITS=types.SimpleNamespace(get=lambda:enabled)),'maybe_detect_nan':lambda *a:events.append('detect'),'maybe_warn_nan':lambda *a:events.append('warn'),'torch':types.SimpleNamespace(nan_to_num_=nan_to_num_)})
 f(values)
 check('SG sanitize enabled=%s'%enabled,(values==[-1e30,1e30,-1e30,2.0]and events==['detect','warn',dict(nan=-1e30,posinf=1e30,neginf=-1e30)])if enabled else math.isnan(values[0])and events==['detect'])
# SG TP sync is conditional MIN, never a sum/mean/first-rank broadcast.
n=node('sglang/srt/layers/sampler.py','Sampler','_sync_token_ids_across_tp')
for forced,grammar in itertools.product([False,True],repeat=2):
 calls=[];reduce=types.SimpleNamespace(MIN='MIN')
 fake=types.SimpleNamespace(all_reduce=lambda ids,**kw:calls.append(kw))
 f=run(n,{'SYNC_TOKEN_IDS_ACROSS_TP':forced,'torch':types.SimpleNamespace(distributed=fake),'dist':types.SimpleNamespace(ReduceOp=reduce)})
 f(types.SimpleNamespace(tp_sync_group='sameTPgroup'),[42],types.SimpleNamespace(grammars=['g']if grammar else None))
 check('SG TP sync force=%s grammar=%s'%(forced,grammar),calls==([{'op':'MIN','group':'sameTPgroup'}]if forced or grammar else []))
# Exact batch-info boolean expression, rather than a restated replica.
n=node('sglang/srt/sampling/sampling_batch_info.py','SamplingBatchInfo','from_schedule_batch')
expr=next(a.value for a in ast.walk(n)if isinstance(a,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='has_custom_logit_processor'for t in a.targets))
for enabled,request in itertools.product([False,True],repeat=2):
 result=eval(compile(ast.Expression(expr),'<saved-original-custom-gate>','eval'),{'get_exec':lambda:types.SimpleNamespace(features=types.SimpleNamespace(enable_custom_logit_processor=enabled)),'reqs':[types.SimpleNamespace(custom_logit_processor='processor'if request else None)]})
 check('SG custom batch service=%s request=%s'%(enabled,request),result==bool(enabled and request))
# Add exact function anchors for source order and downstream processor reachability.
for rel,owner,name,base in [('vllm/v1/sample/sampler.py','Sampler','forward',P/'installed-source'),('vllm/v1/sample/sampler.py','Sampler','apply_logits_processors',P/'installed-source'),('vllm/v1/sample/sampler.py','Sampler','sample',P/'installed-source'),('vllm_ascend/sample/sampler.py','AscendSampler','greedy_sample',P/'installed-source'),('vllm_ascend/worker/model_runner_v1.py','NPUModelRunner','_sample',S),('sglang/srt/model_executor/model_runner.py','ModelRunner','sample',S),('sglang/srt/model_executor/model_runner.py','ModelRunner','_preprocess_logits',S),('sglang/srt/sampling/sampling_batch_info.py','SamplingBatchInfo','apply_logits_bias',S)]:node(rel,owner,name,base)
# Additional exact early-return and ordering checks for possible greedy modifiers.
n=node('vllm/v1/sample/logits_processor/builtin.py','MinTokensLogitsProcessor','add_request')
f=run(n)
for minimum,produced,active in [(0,0,False),(2,0,True),(2,2,False)]:
 result=f(types.SimpleNamespace(min_tokens=minimum,all_stop_token_ids={7},structured_outputs=None),None,[1]*produced)
 check('vLLM min-tokens minimum=%s outputlen=%s'%(minimum,produced),(result is not None)==active)
n=node('vllm/v1/sample/sampler.py','Sampler','apply_penalties',P/'installed-source')
check('vLLM no-penalties early return preserves logits sentinel',run(n)(42,types.SimpleNamespace(no_penalties=True),[])==42)
n=node('sglang/srt/sampling/sampling_batch_info.py','SamplingBatchInfo','apply_logits_bias')
events=[]
f=run(n);obj=types.SimpleNamespace(_apply_pre_grammar_logits_transforms=lambda x:events.append('penalty'),grammar_mask=types.SimpleNamespace(apply=lambda x:events.append('grammar')),_apply_post_grammar_logits_transforms=lambda x:events.append('bias'))
f(obj,42);check('SG original penalty then grammar then bias order',events==['penalty','grammar','bias'])
check('Ascend runner source contains AscendSampler construction and no watermark spelling','self.sampler = AscendSampler()'in(S/'vllm_ascend/worker/model_runner_v1.py').read_text()and 'watermark'not in(S/'vllm_ascend/worker/model_runner_v1.py').read_text().lower())
# Saved logs support only the fields they actually contain.
sglog=(P/'saved-inputs/sglang-serve.log').read_text();check('historical SG server_args disables custom processor',"'enable_custom_logit_processor': False"in sglog)
out={'task_id':'Q27-FULL-MAP-20261025','generation':40,'verified_ns':time.time_ns(),'Host Architecture':__import__('platform').machine(),'scope':'CPU stdlib execution of exact saved AST with sentinels. No torch/NPU/imported framework; no historical runtime sampler telemetry.','source_files_SHA_verified':len(receipts),'source_receipts':receipts,'checks':checks,'source_facts':facts,'historical_sampler_binding_observed':False,'historical_logits_observed':False,'historical_env_observed':False,'vllm_sampled_IDs_available':False,'greedy_equivalence_passed':False,'optimization_adopted':False,'overall_complete':False}
(D/'logits-contract-audit.json').write_text(json.dumps(out,indent=2)+'\n')
with(D/'logits-source-links.csv').open('w',newline='')as f:
 w=csv.DictWriter(f,fieldnames=['path','sha256','symbol','line','end_line']);w.writeheader();w.writerows(facts)
print(json.dumps({'checks':len(checks),'source_facts':len(facts),'source_files_SHA_verified':len(receipts),'overall_complete':False}))
