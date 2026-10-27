"""Execute exact saved AST branches using CPU sentinels, never torch/framework.
Sentinels record control flow/dtypes/ordering; they do not implement tensor math.
"""
from pathlib import Path
import ast,copy,csv,hashlib,json,platform,time,types,builtins
D=Path(__file__).resolve().parent;S=D/'installed-source';facts=[];checks=[]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def node(rel,owner,name):
 p=S/rel;t=ast.parse(p.read_text());scope=t if owner is None else next(n for n in t.body if isinstance(n,ast.ClassDef) and n.name==owner)
 ns=[n for n in scope.body if isinstance(n,ast.FunctionDef) and n.name==name];n=ns[-1]
 facts.append(dict(path='installed-source/'+rel,sha256=sha(p),symbol=(owner+'.'if owner else '')+name,line=n.lineno,end_line=n.end_lineno,definitions_in_scope=len(ns),definition_selected='last'))
 return copy.deepcopy(n)
def run(n,env=None,base=None):
 n.decorator_list=[];body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]
 if base:body.append(ast.ClassDef(name='Saved',bases=[ast.Name(id='Base',ctx=ast.Load())],keywords=[],body=[n],decorator_list=[]))
 else:body.append(n)
 g=dict(env or {});g['Base']=base
 exec(compile(ast.fix_missing_locations(ast.Module(body=body,type_ignores=[])),'<exact-saved-source>', 'exec'),g)
 return g['Saved'] if base else g[n.name]
def check(name,ok):
 assert ok,name
 checks.append(dict(name=name,passed=True))
receipts=[]
for name in ['download.json','extra-download.json','override-download.json']:
 x=json.loads((S/name).read_text())
 for row in x['files']:
  p=S/row['relative_path'];assert sha(p)==row['sha256'] and p.stat().st_size==row['bytes']
  receipts.append(dict(path='installed-source/'+row['relative_path'],sha256=row['sha256'],installed_equals_git_HEAD=row.get('installed_equals_git_HEAD'),git_HEAD=row.get('git_HEAD')))

class DT:
 def __init__(self,name):self.name=name
 def __repr__(self):return self.name
bf16=DT('bfloat16');fp32=DT('float32');fp16=DT('float16');calls=[]
class T:
 def __init__(self,dtype=bf16,shape=(1,7),label='x'):self.dtype=dtype;self.shape=shape;self.label=label;self.is_cuda=False
 @property
 def T(self):calls.append('transpose');return self
 def to(self,dtype):calls.append(('cast',self.label,repr(dtype)));return T(dtype,self.shape,self.label)
 def bfloat16(self):return self.to(bf16)
 def float(self):return self.to(fp32)
 def copy_(self,other):calls.append(('copy',repr(other.dtype),repr(self.dtype)));return self
 def __getitem__(self,key):
  calls.append(('slice',str(key)));stop=key[-1].stop;return T(self.dtype,(self.shape[0],min(self.shape[1],stop)),self.label)
 def __truediv__(self,x):calls.append(('divide',x));return self
 def __mul__(self,x):calls.append(('multiply',x));return self
 def __rmul__(self,x):return self*x
 def __imul__(self,x):calls.append(('scale',x));return self
 def mul_(self,x):calls.append(('scale',x));return self
def matmul(x,w):calls.append(('matmul',repr(x.dtype),repr(w.dtype)));return T(x.dtype)
torch=types.SimpleNamespace(dtype=DT,float32=fp32,float=fp32,bfloat16=bf16,float16=fp16,matmul=matmul,tanh=lambda x:(calls.append('tanh')or x))
env=dict(torch=torch,current_platform=types.SimpleNamespace(supported_dtypes=[bf16,fp32],is_cuda=lambda:False,is_rocm=lambda:False),_STR_DTYPE_TO_TORCH_DTYPE={'float32':fp32,'bfloat16':bf16})
f=run(node('vllm/config/model.py',None,'_get_head_dtype'),env)
for override,runner,expected in [(None,'generate',bf16),(None,'pooling',fp32),('model','generate',bf16),('float32','generate',fp32)]:
 cfg=types.SimpleNamespace(**({'head_dtype':override}if override is not None else {}))
 check('vLLM configured head dtype %s/%s'%(override,runner),f(cfg,bf16,runner)is expected)

class UnquantizedEmbeddingMethod:
 def apply(self,*a,**kw):calls.append('quant_apply');return T()
class UnquantizedLinearMethod:
 def apply(self,*a,**kw):calls.append('quant_apply');return T()
def linear(x,w,b):calls.append(('linear',repr(x.dtype),repr(w.dtype),repr(b.dtype)if b else None));return T(x.dtype)
apply=run(node('vllm/model_executor/layers/logits_processor.py','LogitsProcessor','_apply_head'),dict(env,F=types.SimpleNamespace(linear=linear),UnquantizedEmbeddingMethod=UnquantizedEmbeddingMethod,UnquantizedLinearMethod=UnquantizedLinearMethod))
for method in [UnquantizedEmbeddingMethod(),UnquantizedLinearMethod()]:
 for dtype in [None,bf16,fp32]:
  calls.clear();out=apply(types.SimpleNamespace(head_dtype=dtype),types.SimpleNamespace(quant_method=method,weight=T(label='weight')),T(label='hidden'),None)
  check('vLLM apply %s head=%s'%(type(method).__name__,dtype),calls==['quant_apply']if dtype in [None,bf16]else ('linear','float32','float32',None)in calls and out.dtype is fp32)
# Deliberately not an unquantized subclass for the rejection fixture.
calls.clear()
try:apply(types.SimpleNamespace(head_dtype=fp32),types.SimpleNamespace(quant_method=object(),weight=T()),T(),None)
except ValueError:rejected=True
else:rejected=False
check('vLLM different head dtype rejects incompatible quant method',rejected)
base=type('OriginalBase',(),{'_apply_head':apply})
C=run(node('vllm_ascend/ops/vocab_parallel_embedding.py','AscendLogitsProcessor','_apply_head'),base=base)
obj=C();obj.head_dtype=bf16;calls.clear();obj._apply_head(types.SimpleNamespace(quant_method=UnquantizedEmbeddingMethod()),T(),None)
check('Ascend override delegates actual base head path',calls==['quant_apply'])

vf=run(node('vllm/model_executor/layers/logits_processor.py','LogitsProcessor','forward'),env)
for soft,scale in [(None,1.0),(2.0,1.0),(None,0.5),(2.0,0.5)]:
 calls.clear();t=T();out=vf(types.SimpleNamespace(logits_as_input=True,soft_cap=soft,scale=scale),None,t)
 expected=([('divide',soft),'tanh',('multiply',soft)]if soft is not None else [])+([('scale',scale)]if scale!=1 else [])
 check('vLLM softcap/scale=%s/%s'%(soft,scale),out is t and calls==expected)

sgrel='sglang/srt/layers/logits_processor.py';tree=ast.parse((S/sgrel).read_text())
const=next(n.value for n in tree.body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='_UNQUANTIZED_LM_HEAD_METHODS'for t in n.targets))
names=ast.literal_eval(const)
should=run(node(sgrel,None,'should_apply_lm_head_quant_method'),dict(torch=torch,_UNQUANTIZED_LM_HEAD_METHODS=names))
for name,expected in [('UnquantizedEmbeddingMethod',False),('UnquantizedLinearMethod',False),('PackWeightMethod',False),('ModelSlimLinearMethod',True)]:
 m=type(name,(),{'apply':lambda *a:None})();check('SG effective last helper %s'%name,should(types.SimpleNamespace(weight=T()),m)==expected)
compute=run(node(sgrel,'LogitsProcessor','_compute_lm_head'),dict(torch=torch,should_apply_lm_head_quant_method=should,use_intel_amx_backend=lambda x:False))
for fp in [False,True]:
 for policy in [None,'policy']:
  calls.clear();h=types.SimpleNamespace(weight=T(label='weight'),quant_method=UnquantizedEmbeddingMethod())
  out=compute(types.SimpleNamespace(use_fp32_lm_head=fp,rl_on_policy_target=policy),T(label='hidden'),h)
  check('SG plain head fp32=%s policy=%s'%(fp,policy),('matmul','float32','float32')in calls if fp else ('matmul','bfloat16','bfloat16')in calls)
calls.clear();h=types.SimpleNamespace(weight=T(),quant_method=type('ModelSlimLinearMethod',(),{'apply':lambda *a:(calls.append('modelslim_apply')or T())})())
compute(types.SimpleNamespace(use_fp32_lm_head=True,rl_on_policy_target=None),T(),h)
check('SG quant method takes priority over plain-head fp32 branch',calls==['modelslim_apply'])
copybuf=run(node(sgrel,'LogitsProcessor','_copy_logits_to_buffer'),dict(torch=torch))
for buffer_shape in [None,(1,5),(2,5)]:
 calls.clear();buf=T(fp32,buffer_shape,'buffer')if buffer_shape else None
 out=copybuf(types.SimpleNamespace(vocab_size=5),T(shape=(1,7)),types.SimpleNamespace(next_token_logits_buffer=buf))
 check('SG crop and fp32 buffer branch %s'%(buffer_shape,),out.shape==(1,5) and out.dtype is fp32 and (out is buf if buffer_shape==(1,5)else ('cast','x','float32')in calls))

getlog=run(node(sgrel,'LogitsProcessor','_get_logits'),dict(torch=torch,_is_npu=True,_is_cpu=False,_trace_e2e_logits=lambda *a,**kw:None,envs=types.SimpleNamespace(SGLANG_TRACE_LOGITS_E2E_SYNC=types.SimpleNamespace(get=lambda:False))))
for soft,scale in [(None,None),(2.0,None),(None,0.5),(2.0,0.5)]:
 calls.clear();t=T();obj=types.SimpleNamespace(do_tensor_parallel_all_gather_dp_attn=False,do_tensor_parallel_all_gather=True,use_attn_tp_group=False,logit_scale=scale,final_logit_softcapping=soft,_gather_dp_attn_hidden_states=lambda x,m:(x,x),_compute_lm_head=lambda *a:(calls.append('project')or t),_can_use_tp_lm_head_all_to_all=lambda *a:False,_logits_gatherer=lambda x:(calls.append('gather')or x),_scatter_dp_attn_logits=lambda x,*a:(calls.append('scatter')or x),_copy_logits_to_buffer=lambda x,*a,**kw:(calls.append('fp32buffer')or x))
 getlog(obj,T(),object(),object())
 expected=['project']+([('scale',scale)]if scale is not None else [])+['gather','scatter','fp32buffer']+([('divide',soft),'tanh',('multiply',soft)]if soft else [])
 check('SG source order project/scale/gather/buffer/softcap %s/%s'%(soft,scale),calls==expected)

# Exact modelslim classification for an actual ParallelLMHead class hierarchy.
class LinearBase:pass
class FusedMoE:pass
class VocabParallelEmbedding:pass
class ParallelLMHead(VocabParallelEmbedding):pass
def fake_import(name,globals=None,locals=None,fromlist=(),level=0):
 if name=='sglang.srt.layers.linear':return types.SimpleNamespace(LinearBase=LinearBase)
 if name=='sglang.srt.layers.moe.fused_moe_triton':return types.SimpleNamespace(FusedMoE=FusedMoE)
 return builtins.__import__(name,globals,locals,fromlist,level)
sgquant=run(node('sglang/srt/layers/quantization/modelslim/modelslim.py','ModelSlimConfig','get_quant_method'),{'__builtins__':dict(vars(builtins),__import__=fake_import)})
check('SG ParallelLMHead is not LinearBase; modelslim returns None for embedding fallback',sgquant(None,ParallelLMHead(),'lm_head')is None)
qtype=run(node('vllm_ascend/quantization/configs/modelslim_config.py',None,'get_quant_type_for_layer'))
for value,expected in [('FLOAT',None),('W8A8_DYNAMIC','W8A8_DYNAMIC')]:
 check('vLLM modelslim head quant type %s'%value,qtype({'lm_head.weight':value},'lm_head')==expected)

for rel,owner,name in [
 ('vllm/model_executor/models/qwen3_5.py','Qwen3_5ForCausalLMBase','__init__'),('vllm/model_executor/models/qwen3_5.py','Qwen3_5ForCausalLMBase','compute_logits'),('vllm/model_executor/models/qwen3_vl.py','Qwen3VLForConditionalGeneration','compute_logits'),('vllm/model_executor/layers/logits_processor.py','LogitsProcessor','__init__'),('vllm_ascend/ops/vocab_parallel_embedding.py','AscendLogitsProcessor','_get_logits'),('vllm_ascend/ops/vocab_parallel_embedding.py','AscendLogitsProcessor','_get_logits_normal'),('vllm_ascend/ops/vocab_parallel_embedding.py','AscendVocabParallelEmbedding','__init__'),('vllm_ascend/quantization/configs/modelslim_config.py','AscendModelSlimConfig','get_quant_method'),('sglang/srt/models/qwen3_5.py','Qwen3_5ForConditionalGeneration','__init__'),('sglang/srt/models/qwen3_vl.py','Qwen3VLForConditionalGeneration','__init__'),('sglang/srt/models/qwen3_vl.py','Qwen3VLForConditionalGeneration','forward'),('sglang/srt/layers/vocab_parallel_embedding.py','VocabParallelEmbedding','__init__'),(sgrel,'LogitsProcessor','__init__')]:node(rel,owner,name)
configs={}
for kind in ['vllm','sglang']:
 config=json.loads((S/kind/'mounted-model/config.json').read_text());q=json.loads((S/kind/'mounted-model/quant_model_description.json').read_text());text=config['text_config']
 configs[kind]={'config_sha256':sha(S/kind/'mounted-model/config.json'),'quant_description_sha256':sha(S/kind/'mounted-model/quant_model_description.json'),'architectures':config['architectures'],'dtype':text['dtype'],'vocab_size':text['vocab_size'],'hidden_size':text['hidden_size'],'tie_word_embeddings':text['tie_word_embeddings'],'head_dtype_field_present':('head_dtype'in config or 'head_dtype'in text),'softcap_field_present':('final_logit_softcapping'in config or 'final_logit_softcapping'in text),'lm_head_quant_description':q['lm_head.weight'],'model_quant_type':q['model_quant_type']}
check('mounted config/quant-description two-arm exact bytes',configs['vllm']==configs['sglang'])
check('mounted head FLOAT despite model W8A8_DYNAMIC',configs['vllm']['lm_head_quant_description']=='FLOAT'and configs['vllm']['model_quant_type']=='W8A8_DYNAMIC')
headers=json.loads((D/'mounted-head-metadata.json').read_text())
for kind in ['vllm','sglang']:
 h=headers['arms'][kind]['metadata'];q=h['tensor_headers'][0]
 check(kind+' safetensors head header BF16 shape',q['tensor_metadata']['dtype']=='BF16'and q['tensor_metadata']['shape']==[248320,5120]and not q['payload_read']and q['full_shard_sha256']is None)
 check(kind+' header index bound to received mounted index',h['index_sha256']==sha(S/kind/'mounted-model/quant_model_weights.safetensors.index.json'))
logs=json.loads((D/'saved-input-receipt.json').read_text())
for row in logs['files']:assert sha(D/row['path'])==row['sha256']
sglog=(D/'saved-inputs/sglang-serve.log').read_text()
args=[ast.literal_eval(l.split('server_args=',1)[1])for l in sglog.splitlines()if 'server_args='in l]
assert len(args)==1
sgflags={k:args[0][k]for k in ['dtype','quantization','enable_fp32_lm_head','rl_on_policy_target','enable_dp_lm_head','enable_tp_lm_head_all_to_all']}
check('saved SG diagnostic log bfloat16/modelslim and fp32 head disabled',sgflags['dtype']=='bfloat16'and sgflags['quantization']=='modelslim'and sgflags['enable_fp32_lm_head']is False and sgflags['rl_on_policy_target']is None)
vlog=(D/'saved-inputs/vllm-serve.log').read_text()
engine=[l for l in vlog.splitlines()if 'Initializing a V1 LLM engine'in l]
check('saved vLLM diagnostic engine bfloat16 Ascend TP4 log',len(engine)==1 and 'dtype=torch.bfloat16'in engine[0]and 'quantization=ascend'in engine[0]and 'tensor_parallel_size=4'in engine[0])
O=D/'overlay-lineage';oreceipt=json.loads((O/'download.json').read_text())
for row in oreceipt['files']:assert sha(O/row['relative_path'])==row['sha256']
manifest=json.loads((O/'runtime/sglang-manifest.json').read_text());binding=next(r for r in manifest['modules']if r['module']=='sglang.srt.models.qwen3_5')
check('SG manifest qwen3.5 original source matches installed source',binding['original_sha256']==sha(S/'sglang/srt/models/qwen3_5.py'))
check('SG manifest candidate matches saved overlay bytes',binding['candidate_sha256']==sha(O/'compat/qwen3_5.py'))
startup=[json.loads(p.read_text())for p in sorted((O/'startup-receipts').glob('*.json'))]
check('five saved gen67 startup qwen3.5 original/candidate identities',len(startup)==5 and all(r['original_sha256']==binding['original_sha256']and r['candidate_sha256']==binding['candidate_sha256']for r in startup))
original=ast.parse((S/'sglang/srt/models/qwen3_5.py').read_text());overlay=ast.parse((O/'compat/qwen3_5.py').read_text())
for owner in ['Qwen3_5ForCausalLM','Qwen3_5ForConditionalGeneration']:
 a=next(n for n in original.body if isinstance(n,ast.ClassDef)and n.name==owner);b=next(n for n in overlay.body if isinstance(n,ast.ClassDef)and n.name==owner)
 check(owner+' overlay retains class bases', [ast.dump(x,include_attributes=False)for x in a.bases]==[ast.dump(x,include_attributes=False)for x in b.bases])
 f=lambda t:next(n for n in t.body if isinstance(n,ast.FunctionDef)and n.name=='__init__')
 check(owner+' overlay retains exact constructor AST',ast.dump(f(a),include_attributes=False)==ast.dump(f(b),include_attributes=False))
check('SG saved manifest does not replace logits processor or qwen3_vl',not any(r['module']in ['sglang.srt.layers.logits_processor','sglang.srt.models.qwen3_vl']for r in manifest['modules']))
check('mounted config matches prior model-identity receipt SHA',configs['vllm']['config_sha256']=='191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab')
check('mounted quant description matches prior model-identity receipt SHA',configs['vllm']['quant_description_sha256']=='d3725f719225567a6cd0fd3e340860082e36c3643dfe0f32d64e3172547bf793')
out={'task_id':'Q27-FULL-MAP-20261025','generation':43,'verified_ns':time.time_ns(),'Host Architecture':platform.machine(),'scope':'Exact saved AST with CPU stdlib sentinels. No tensor numerics, framework import, NPU, model, service, quality or performance validation.','source_SHA_observations':len(receipts),'source_receipts':receipts,'checks':checks,'source_facts':facts,'mounted_config_facts':configs,'historical_lm_head_dtype_observed':False,'historical_logits_observed':False,'full_weight_SHA_verified':False,'cross_framework_numeric_cause_proven':False,'greedy_equivalence_passed':False,'optimization_adopted':False,'root_acceptance':'pending','overall_complete':False}
out['saved_SG_diagnostic_flags']=sgflags
out['overlay_lineage']={'SHA_observations':len(oreceipt['files']),'saved_gen67_qwen_startup_receipts':len(startup),'qwen3_5_original_sha256':binding['original_sha256'],'qwen3_5_candidate_sha256':binding['candidate_sha256'],'relevant_constructor_AST_unchanged':True,'head_modules_exhaustive_runtime_identity':False}
(D/'head-contract-audit.json').write_text(json.dumps(out,indent=2)+'\n')
with(D/'head-source-links.csv').open('w',newline='')as f:
 w=csv.DictWriter(f,fieldnames=list(facts[0]));w.writeheader();w.writerows(facts)
print(json.dumps({'checks':len(checks),'source_anchors':len(facts),'source_SHA_observations':len(receipts),'overall_complete':False}))
