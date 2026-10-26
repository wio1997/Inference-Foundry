"""Exact grammar/FSM/cache/MTP host bodies; real xgrammar CPU, explicit native boundaries."""
import ast,concurrent.futures as futures,ctypes,dataclasses,hashlib,importlib.util,json,logging,pathlib,sys,time,types,typing
from concurrent.futures import Future,ThreadPoolExecutor
from types import SimpleNamespace as NS
import torch,xgrammar as xg
from transformers import AutoTokenizer
R=pathlib.Path(__file__).resolve().parents[1];S=R/'sources/installed';records=[];mod=types.ModuleType('lf32_grammar');sys.modules[mod.__name__]=mod;ns=vars(mod);ns.update({k:v for k,v in vars(typing).items() if not k.startswith('_')});ns.update(torch=torch,dataclasses=dataclasses,dataclass=dataclasses.dataclass,field=dataclasses.field,json=json,logging=logging,time=time,futures=futures,Future=Future,ThreadPoolExecutor=ThreadPoolExecutor,ctypes=ctypes,logger=logging.getLogger('L032'))
def load(path,classes=(),functions=(),aliases=(),methods=None):
 p=S/path;t=ast.parse(p.read_text());nodes=[]
 for n in t.body:
  if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name in functions:nodes.append(n)
  elif isinstance(n,ast.ClassDef):
   if n.name in classes:nodes.append(n)
   elif methods and n.name in methods:
    body=[m for m in n.body if isinstance(m,(ast.FunctionDef,ast.AsyncFunctionDef)) and m.name in methods[n.name]];assert {m.name for m in body}==set(methods[n.name]);nodes.append(ast.ClassDef(name=n.name,bases=[],keywords=[],body=body,decorator_list=[]))
  elif isinstance(n,ast.Assign) and any(isinstance(z,ast.Name) and z.id in aliases for z in n.targets):nodes.append(n)
 for n in nodes:records.append({'path':path,'name':getattr(n,'name',ast.unparse(n.targets[0]) if isinstance(n,ast.Assign) else '?'),'AST_sha256':hashlib.sha256(ast.dump(n,include_attributes=False).encode()).hexdigest(),'complete_body':True})
 m=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]+nodes,type_ignores=[]);ast.fix_missing_locations(m);exec(compile(m,str(p),'exec'),ns)
def import_source(name,path):
 sp=importlib.util.spec_from_file_location(name,S/path);m=importlib.util.module_from_spec(sp);sys.modules[name]=m;sp.loader.exec_module(m);records.append({'path':path,'name':'WHOLE_MODULE','sha256':hashlib.sha256((S/path).read_bytes()).hexdigest()});return m
validation=import_source('lf32_json_validation','python/sglang/srt/constrained/json_schema_validation.py');ns.update(UnsupportedJSONSchemaFeature=validation.UnsupportedJSONSchemaFeature,validate_xgrammar_json_schema=validation.validate_xgrammar_json_schema)
load('python/sglang/srt/constrained/base_grammar_backend.py',classes=['GrammarStats','GrammarRow','BaseGrammarObject','GrammarMask','InvalidGrammarObject','BaseGrammarBackend'],functions=['_grammar_key_contains_nul'])
for n in ['CompiledGrammar','GrammarCompiler','GrammarMatcher','StructuralTag','StructuralTagItem','TokenizerInfo','bitmask_dtype','get_bitmask_shape']:ns[n]=getattr(xg,n)
class NativeBoundary(RuntimeError):pass
def native_boundary(*a,**kw):raise NativeBoundary('NPU/CUDA/Triton path unexecuted in CPU grammar fixture')
ns.update(_is_hip=False,is_pin_memory_available=lambda:False,apply_token_bitmask_inplace_cuda=native_boundary,apply_token_bitmask_inplace_triton=native_boundary,set_token_filter_triton=native_boundary)
load('python/sglang/srt/constrained/utils.py',functions=['is_legacy_structural_tag']);load('python/sglang/srt/constrained/torch_ops/token_filter_torch_ops.py',functions=['set_token_filter_torch'])
load('python/sglang/srt/constrained/xgrammar_backend.py',classes=['XGrammarGrammar','TokenizerNotSupportedError','XGrammarGrammarBackend'],functions=['_allocate_token_bitmask'],aliases=['MAX_ROLLBACK_TOKENS'])
# Reasoner wrapper and distributed callbacks are not selected by these fixture cases.
ns['ReasonerGrammarObject']=type('UnselectedReasonerBoundary',(),{})
load('python/sglang/srt/constrained/grammar_manager.py',classes=['GrammarManager'])
load('python/sglang/srt/sampling/sampling_batch_info.py',methods={'SamplingBatchInfo':['update_regex_vocab_mask']})
load('python/sglang/srt/speculative/spec_utils.py',classes=['GrammarTree'],functions=['traverse_tree','generate_token_bitmask','build_grammar_vocab_mask'],aliases=['TREE_TRAVERSE_TIME_THRESHOLD'])
Backend=ns['XGrammarGrammarBackend'];Grammar=ns['XGrammarGrammar'];Invalid=ns['InvalidGrammarObject'];Manager=ns['GrammarManager'];Tree=ns['GrammarTree'];Mask=ns['GrammarMask'];Sampling=ns['SamplingBatchInfo']
tokenizer=AutoTokenizer.from_pretrained(str(R/'fixtures/L030-exact-tokenizer'),local_files_only=True,trust_remote_code=False);V=len(tokenizer);EOS=tokenizer.eos_token_id
SCHEMA={'type':'object','properties':{'kind':{'type':'string','enum':['红','blue']},'n':{'type':'integer','minimum':0,'maximum':9},'items':{'type':'array','items':{'type':'boolean'},'minItems':1,'maxItems':2}},'required':['kind','n','items'],'additionalProperties':False}
KEY=('json',json.dumps(SCHEMA,ensure_ascii=False,separators=(',',':')))
DOCS=[{'kind':'红','n':2,'items':[True,False]},{'kind':'blue','n':9,'items':[False]},{'kind':'红','n':0,'items':[True]}]
def encode(s):return tokenizer.encode(s,add_special_tokens=False)
def doc_ids(d):return encode(json.dumps(d,ensure_ascii=False,separators=(',',':')))
def backend():return Backend(tokenizer,V,model_eos_token_ids=[EOS])
def snapshot(g):
 m=g.allocate_vocab_mask(V,1,'cpu');g.fill_vocab_mask(m,0);return m

def allowed(row):
 # Independent packed-int representation oracle, not xgrammar's mask apply.
 ids=torch.arange(V,dtype=torch.int64);return ((row.to(torch.int64)[ids//32] >> (ids%32)) & 1).bool()
def manager(b):
 m=Manager.__new__(Manager);m.grammar_backend=b;m.grammar_queue=[];m._enable_strict_thinking=False;m.scheduler=NS(model_config=NS(request_selectable_think_end_id_sequences=None));m.grammar_sync_size=1;m.grammar_sync_group=None;m.pp_rank=0;m.pp_size=1;m.pp_group=None;m.grammar_pp_sync_work_list=[];m.SGLANG_GRAMMAR_POLL_INTERVAL=.01;m.SGLANG_GRAMMAR_MAX_POLL_ITERATIONS=100;return m
class Request:
 def __init__(self,rid,key=KEY):
  self.rid=rid;self.sampling_params=NS(json_schema=key[1] if key[0]=='json' else None,regex=key[1] if key[0]=='regex' else None,ebnf=key[1] if key[0]=='ebnf' else None,structural_tag=None,ebnf_full_assistant=False,custom_params=None);self.require_reasoning=False;self.grammar=None;self.grammar_key=None;self.grammar_wait_ct=0;self.finish_reason=None
 def finished(self):return self.finish_reason is not None
 def set_finish_with_abort(self,message):self.finish_reason={'type':'abort','message':message}
BOUNDARIES=['Real xgrammar0.2.7/TVMFFI CPU compiler/matcher, Torch2.8 local; targetinstalledlibrary/Torch2.10 and sourceTorch2.13 declaration remain distinct','V=len exact historical tokenizer and EOS=tokenizer.eos_token_id are CPU fixture policies. Exactmodelconfig recovery SHA mismatched, so nativepaddedvocab/modelEOS remainsUNKNOWN','Whole source classes/selected methods compiled without heavy SGLang imports/factory/constructors; manager single-rank PP1/TP1 fixture, originalconstructor not executed','CPU allocator pin flag=False and _is_hip=False explicit host bindings; accidental GPU/Triton paths raise, no fake NPU success','Reasoner/strict-thinking/parser wrapper and fullSampler/RNG/model/nativeP/D not exercised; req.require_reasoning=False','GrammarTree.from_host and realCPU masks only; no D2H event/pinning/collective/graph timing or native mask apply','Controlled Future cancellation/timeout cases use real Future primitive with explicit compile-job boundary; no realmodel cancel/KV-reclaim evidence']
