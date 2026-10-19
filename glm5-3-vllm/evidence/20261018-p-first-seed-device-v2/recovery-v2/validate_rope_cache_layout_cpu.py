from pathlib import Path
import importlib.util,json,os,tempfile,types
import numpy as np
repo=Path('/Users/wio/work/Inference-Foundry-execution-plan');r=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('rope_layout',repo/'glm5-3-vllm/runtime-patches/p-first-seed-v2/rope_cache_layout.py');h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
os.environ['GLM53_P_FIRST_DRAFT_EXPORT']='0'
class Tensor:
 def __init__(self,data,shape=None,stride=(128,1),ptr=None):self.data=np.array(data,copy=True);self.shape=shape or (1048576,64);self._stride=stride;self.dtype='bf16';self.device=types.SimpleNamespace(type='npu');self.ptr=ptr or id(self)
 def stride(self):return self._stride
 def data_ptr(self):return self.ptr
 def is_contiguous(self):return self._stride==(64,1)
 def contiguous(self):return Tensor(self.data,shape=self.shape,stride=(64,1))
 def numel(self):return int(np.prod(self.shape))
 def element_size(self):return 2
class Module:
 def __init__(self,c,s):self._buffers={'_rope_derived_cos_cache':c,'_rope_derived_sin_cache':s};self.nonpersistent=set()
 def named_modules(self):return [('rotary',self)]
 def register_buffer(self,n,t,persistent):assert n not in self._buffers;self._buffers[n]=t;assert persistent is False;self.nonpersistent.add(n)
passed=[]
with tempfile.TemporaryDirectory() as temp:
 root=Path(temp)
 cos=Tensor(np.arange(128).reshape(2,64));sin=Tensor(-cos.data);owner=Module(cos,sin)
 rot=types.SimpleNamespace(_cos_cache=cos,_sin_cache=sin,_cos_mla=Tensor(np.zeros((2,64))),_sin_mla=Tensor(np.zeros((2,64))))
 fail={'value':False};selected=[]
 def original(positions,use_cache=False):
  selected.append((rot._cos_cache,rot._sin_cache))
  if fail['value']:raise RuntimeError('lookup error')
  c=rot._cos_cache.data[positions];s=rot._sin_cache.data[positions]
  if use_cache:
   rot._cos_mla.data[:]=c;rot._sin_mla.data[:]=s
   return rot._cos_mla,rot._sin_mla
  return Tensor(c,shape=(len(positions),1,1,64),stride=(64,1)),Tensor(s,shape=(len(positions),1,1,64),stride=(64,1))
 rot.get_cos_and_sin_mla=original
 rot._record_cos_and_sin_cache_interleaved=lambda owner,table:None
 class Runner:
  def __init__(self):
   self.parallel_config=types.SimpleNamespace(tensor_parallel_size=16,data_parallel_size=1,pipeline_parallel_size=1,decode_context_parallel_size=1,prefill_context_parallel_size=1)
   self.is_kv_consumer=True;self.is_kv_producer=False;self.use_async_scheduling=True;self.num_spec_tokens=5;self.model_config=types.SimpleNamespace(use_mla=True,is_hybrid=False);self.dynamic_eplb=False;self.drafter=types.SimpleNamespace(model=owner);self.input_batch=types.SimpleNamespace(req_ids=['rid']);self.requests={'rid':types.SimpleNamespace(sampling_params=types.SimpleNamespace(max_tokens=64))};self.tp_rank=0
  def get_model(self):return owner
  def execute_model(self,use_cache=False):return rot.get_cos_and_sin_mla(np.array([1,0]),use_cache)
  def sample_tokens(self):return rot.get_cos_and_sin_mla(np.array([1,0]),False)
 capturing={'value':False};ctx=lambda:types.SimpleNamespace(capturing=capturing['value'])
 h.install({'torch':types.SimpleNamespace(bfloat16='bf16',equal=lambda a,b:np.array_equal(a.data,b.data)),'NPUModelRunner':Runner,'enable_sp':lambda:False},root=root,deps={'rotary':rot,'ctx':ctx,'ctx_available':lambda:True})
 rot._record_cos_and_sin_cache_interleaved(owner,None)
 assert len(owner._buffers)==4
 runner=Runner();a=runner.execute_model();assert selected[-1]==(cos,sin) and len(owner._buffers)==4;passed.append('default off native source; buffers prepared before compilation')
 (root/'enable').touch();b=runner.execute_model();assert all(np.array_equal(x.data,y.data) for x,y in zip(a,b));assert all(t.is_contiguous() for t in selected[-1]);passed.append('candidate bitwise and contiguous')
 assert len(owner.nonpersistent)==2 and all(owner._buffers[n] is t for n,t in zip(['_glm53_rope_cos_contiguous','_glm53_rope_sin_contiguous'],selected[-1]));passed.append('derived buffers have native model ownership')
 assert rot._cos_cache is cos and rot._sin_cache is sin;passed.append('native globals restored after normal call')
 cached=selected[-1];runner.sample_tokens();assert selected[-1]==cached and len(owner._buffers)==4;passed.append('cached pair reused without new allocation')
 addresses=(rot._cos_mla.data_ptr(),rot._sin_mla.data_ptr());ret=runner.execute_model(True);assert ret==(rot._cos_mla,rot._sin_mla) and addresses==(rot._cos_mla.data_ptr(),rot._sin_mla.data_ptr());passed.append('persistent graph output pointers retained')
 fail['value']=True
 try:runner.execute_model()
 except RuntimeError:pass
 else:raise AssertionError('exception lost')
 assert rot._cos_cache is cos and rot._sin_cache is sin;fail['value']=False;passed.append('exception restores globals and frame')
 rot.get_cos_and_sin_mla(np.array([1,0]));assert selected[-1]==(cos,sin);passed.append('outside runner remains native')
 capturing['value']=True;runner.execute_model();assert selected[-1]==(cos,sin);capturing['value']=False;passed.append('capture fallback')
 runner.requests['rid'].sampling_params.max_tokens=1;runner.execute_model();assert selected[-1]==(cos,sin);runner.requests['rid'].sampling_params.max_tokens=64;passed.append('one token request fallback')
 runner.parallel_config.tensor_parallel_size=8;runner.execute_model();assert selected[-1]==(cos,sin);runner.parallel_config.tensor_parallel_size=16;passed.append('parallel guard')
 (root/'enable').unlink();runner.execute_model();assert selected[-1]==(cos,sin);passed.append('native restored after candidate')
 (root/'enable').touch();replacement=Tensor(cos.data);rot._cos_cache=replacement;runner.execute_model();assert selected[-1]==(replacement,sin);rot._cos_cache=cos;passed.append('native cache replacement invalidates selected cache')
 before=sum(len(p.read_text().splitlines()) for p in root.glob('rank-*.jsonl'));(root/'disabled').touch();runner.execute_model();after=sum(len(p.read_text().splitlines()) for p in root.glob('rank-*.jsonl'));assert before==after and selected[-1]==cached;passed.append('observer off preserves candidate')
result={'completed':True,'checks':passed,'count':len(passed),'scope':'CPU interface/ownership/guard/error-path checks; real NPU bitwise/layout cost is in rope-cache-layout-fragment-v3, full model path and service gain still pending.'};(r/'rope-cache-layout-cpu-check-v1.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
