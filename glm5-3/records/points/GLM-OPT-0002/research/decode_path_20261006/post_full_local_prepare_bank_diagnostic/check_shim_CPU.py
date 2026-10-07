"""Actual diagnostic AST with real mmap/filesystem and device doubles only."""
from pathlib import Path
import ast,json,sys,types,tempfile,os,hashlib
R=Path(__file__).resolve().parent
# Reuse actual-source lifecycle harness setup only, before its experiments/writes.
setup=(R.parent/'post_full_bank_sources/check_lifecycle_CPU.py').read_text()
setup=setup.split('for mode,bank in [(0,A),(1,B)]:')[0]
sys.modules['fixture']=types.ModuleType('fixture')
g={'__file__':str(R.parent/'post_full_bank_sources/check_lifecycle_CPU.py'),'__name__':'fixture'}
exec(compile(setup,'actual_lifecycle_setup','exec'),g)
ns=g['ns'];torch=g['torch'];state=g['state'];ctx=g['ctx'];log=g['log'];FULL=g['FULL'];NONE=g['NONE'];Tensor=g['Tensor'];Desc=g['Desc'];config=g['config'];runner=g['runner'];shared=g['shared']
ns['NPUModelRunner']=ns['TestRunner'];ns['get_tp_group']=lambda:types.SimpleNamespace(rank_in_group=0)
# Small value tensor double only for diagnostic byte/layout branch, not numeric proof.
Tensor.shape=property(lambda t:(t.n,));Tensor.dtype='fixture';Tensor.device='cpu-double'
Tensor.clone=lambda t:Tensor(t.n,value=t.value)
Tensor.detach=lambda t:t;Tensor.contiguous=lambda t:t;Tensor.cpu=lambda t:t
Tensor.stride=lambda t:(1,);Tensor.storage_offset=lambda t:0
torch.equal=lambda a,b:a.n==b.n and a.value==b.value
fake_npu=types.SimpleNamespace(get_npu_format=lambda t:2)
original_calls=[];candidate_calls=[];byte_calls=[]
class Prep:pass
class Candidate:pass
def make(t):return types.SimpleNamespace(hidden_states=t.clone(),router_logits=t.clone(),mc2_mask=t.clone(),padded_hidden_states_shape=t.shape,pertoken_scale=None)
def orig(self,h,r,*args):original_calls.append(1);return make(h)
def cand(self,h,r,*args):candidate_calls.append(1);return make(h)
prep=types.ModuleType('vllm_ascend.ops.fused_moe.prepare_finalize')
prep.__dict__.update(torch=torch,torch_npu=fake_npu,QuantType=types.SimpleNamespace(NONE=None),_H13_OVERRIDE=None,_H13_MAP=None,_H13_LAYOUTS={},_H13_ORIGINAL_PREPARE=orig,_H13Candidate=types.SimpleNamespace(_h13_candidate_prepare=cand),_EXTRA_CTX=types.SimpleNamespace(padded_num_tokens=16,mc2_mask=Tensor(2,value=7)))
def extract(path,names,target):
 nodes=[n for n in ast.parse(path.read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
 assert len(nodes)==len(names)
 m=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]+nodes,type_ignores=[])
 exec(compile(ast.fix_missing_locations(m),str(path),'exec'),target)
extract(R/'prepare_finalize.py',['_h13_control','_h13_layout','_h13_bytes_equal','_h13_prepare'],prep.__dict__)
raw_bytes=prep._h13_bytes_equal
def counted(a,b):byte_calls.append(1);return raw_bytes(a,b)
prep._h13_bytes_equal=counted
pkg=types.ModuleType('vllm_ascend.ops.fused_moe');pkg.prepare_finalize=prep;sys.modules[pkg.__name__]=pkg
fc=types.ModuleType('vllm.forward_context');fc.get_forward_context=lambda:ctx;sys.modules[fc.__name__]=fc
cfg=types.ModuleType('vllm.config');cfg.CUDAGraphMode=g['Modes'];sys.modules[cfg.__name__]=cfg
ns['_H13_NATIVE_CAPTURE_MODEL']=ns['TestRunner'].capture_model
extract(R/'model_runner_v1.py',['_h13_write','_h13_outputs','_H13BankSelector','_h13_capture_model'],ns)
# Real immutable witness writes; synthetic pool handles and tensors.
counter=iter(range(10));torch.npu.graph_pool_handle=lambda:('npu-pool',next(counter))
with tempfile.TemporaryDirectory() as td:
 root=Path(td);(root/'witnesses').mkdir();(root/'h13_mode.bin').write_bytes(bytes([0,1]));os.environ['GLM_H13_BANK_ROOT']=td
 def control(mode,observe):
  with (root/'h13_mode.bin').open('r+b') as f:f.write(bytes([mode,observe]));f.flush()
 def model(**kwargs):
  mode,_=prep._h13_control();state.mode=mode
  t=Tensor(2,value=7);prep._h13_prepare(Prep(),t,t)
  log.append(('raw_model',mode))
  if state.fail and state.graph is not None:raise RuntimeError('injected capture failure')
  out=Tensor(2,pool=state.graph.pool if state.graph else None,value=mode)
  if state.graph is not None:state.graph.ops.append(lambda:setattr(out,'value',mode))
  return out
 def dummy(n,cudagraph_runtime_mode,**kwargs):
  ctx.cudagraph_runtime_mode=cudagraph_runtime_mode;ctx.batch_descriptor=Desc();ctx.capturing=False
  return runner.model(input_ids=shared)
 runner._dummy_run=dummy;runner.model=ns['BreakableACLGraphWrapper'](model,config)
 ns['_h13_capture_model'](runner)
 selector=runner.model;assert isinstance(selector,ns['BreakableACLGraphWrapper']) and selector.unwrap() is model
 assert selector.banks[0].graph_pool!=selector.banks[1].graph_pool
 assert len(byte_calls)==12 # six byte comparisons in each of two NONE warmups
 capture_record=json.loads((root/'witnesses/h13_capture_rank0.json').read_text());assert len(capture_record['layouts'])==2
 raw_before=sum(v[0]=='raw_model' for v in log);bytes_before=len(byte_calls)
 outputs=[]
 for mode in [0,1,0,1]:
  control(mode,1);ctx.cudagraph_runtime_mode=FULL;outputs.append(selector(input_ids=shared));assert outputs[-1].value==mode
 assert outputs[0] is outputs[2] and outputs[1] is outputs[3] and outputs[0] is not outputs[1]
 assert sum(v[0]=='raw_model' for v in log)==raw_before and len(byte_calls)==bytes_before
 before_files={p.name:p.read_bytes() for p in (root/'witnesses').iterdir()}
 # Observe-off mode transitions replay but neither validate bytes nor write files.
 for mode in [0,1]:control(mode,0);ctx.cudagraph_runtime_mode=FULL;selector(input_ids=shared)
 assert {p.name:p.read_bytes() for p in (root/'witnesses').iterdir()}==before_files
 # NONE dispatch must use currently selected prepare without demanding FULL witness.
 for mode in [1,0]:
  control(mode,1);ctx.cudagraph_runtime_mode=NONE;assert selector(input_ids=shared).value==mode
 assert len(byte_calls)==bytes_before
 try:ns['_h13_write']('h13_capture_rank0.json',{});raise RuntimeError('overwrite accepted')
 except FileExistsError:pass
 # Invalid mode is rejected before model work.
 control(2,0)
 try:selector(input_ids=shared);raise RuntimeError('invalid mode accepted')
 except RuntimeError as e:assert 'Invalid H13' in str(e)
 # Failure during second bank capture propagates; selector is not installed.
 control(0,1);runner.model=ns['BreakableACLGraphWrapper'](model,config);state.fail=True
 # Inject only B: preserve A capture, then raise when override==1 inside graph.
 saved=model
 def fail_b(**kwargs):
  state.fail=prep._H13_OVERRIDE==1
  return saved(**kwargs)
 runner.model=ns['BreakableACLGraphWrapper'](fail_b,config)
 try:ns['_h13_capture_model'](runner);raise RuntimeError('failure swallowed')
 except RuntimeError as e:assert 'injected capture' in str(e)
 assert prep._H13_OVERRIDE is None and not isinstance(runner.model,ns['_H13BankSelector']) and state.enabled
 prep._H13_MAP.close();prep._H13_MAP=None
 del os.environ['GLM_H13_BANK_ROOT']
# Env-disabled module tail must not mutate original methods.
for filename,classname,method in [('model_runner_v1.py','NPUModelRunner','capture_model'),('prepare_finalize.py','PrepareAndFinalizeWithMC2','prepare')]:
 tree=ast.parse((R/filename).read_text());tail=tree.body[-1];assert isinstance(tail,ast.If)
 sentinel=object();cls=type('Fixture',(),{method:sentinel});scope={classname:cls}
 exec(compile(ast.Module(body=[tail],type_ignores=[]),filename,'exec'),scope);assert getattr(cls,method) is sentinel
result=dict(passed=True,actual_shim_AST=True,real_filesystem_mmap=True,tests=['two banks actual native capture wrapper contexts','immutable capture/layout witness','four runtime ABAB private outputs','no target Python reentry','NONE current prepare with observe1','observe0 no validation/files','FULL capture no byte D2H','invalid mode rejected','capture B failure propagates and clears override; lifecycle invalid','env-disabled methods unchanged'],numerical_validation='tensor doubles only; separate Torch CPU3888 is required numerical evidence',NPU_requests=0,torch_imported='torch' in sys.modules,torch_npu_imported='torch_npu' in sys.modules,source_sha256={n:hashlib.sha256((R/n).read_bytes()).hexdigest() for n in ['prepare_finalize.py','model_runner_v1.py']},limitations=['synthetic graph/pool/stream/tensor and weakref lifetime','no actual native format/capture/HCCL/device correctness','idle boundary must be enforced by external controller; selector cannot infer scheduler quiescence'])
(R/'shim_CPU_result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
