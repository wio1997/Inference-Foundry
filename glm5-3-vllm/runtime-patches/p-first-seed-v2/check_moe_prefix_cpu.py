"""CPU protocol checks for graph family/pool ownership and fail-closed selection."""
from pathlib import Path
from types import SimpleNamespace as S
import importlib.util,json,os,tempfile
os.environ['GLM53_P_FIRST_DRAFT_EXPORT']='0'
p=Path(__file__).with_name('moe_prefix_histogram.py');spec=importlib.util.spec_from_file_location('prefix_under_test',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class Mode:NONE=0;FULL=1
class Tensor:
 def __init__(self,n=6):self.shape=(n,6144);self.ndim=2;self.dtype='bf16';self.device=S(type='npu')
def config():
 return S(parallel_config=S(tensor_parallel_size=16,data_parallel_size=1,pipeline_parallel_size=1,decode_context_parallel_size=1,prefill_context_parallel_size=1),model_config=S(use_mla=True,is_hybrid=False),kv_transfer_config=S(kv_role='kv_consumer'),speculative_config=S(num_speculative_tokens=5),scheduler_config=S(max_num_seqs=4),lora_config=None)
class Desc:
 def __init__(self,n):self.num_tokens=n
 def __eq__(self,o):return isinstance(o,Desc) and self.num_tokens==o.num_tokens
 def __hash__(self):return hash(self.num_tokens)
checks=[]
def setup(root):
 ctx=S(batch_descriptor=Desc(6),cudagraph_runtime_mode=Mode.FULL,capturing=False,device_metadata_executor=None)
 conf=config();inp=S(hidden_states=Tensor(),quant=S(quant_type='W8A8'),routing=S(expert_map=None,global_redundant_expert_num=0))
 class Disp:
  def __init__(self):self.lora_context=None;self.num_experts_local=256;self.top_k=8;self.fail=False
  def token_dispatch(self,inp):return S(group_list_type=1)
 disp=Disp()
 def candidate(self,inp):
  if self.fail:raise RuntimeError('candidate-capture-failure')
  return S(group_list_type=0)
 class SFA:pass
 class Replay:
  def __init__(self,mode):self.mode=mode;self.tasks=[];self.fail=False
  def replay(self):
   if self.fail:raise RuntimeError('replay-failure')
 class Graph:
  def __init__(self):
   self.runtime_mode=Mode.FULL;self.attn_backend=SFA;self.vllm_config=conf;self.enable_enpu=False;self.graph_pool='nativepool';self.concrete_aclgraph_entries={};self.capture_modes=[]
  def __call__(self):
   d=ctx.batch_descriptor
   if d not in self.concrete_aclgraph_entries:
    ctx.capturing=True;out=None
    for _ in range(75):out=disp.token_dispatch(inp)
    self.capture_modes.append(out.group_list_type);out.graph=Replay(out.group_list_type);self.concrete_aclgraph_entries[d]=S(aclgraph=out.graph,output=out)
    return out
   entry=self.concrete_aclgraph_entries[d];entry.aclgraph.replay();return entry.output
 graph=Graph()
 class Runner:
  def __init__(self):
   self.is_kv_consumer=True;self.is_kv_producer=False;self.use_async_scheduling=True;self.dynamic_eplb=False;self.vllm_config=conf;self.tp_rank=0;self.input_batch=S(req_ids=['real']);self.requests={'real':S(sampling_params=S(max_tokens=64))};self.fail=False
  def execute_model(self):
   ctx.capturing=False
   if self.fail:raise RuntimeError('runner-failure')
   return graph()
  def sample_tokens(self):return [disp.token_dispatch(inp).group_list_type for _ in range(5)]
 runner=Runner()
 m.install({'torch':S(Tensor=Tensor,bfloat16='bf16',npu=S(graph_pool_handle=lambda:'privatepool')),'NPUModelRunner':Runner,'CUDAGraphMode':Mode,'enable_sp':lambda:False},root=root,deps={'Graph':Graph,'Dispatcher':Disp,'candidate':candidate,'get_context':lambda:ctx,'is_draft':lambda:False,'SFA':SFA,'updatable':lambda x:False,'W8A8':'W8A8','has_lora':lambda x:x is not None})
 return S(ctx=ctx,config=conf,inp=inp,disp=disp,graph=graph,runner=runner)
with tempfile.TemporaryDirectory() as tmp:
 root=Path(tmp);s=setup(root);native_dict=s.graph.concrete_aclgraph_entries
 assert s.graph().group_list_type==1 and s.graph.capture_modes==[1,1,0]
 assert s.graph.concrete_aclgraph_entries is native_dict and s.graph.graph_pool=='nativepool';checks.append('three-startup-families-owned-and-native-pool-restored')
 assert s.runner.execute_model().group_list_type==1 and s.runner.sample_tokens()==[1]*5;checks.append('default-native')
 (root/'enable').touch();assert s.runner.execute_model().group_list_type==0 and s.runner.sample_tokens()==[0]*5;checks.append('actual-target-and-MTP-candidate-selected')
 assert s.graph.concrete_aclgraph_entries is native_dict and s.disp.token_dispatch(s.inp).group_list_type==1;checks.append('outside-scope-native-and-entry-owner-restored')
 (root/'target-only').touch();assert s.runner.execute_model().group_list_type==0 and s.runner.sample_tokens()==[1]*5;(root/'target-only').unlink();checks.append('target-only-selector')
 (root/'MTP-only').touch();assert s.runner.execute_model().group_list_type==1 and s.runner.sample_tokens()==[0]*5;(root/'MTP-only').unlink();checks.append('MTP-only-selector')
 (root/'enable').unlink();(root/'replica-enable').touch();assert s.runner.execute_model().group_list_type==1 and s.runner.sample_tokens()==[1]*5;(root/'replica-enable').unlink();(root/'enable').touch();checks.append('native-replica-family-and-native-MTP')
 s.runner.requests['real'].sampling_params.max_tokens=1;assert s.runner.execute_model().group_list_type==1 and s.runner.sample_tokens()==[1]*5;checks.append('one-token-fallback')
 s.runner.requests['real'].sampling_params.max_tokens=64;s.inp.routing.expert_map=object();assert s.runner.sample_tokens()==[1]*5;checks.append('expert-map-fallback');s.inp.routing.expert_map=None
 s.runner.dynamic_eplb=True;assert s.runner.execute_model().group_list_type==1;checks.append('dynamic-eplb-runner-fallback');s.runner.dynamic_eplb=False
 s.runner.fail=True
 try:s.runner.execute_model();assert False
 except RuntimeError:pass
 s.runner.fail=False;assert s.disp.token_dispatch(s.inp).group_list_type==1;checks.append('runner-exception-restores-selector')
 before=next(root.glob('rank-*.jsonl')).stat().st_size;(root/'disabled').touch();assert s.runner.execute_model().group_list_type==0 and s.runner.sample_tokens()==[0]*5
 assert next(root.glob('rank-*.jsonl')).stat().st_size==before;checks.append('observer-off-keeps-candidate-without-logging')
 candidate_output=s.runner.execute_model();candidate_output.graph.fail=True
 try:s.runner.execute_model();assert False
 except RuntimeError:pass
 assert s.graph.concrete_aclgraph_entries is native_dict and s.disp.token_dispatch(s.inp).group_list_type==1
 candidate_output.graph.fail=False;checks.append('private-replay-exception-restores-native-family-and-selector')
 # Missing/partial candidate capture must fail closed, including exception cleanup.
 s.ctx.batch_descriptor=Desc(12);s.ctx.capturing=False;s.inp.hidden_states=Tensor(12);s.disp.fail=True
 try:s.graph();assert False
 except RuntimeError:pass
 assert s.graph.concrete_aclgraph_entries is native_dict and s.graph.graph_pool=='nativepool'
 s.disp.fail=False;assert s.runner.execute_model().group_list_type==1;checks.append('failed-candidate-capture-restores-pool-and-falls-back')
 # Fresh family captures rejected quantization as native, never mislabeled ready.
 s.ctx.batch_descriptor=Desc(18);s.ctx.capturing=False;s.inp.hidden_states=Tensor(18);s.inp.quant.quant_type='OTHER';s.graph()
 assert s.runner.execute_model().group_list_type==1;checks.append('non-W8A8-capture-not-selected')
 s.inp.quant.quant_type='W8A8';s.ctx.batch_descriptor=Desc(6);s.inp.hidden_states=Tensor(6)
 # Trigger a failure only in the private family's replay object.
 s.ctx.capturing=False
 original=native_dict[s.ctx.batch_descriptor].aclgraph
 # Candidate returned output remains a different object; native entry is untouched.
 assert s.runner.execute_model().group_list_type==0 and native_dict[s.ctx.batch_descriptor].aclgraph is original;checks.append('native-graph-object-preserved-across-mode-switch')
print(json.dumps({'completed':True,'passed':len(checks),'checks':checks,'scope':'CPU protocol/failure/ownership checks only; NPU routing/graph fragment evidence is separate.'}))
