"""Pure CPU AST/hook qualification; fake operators do not prove NPU numerics."""
from pathlib import Path
import ast,hashlib,importlib.util,json,os,subprocess,sys,types
ROOT=Path(__file__).resolve().parents[1]
HELPER=ROOT/'config/optimization-overlays/C32-target-graph-fullmesh-v1/python/glm53_C32_target_graph_fullmesh_v1.py'
SHA='0216476710ec9877501979d9d7012b35f827d2e6403ca35c4276ec53e5c9166a'
ENV={'GLM53_C32_TARGET_GRAPH_ALIGN':'1','GLM53_NODE':'36','DEEP_USE_MODE':'ops','HCCL_BUFFSIZE':'256','DEEPEP_HCCL_BUFFSIZE':'','SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK':'32','TASK_QUEUE_ENABLE':'0'}
class Tensor:
 def __init__(self,shape,dtype='torch.bfloat16'):self.shape=shape;self.dtype=dtype;self.device='npu:0'
 def item(self):raise AssertionError('Forbidden value read')
 def cpu(self):raise AssertionError('Forbidden CPU copy')
 def tolist(self):raise AssertionError('Forbidden value read')
 def __bool__(self):raise AssertionError('Forbidden tensor boolean')
class Base:
 def __init__(self,group):self.group=group;self.group_size=32;self.rank=3;self.group_name='fixture-owned-group'
def main():
 assert hashlib.sha256(HELPER.read_bytes()).hexdigest()==SHA
 os.environ.update(ENV);spec=importlib.util.spec_from_file_location('compat',HELPER);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);events=[];m.emit=lambda event,**kw:events.append({'event':event,**kw})
 source=next((ROOT/'runs/20261018T-C32-alignment-installed-source-v1/source/sglang').glob('*low_latency_strategy.py'));tree=ast.parse(source.read_text());cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='OpsLowLatencyCommStrategy');body=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),cls],type_ignores=[]);ast.fix_missing_locations(body)
 calls=[];out=[Tensor((8192,6144),'torch.int8'),Tensor((8192,),'torch.float32'),Tensor((8192,),'torch.int32'),Tensor((8,),'torch.int64'),Tensor((256,),'torch.int32'),None,None];combined=Tensor((32,6144))
 def dispatch(**kw):calls.append(('dispatch',kw));return tuple(out)
 def combine(**kw):calls.append(('combine',kw));return combined
 torch=types.SimpleNamespace(_check=lambda condition,message: condition or (_ for _ in ()).throw(AssertionError(message())))
 ns={'LowLatencyEPCommStrategy':Base,'register_low_latency_strategy':lambda name:lambda cls:cls,'torch':torch,'torch_npu':types.SimpleNamespace(npu_moe_distribute_dispatch_v2=dispatch,npu_moe_distribute_combine_v2=combine),'os':os}
 exec(compile(body,str(source),'exec'),ns);module=types.ModuleType(m.OPS);module.__file__=str(source);module.OpsLowLatencyCommStrategy=ns['OpsLowLatencyCommStrategy']
 class Loader:
  def exec_module(self,module):pass
 m.Loader(Loader()).exec_module(module)
 inst=module.OpsLowLatencyCommStrategy(None,object());assert inst.comm_alg=='fullmesh_v1'
 x=Tensor((32,6144));ids=Tensor((32,8),'torch.int32');weights=Tensor((32,8),'torch.float32');mask=Tensor((32,),'torch.bool')
 got=inst._npu_low_latency_dispatch(x,ids,256,quant_mode=2,comm_alg=inst.comm_alg,x_active_mask=mask,topk_weights=weights,num_max_dispatch_tokens_per_rank=32)
 assert calls[-1][1]['expert_scales'] is None and calls[-1][1]['comm_alg']=='fullmesh_v1' and calls[-1][1]['x_active_mask'] is mask and calls[-1][1]['quant_mode']==2
 result=inst._npu_low_latency_combine(x=Tensor((8192,6144)),topk_idx=ids,topk_weights=weights,assist_info_for_combine=out[2],ep_send_counts=out[4],num_experts=256,comm_alg=inst.comm_alg,expand_scales=None,x_active_mask=mask,num_max_dispatch_tokens_per_rank=32)
 assert result is combined and calls[-1][1]['expert_scales'] is weights and calls[-1][1]['expand_scales'] is None
 before=len(calls)
 try:inst._npu_low_latency_dispatch(Tensor((128,6144)),ids,256,comm_alg=inst.comm_alg,x_active_mask=mask,topk_weights=weights,num_max_dispatch_tokens_per_rank=32)
 except AssertionError:pass
 else:raise AssertionError('Oversized dispatch admitted')
 assert len(calls)==before
 class Draft:
  def _capture_cuda_graphs(self):raise AssertionError('Draft graph should stay eager')
 draft=types.SimpleNamespace(EagleDraftWorker=Draft);m.install_draft(draft);dw=Draft();dw.target_config=object();marker=dw.target_config;dw._capture_cuda_graphs();assert dw.cuda_graph_runner is None and dw.cuda_graph_runner_for_draft_extend is None and dw.target_config is marker
 sentinel=object()
 class Graph:
  def can_run_graph(self,b):return b.admit
  def execute(self,b,*args,**kw):return sentinel
 graph=types.SimpleNamespace(DecodeCudaGraphRunner=Graph);m.install_graph(graph);gr=Graph();gr.model_runner=types.SimpleNamespace(is_draft_worker=False);gr._replay_graph_key=types.SimpleNamespace(size=2);gr.captured_req_width=6;gr.backend=object()
 for admitted in [True,False]:
  b=types.SimpleNamespace(admit=admitted,batch_size=2,forward_mode=types.SimpleNamespace(name='TARGET_VERIFY'));assert gr.can_run_graph(b) is admitted
 assert gr.execute(b) is sentinel
 assert any(x['event']=='target_graph_decision' and x['eligible'] is False for x in events)
 bad=types.ModuleType(m.OPS);bad.__file__=str(HELPER)
 try:m.install(bad)
 except AssertionError:pass
 else:raise AssertionError('Source mismatch admitted')
 old=os.environ['HCCL_BUFFSIZE'];os.environ['HCCL_BUFFSIZE']='1024'
 try:m.gate()
 except AssertionError:pass
 else:raise AssertionError('Unsupported window admitted')
 os.environ['HCCL_BUFFSIZE']=old
 code="import importlib.util,os,sys; p="+repr(str(HELPER))+";s=importlib.util.spec_from_file_location('fresh',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);m.activate();assert not any(k.startswith(('torch','sglang','deep_ep','memcache_hybrid','memfabric_hybrid')) for k in sys.modules);print('FRESH_IMPORT_HOOK_PASS')"
 child=subprocess.run([sys.executable,'-c',code],env=os.environ.copy(),capture_output=True,text=True,timeout=20);assert child.returncode==0,(child.stdout,child.stderr)
 v={'project_id':'glm53','pass':True,'helper_sha256':SHA,'actual_original_class_source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'original_AST_class_executed_with_fake_operators':True,'dispatch_expert_scales_none':True,'combine_original_weights_identity_retained':True,'oversize_fails_before_operator':True,'draft_eager_target_config_unchanged':True,'graph_results_and_ineligible_decisions_preserved':True,'source_and_window_mismatch_fail_closed':True,'fresh_process_import_hook_pass':True,'SDK_init_calls':0,'model_requests':0,'NPU_calls':0,'tensor_value_reads':0,'NPU_numerical_correctness_proven':False,'native_qualification_required':True}
 (ROOT/'results/C32-target-graph-fullmesh-local-compatibility-v1.json').write_text(json.dumps(v,indent=2)+'\n');print(json.dumps(v))
if __name__=='__main__':main()
