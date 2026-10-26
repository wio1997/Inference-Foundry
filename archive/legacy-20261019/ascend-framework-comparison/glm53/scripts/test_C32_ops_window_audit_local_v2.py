"""Pure stubs: execute actual native adapter methods with fake NPU calls; no framework imports."""
import ast, contextlib, hashlib, importlib.util, io, json, os, tempfile, types
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
AUDIT_SHA='2e3ba957fab37574fd69ed76fdfeb71c3fe18acb08adb8d0407f4d83a7d24177'
def main():
 path=ROOT/'scripts/glm53_C32_ops_window_audit_v2.py';assert hashlib.sha256(path.read_bytes()).hexdigest()==AUDIT_SHA
 spec=importlib.util.spec_from_file_location('audit_fixture_target',path);audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
 wire=json.loads((ROOT/'runs/20261017T2062-sglang-C32-ops-prerequisites-v1/node-36.json').read_text());capture=json.loads(wire['stdout'])
 entry=next(x for x in capture['files'] if x['path'].endswith('/deep_ep/strategies/low_latency_strategy.py'));source=entry['source'];assert hashlib.sha256(source.encode()).hexdigest()==audit.SOURCE_SHA
 tree=ast.parse(source);cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='OpsLowLatencyCommStrategy');cls.bases=[];cls.decorator_list=[]
 tree=ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),cls],type_ignores=[]))
 class Tensor:
  def __init__(self,shape,dtype='torch.bfloat16'):self.shape=tuple(shape);self.dtype=dtype;self.device='npu:0'
  def cpu(self):raise AssertionError('Forbidden value download')
  item=tolist=numpy=cpu
 cases=[];old_env=dict(os.environ)
 try:
  with tempfile.TemporaryDirectory(prefix='glm53-C32-ops-stubs-') as tmp:
   native=Path(tmp)/'low_latency_strategy.py';native.write_text(source)
   for rank in range(32):
    for quant_mode in [0,2]:
     calls=[];x=Tensor([128,6144]);idx=Tensor([128,8],'torch.int32');weights=Tensor([128,8],'torch.float32');mask=Tensor([128],'torch.bool')
     outputs=(Tensor([32768,6144],'torch.int8' if quant_mode==2 else 'torch.bfloat16'),Tensor([32768],'torch.float32'),Tensor([32768*128],'torch.int32'),Tensor([8],'torch.int64'),Tensor([256],'torch.int32'),None,None)
     combined=Tensor([128,6144])
     def dispatch(**kw):calls.append(('dispatch',kw));return outputs
     def combine(**kw):calls.append(('combine',kw));return combined
     module=types.ModuleType('native_adapter_fixture');module.__file__=str(native);module.__dict__.update({'os':os,'torch_npu':types.SimpleNamespace(npu_moe_distribute_dispatch_v2=dispatch,npu_moe_distribute_combine_v2=combine)})
     exec(compile(tree,str(native),'exec'),module.__dict__)
     os.environ.update({'GLM53_C32_OPS_AUDIT':'1','GLM53_NODE':'36' if rank<16 else '41','DEEP_USE_MODE':'ops','HCCL_BUFFSIZE':'1024','DEEPEP_HCCL_BUFFSIZE':'1024'})
     installed=audit.install(module);obj=object.__new__(module.OpsLowLatencyCommStrategy);obj.group_name='fixture-EP';obj.group_size=32;obj.rank=rank
     out=io.StringIO()
     with contextlib.redirect_stdout(out):
      result=obj._npu_low_latency_dispatch(x,idx,256,quant_mode=quant_mode,comm_alg='hierarchy',x_active_mask=mask,topk_weights=weights,num_max_dispatch_tokens_per_rank=128)
      assert len(result)==6 and all(a is b for a,b in zip(result,[outputs[0],outputs[1],outputs[3],outputs[2],outputs[4],outputs[6]]))
      merged=obj._npu_low_latency_combine(result[0],idx,weights,result[3],result[4],num_experts=256,comm_alg='hierarchy',x_active_mask=mask,expand_scales=result[5],num_max_dispatch_tokens_per_rank=128)
      assert merged is combined
      again=obj._npu_low_latency_combine(result[0],idx,weights,result[3],result[4],num_experts=256,comm_alg='hierarchy',x_active_mask=mask)
      assert again is combined
     rows=[json.loads(line.split('=',1)[1]) for line in out.getvalue().splitlines()];assert len(rows)==2 and all(r['rank']==rank and r['group_size']==32 and r['original_method_returned'] and r['HCCL_BUFFSIZE_environment']==r['DEEPEP_HCCL_BUFFSIZE_environment']=='1024' for r in rows)
     for kind,kw in calls:
      assert kw['ep_rank_id']==rank and kw['ep_world_size']==32 and kw['group_ep']=='fixture-EP' and kw['shared_expert_num']==kw['shared_expert_rank_num']==0 and kw['comm_alg']=='hierarchy'
      assert kw['expert_ids'] is idx and kw['expert_scales'] is weights and kw['x_active_mask'] is mask
      if kind=='dispatch':assert kw['x'] is x and kw['quant_mode']==quant_mode and kw['expert_token_nums_type']==1
      else:assert kw['expand_x'] is outputs[0] and kw['assist_info_for_combine'] is outputs[2] and kw['ep_send_counts'] is outputs[4] and kw['comm_quant_mode']==0
     def fail(**kw):raise ValueError('native failure remains visible')
     module.torch_npu.npu_moe_distribute_combine_v2=fail
     try:obj._npu_low_latency_combine(result[0],idx,weights,result[3],result[4],num_experts=256)
     except ValueError as error:assert str(error)=='native failure remains visible'
     else:raise AssertionError('Native failure swallowed')
     try:audit.install(module)
     except AssertionError:pass
     else:raise AssertionError('Duplicate install permitted')
     cases.append({'rank':rank,'quant_mode':quant_mode,'native_argument_identity_preserved':True,'native_return_identity_preserved':True,'native_error_propagated':True,'first_use_rows':2})
   os.environ['GLM53_NODE']='34'
   try:audit.install(module)
   except AssertionError:pass
   else:raise AssertionError('P role accepted')
   os.environ['GLM53_NODE']='36';native.write_text(source+'\n')
   try:audit.install(module)
   except AssertionError:pass
   else:raise AssertionError('Wrong upstream source accepted')
 finally:os.environ.clear();os.environ.update(old_env)
 for node in ast.walk(ast.parse(path.read_text())):
  if isinstance(node,ast.Attribute):assert node.attr not in ['cpu','numpy','tolist','item','synchronize']
 proof={'project_id':'glm53','pure_stub_pass':True,'actual_image_CPU_test':False,'actual_NPU_qualification':False,'cases':cases,'wrong_source_and_P_role_rejected':True,'no_tensor_value_reads_or_synchronization_added':True,'model_requests':0,'SDK_init_calls':0,'NPU_calls':0,'source_evidence_sha256':{'scripts/glm53_C32_ops_window_audit_v2.py':AUDIT_SHA,'runs/20261017T2062-sglang-C32-ops-prerequisites-v1/node-36.json':hashlib.sha256((ROOT/'runs/20261017T2062-sglang-C32-ops-prerequisites-v1/node-36.json').read_bytes()).hexdigest()}}
 (ROOT/'results/C32-ops-window-audit-pure-stub-v2.json').write_text(json.dumps(proof,indent=2)+'\n');print('PASS64rank/quantstubcases; no framework/SDK/NPU calls')
if __name__=='__main__':main()
