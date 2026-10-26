"""Independent local audit of bounded new NPU tensor qualification."""
from pathlib import Path
import json,hashlib
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 run=ROOT/'runs/20261018T-L09-packed-dispatch-NPU-tensor-v1';receipt=json.loads((run/'receipt.json').read_text());assert receipt['rc']==0 and receipt['launcher_sha256']=='3dca226a33cc7326c48eb7e01a1ffaeaf1952ceb92aece94b42d004842e7bf07'
 p=run/'node-36.json';w=json.loads(p.read_text());assert w['rc']==0;o=json.loads(w['stdout']);assert o['probe_rc']==0
 rt=json.loads((ROOT/'config/C32-window-L04-rollback-runtime-v2/runtime.json').read_text());assert o['container_id']==rt['runtime_containers']['36']['container_id'] and o['image']==rt['model_image']
 assert o['external_model_HTTP_requests']==o['SDK_init_calls']==o['model_service_control_calls']==0
 assert o['before']['owned_model_context_pids']==o['after']['owned_model_context_pids'] and len(o['after']['owned_model_context_pids'])==18 and o['physical_free_MiB_before']>=2048
 for phase in ['before','after']:
  x=o[phase];assert x['supervisor']['phase']=='MODEL_RUNNING' and 'model_exit_code' not in x['supervisor'] and x['idle_gauges'] and all(float(l.rsplit(' ',1)[1])==0 for l in x['idle_gauges'])
 result=o['result'];assert result['pass'] and result['cases']==len(result['checks'])==8 and all(x['pass'] and x['inputs_not_mutated'] and x['hidden']==6144 and x['quantized_token_dtype']=='torch.int8' and x['dynamic_scale_dtype']=='torch.float32' and x['wire_device']==x['restored_token_device']==x['restored_scale_device']=='npu:0' and x['wire_shape']==[x['rows'],6148] for x in result['checks'])
 assert {x['case'] for x in result['checks']}=={'direct_special_bits_n'+str(n) for n in [0,1,2,6,17]}|{'actual_native_MoeInitRoutingV2_int8_n'+str(n) for n in [1,2,6]}
 assert not result['native_HCCL_collectives_executed'] and not result['model_tensor_values_observed'] and not result['system_settings_changed'] and result['probe_process_Torch_allocator_fraction']==0.01
 helper='336a64b8f6657541f24daa49588891072d350b4a0d9d94b383c6f3fe08cea5cb';assert result['helper_SHA256']==sha(ROOT/'scripts/glm53_deepep_packed_dispatch_v1.py')==helper
 out={'project_id':'glm53','pass':True,'synthetic_NPU_cases':8,'helper_SHA256':helper,'live_L04_preserved':True,'qualification':result['qualification'],'image':o['image'],'qualified_on_current_node36_container':o['container_id'],'external_model_HTTP_requests':0,'SDK_init_calls':0,'model_service_control_calls':0,'native_HCCL_collective_and_whole_model_correctness':'PENDING','original_inputs_and_restored_token_scale_bytes_exact':True,'probe_Torch_peak_allocated_bytes':result['probe_process_peak_Torch_allocated_bytes'],'probe_Torch_peak_reserved_bytes':result['probe_process_peak_Torch_reserved_bytes'],'memory_limit_scope':result['memory_limit_scope'],'source_evidence_sha256':{str(x.relative_to(ROOT)):sha(x) for x in [run/'receipt.json',p,ROOT/'scripts/glm53_packed_dispatch_NPU_tensor_fixture_v1.py',ROOT/'scripts/glm53_deepep_packed_dispatch_v1.py']}}
 p=ROOT/'results/C32-L09-packed-dispatch-NPU-tensor-qualification-v1.json';assert not p.exists();p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'pass':True,'NPU_tensor_cases':8,'whole_model_correctness':'PENDING','sha256':sha(p)}))
if __name__=='__main__':main()
