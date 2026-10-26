"""Independent current-CID native counters baseline gate; local artifacts only."""
from pathlib import Path
import json,hashlib,collections,importlib.util
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 rid='20261018T-L09-sglang-L04-after-control-before-native-v1';run=ROOT/'runs'/rid;receipt=json.loads((run/'receipt.json').read_text());assert len(receipt['nodes'])==4 and all(x==0 for x in receipt['nodes'].values())
 settings=json.loads((ROOT/'config/C32-packed-normal-runtime-v1/formal-C32-L04-after-control-v1.json').read_text());startup=ROOT/'runs/20261018T-L09-packed-L04-rollback-startup-v1';audit=ROOT/'scripts/audit_sglang_C32_L09_L04_after_control_native_local_v1.py';spec=importlib.util.spec_from_file_location('native_auditor',audit);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 result={'project_id':'glm53','run_id':rid,'pass':True,'model_requests':0,'SDK_init_calls':0,'before_formal_requests':True,'nodes':{},'source_evidence_sha256':{str(audit.relative_to(ROOT)):sha(audit)}}
 for node in [34,35,36,41]:
  p=run/f'node-{node}.json';w=json.loads(p.read_text());assert w['rc']==0;o=json.loads(w['stdout']);assert o['container_id']==settings['runtime_containers'][str(node)]['container_id'] and o['/health']['status']==200 and o['/metrics']['status']==200 and o['provider_phase']=='READY' and o['supervisor']['phase']=='MODEL_RUNNING' and 'model_exit_code' not in o['supervisor']
  assert o['actual_model_identity']['namespace_pid']==o['supervisor']['model_pid'];env=o['actual_model_identity']['selected_environment'];assert not env['GLM53_PACKED_DISPATCH'] and env['DEEP_USE_MODE']=='alltoall'
  gauges=[float(l.rsplit(' ',1)[1]) for l in o['/metrics']['body'].splitlines() if l.startswith(('sglang:num_running_reqs{','sglang:num_queue_reqs{'))];assert gauges and not any(gauges)
  assert hashlib.sha256(o['/metrics']['body'].encode()).hexdigest()==o['/metrics']['sha256'];assert len(o['owned_npu_process_rows'])==18 and not o['profiler_artifacts']
  logs=o['logs'];assert all(x['full_archive_complete'] and x['error_line_count']==len(x['error_lines']) for x in logs.values())
  oldfile=startup/f'node-{node}.json';prior=json.loads(json.loads(oldfile.read_text())['stdout'])['model']['logs'];errors=[l for stream in ['stdout','stderr'] for l in logs[stream]['error_lines']];original=[l for stream in ['stdout','stderr'] for l in prior[stream+'_error_lines']];assert collections.Counter(errors)==collections.Counter(original),'No new errors since independent startup qualification'
  events=[]
  for item in o['persistent_audit'].values():
   raw=''.join(json.dumps(x,sort_keys=True)+'\n' for x in item['rows']).encode();assert len(raw)==item['bytes'] and hashlib.sha256(raw).hexdigest()==item['sha256'];events+=item['rows']
  initial={x['pid']:x for x in events if x['event']=='initialize_complete'};assert len(initial)==16 and all(x['rc']==0 and x['protocol']=='SDMA' and x['protocol_value']==1 for x in initial.values());assert len({x['pid'] for x in events if x['event']=='model_state_registered'})==16
  markers=[l for log in logs.values() for l in log['metadata_marker_lines']];first=[json.loads(l.split('GLM53_DEEPEP_METADATA_BATCH_FIRST_USE=',1)[1]) for l in markers if 'GLM53_DEEPEP_METADATA_BATCH_FIRST_USE=' in l];assert len(first)==(16 if node in [36,41] else 0)
  if first:assert {x['pid'] for x in first}==set(initial) and all(x['group_size']==32 and x['host_elements']==72 and x['metadata_host_downloads']==1 and x['repeat_output_size_from_same_snapshot'] for x in first)
  counts=m.sdk_counts(o,set(initial));assert set(counts)==set(initial)
  result['nodes'][str(node)]={'container_id':o['container_id'],'idle_gauges':gauges,'initialized_SDMA_workers':16,'registered_state_workers':16,'actual_normal_metadata_first_use_workers':len(first),'balanced_SDK_worker_files':len(counts),'initial_bind_error_lines_preserved':len(errors),'HCCL_BUFFSIZE_actual_environment':env['HCCL_BUFFSIZE'],'provider_namespace_pid':o['supervisor']['provider_pid'],'actual_model_identity':o['actual_model_identity']}
  result['source_evidence_sha256'].update({str(q.relative_to(ROOT)):sha(q) for q in [p,oldfile]})
 path=ROOT/'results/C32-L09-L04-after-control-native-baseline-v1.json';assert not path.exists();path.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'pass':True,'workers64':True,'metadata_first_use32':True,'sha256':sha(path)}))
if __name__=='__main__':main()
