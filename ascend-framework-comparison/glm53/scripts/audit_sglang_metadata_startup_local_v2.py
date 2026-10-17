"""Independently verify the fresh runtime and all 32 real D patched workers."""
from pathlib import Path
import hashlib,json,re
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 run=ROOT/'runs/20261017T1963-sglang-metadata-startup-stable-v1';receipt=json.loads((run/'receipt.json').read_text());assert len(receipt['nodes'])==4 and all(x==0 for x in receipt['nodes'].values())
 cap_path=ROOT/'results/sglang-metadata-pool-capacity-v1.json';cap=json.loads(cap_path.read_text());assert cap['pass']
 result={'project_id':'glm53','framework':'sglang','pass':True,'shared_run_id':'20261017T1954-sglang-metadata-v1','nodes':{},'runtime_CIDs':{n:x['container_id'] for n,x in cap['providers'].items()},'all32_D_actual_first_use_verified':True,'external_model_requests':0,'native_internal_startup_warmup_separate':True,'NPU_semantic_and_formal_correctness':'PENDING','source_evidence_sha256':{str(cap_path.relative_to(ROOT)):sha(cap_path)}}
 for n in [34,35,36,41]:
  p=run/f'node-{n}.json';wire=json.loads(p.read_text());assert wire['rc']==0;o=json.loads(wire['stdout']);model=o['model']
  assert model['id']==cap['providers'][str(n)]['container_id'] and model['state']['Running'] and model['image']=='sha256:c5cf2deb674cd1c8aba10a70d8dcb3e413d3739f6a54808c568e11d711d2e263'
  assert o['supervisor_status']['phase']=='MODEL_RUNNING' and 'model_exit_code' not in o['supervisor_status'] and o['provider_status']['phase']=='READY'
  metrics=o['/metrics'];assert metrics['status']==200 and sha_bytes(metrics['body'].encode())==metrics['sha256']
  gauges=[float(l.rsplit(' ',1)[1]) for l in metrics['body'].splitlines() if l.startswith(('sglang:num_running_reqs{','sglang:num_queue_reqs{'))];assert gauges and not any(gauges)
  events=[]
  for item in o['persistent_audit'].values():
   raw=''.join(json.dumps(x,sort_keys=True)+'\n' for x in item['rows']).encode();assert len(raw)==item['bytes'] and sha_bytes(raw)==item['sha256'];events+=item['rows']
  initial={x['pid']:x for x in events if x['event']=='initialize_complete'};assert len(initial)==16
  assert all(x['rc']==0 and x['protocol']=='SDMA' and x['protocol_value']==1 for x in initial.values())
  layouts=[x for x in events if x['event']=='model_state_registered'];assert len(layouts)==16
  for x in layouts:
   metadata=x['metadata'];assert metadata['hidden_kv_layers']==78 and metadata['draft_kv_layers']==1 and metadata['kv_buf_groups']==3 and len(metadata['kv_item_lens'])==237 and metadata['kv_cache_dtype_str']=='bfloat16'
  logs=model['logs'];markers=[l for stream in ['stdout','stderr'] for l in logs[stream+'_metadata_markers']]
  assert not any('GLM53_METADATA_HOOK_FATAL' in l for l in markers)
  first={};installed={}
  for line in markers:
   for token,target in [('GLM53_DEEPEP_METADATA_BATCH_FIRST_USE=',first),('GLM53_DEEPEP_METADATA_BATCH_INSTALLED=',installed)]:
    if token in line:
     x=json.loads(line.split(token,1)[1]);target.setdefault(x['pid'],[]).append(x)
  if n in [36,41]:
   assert set(initial)<=set(installed) and set(first)==set(initial),('All actual D initialized worker PIDs need native first-use markers',n,set(initial),set(first))
   for pid,rows in first.items():
    assert len(rows)==1,('Unexpected multiple strategy instances; review before accepting',pid)
    x=rows[0];assert x['device_index']==initial[pid]['device_id'] and x['group_size']==32 and x['router_rows']>0 and x['host_elements']==72 and x['host_vectors']==3 and x['metadata_host_downloads']==1 and x['repeat_output_size_from_same_snapshot'] is True
    assert x['upstream_normal_strategy_sha256']=='48fbc94dd60c7c37e198935717357d2479150099487ee0d9a749ed41083a7f84'
  else:assert not installed and not first
  if n in [34,36]:
   expected=f'Disaggregation warmup requests completed for all {4 if n==34 else 16} DP ranks'
   assert any(expected in l for stream in ['stdout','stderr'] for l in logs[stream+'_capacity_lines'])
   selected={x['path']:x['value'] for x in o['/get_server_info']['selected']}
   for key,value in [('tp_size',32),('ep_size',32),('dp_size',4 if n==34 else 16),('speculative_num_steps',5),('speculative_num_draft_tokens',6),('kv_cache_dtype','bfloat16'),('deepep_mode','normal'),('max_total_num_tokens',254848 if n==34 else 181632)]:assert selected[key]==value,(n,key,selected.get(key))
  errors=[l for stream in ['stdout','stderr'] for l in logs[stream].splitlines() if any(k in l for k in ['Traceback','GLM53_METADATA_HOOK_FATAL'])];assert not errors,errors
  startup_errors=[l for stream in ['stdout','stderr'] for l in logs[stream+'_error_lines']]
  for line in startup_errors:
   assert 'AccLink acc_tcp_port_util.cpp:' in line and ('BindTcpPortV4' in line or 'BindTcpPortV6' in line) and 'errno=98' in line and 'Address already in use' in line,('Unreviewed startup error',line)
  result['nodes'][str(n)]={'preserved_startup_control_port_bind_errors':startup_errors,'diagnostic_scope':'Complete initialization and native routing observed after these startup diagnostics; poststartup functional/SDMA qualification still required','container_id':model['id'],'native_initialized_workers':16,'model_state_registered_workers':16,'actual_metadata_batch_workers':len(first),'idle_gauges':gauges,'first_use':first,'source_sha256':sha(p)}
  result['source_evidence_sha256'][str(p.relative_to(ROOT))]=sha(p)
 p=ROOT/'results/sglang-metadata-startup-v1.json';assert not p.exists();p.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'pass':True,'actual_D_first_use_workers':32,'sha256':sha(p)}))
def sha_bytes(raw):return hashlib.sha256(raw).hexdigest()
if __name__=='__main__':main()
