"""Independently verify the fresh runtime and all 32 real D patched workers."""
from pathlib import Path
import hashlib,json,re
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 run=ROOT/'runs/20261018T-L08-sglang-window-startup-v2';receipt=json.loads((run/'receipt.json').read_text());assert len(receipt['nodes'])==4 and all(x==0 for x in receipt['nodes'].values())
 cap_path=ROOT/'results/sglang-C32-ops-window-pool-capacity-v2.json';cap=json.loads(cap_path.read_text());assert cap['pass']
 result={'project_id':'glm53','framework':'sglang','pass':True,'shared_run_id':'20261018T-L08-sglang-C32-ops-window-v2','nodes':{},'runtime_CIDs':{n:x['container_id'] for n,x in cap['providers'].items()},'all32_D_actual_native_ops_dispatch_combine_verified':True,'P_archived_L05_candidate_disabled':True,'C32_only':True,'external_model_requests':0,'native_internal_startup_warmup_separate':True,'NPU_semantic_and_formal_correctness':'PENDING','source_evidence_sha256':{str(cap_path.relative_to(ROOT)):sha(cap_path)}}
 for n in [34,35,36,41]:
  p=run/f'node-{n}.json';wire=json.loads(p.read_text());assert wire['rc']==0;o=json.loads(wire['stdout']);model=o['model'];assert o['startup_wait']['completed'] and not o['startup_wait'].get('timeout')
  assert model['id']==cap['providers'][str(n)]['container_id'] and model['state']['Running'] and model['image']=='sha256:c5cf2deb674cd1c8aba10a70d8dcb3e413d3739f6a54808c568e11d711d2e263'
  assert o['supervisor_status']['phase']=='MODEL_RUNNING' and 'model_exit_code' not in o['supervisor_status'] and o['provider_status']['phase']=='READY'
  assert o['npu']['rc']==model['top']['rc']==0
  own_host_pids={int(line.split()[0]) for line in model['top']['stdout'].splitlines()[1:] if line.split() and line.split()[0].isdigit()}
  npu_pids=[]
  for line in o['npu']['stdout'].split('Process id',1)[-1].splitlines():
   fields=[x.strip() for x in line.split('|')]
   if len(fields)>=5 and len(fields[1].split())==2 and all(x.isdigit() for x in fields[1].split()) and fields[2].isdigit():npu_pids.append(int(fields[2]))
  assert len(npu_pids)==18 and set(npu_pids)<=own_host_pids,'Unexpected or foreign NPU context'

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
  logs=model['logs'];assert not any(logs[stream+'_archived_candidate_markers'] for stream in ['stdout','stderr']), 'RestoredC32L04mustnotenablearchivedPcandidate'
  markers=[l for stream in ['stdout','stderr'] for l in logs[stream+'_metadata_markers']]
  assert not any('GLM53_METADATA_HOOK_FATAL' in l for l in markers)
  first={};installed={};normal_first=[]
  for line in markers:
   if 'GLM53_C32_OPS_HOOK_FATAL' in line:raise AssertionError(line)
   if 'GLM53_DEEPEP_METADATA_BATCH_FIRST_USE=' in line:normal_first.append(line)
   for token,target in [('GLM53_C32_OPS_FIRST_USE=',first),('GLM53_C32_OPS_AUDIT_INSTALLED=',installed)]:
    if token in line:
     x=json.loads(line.split(token,1)[1]);target.setdefault(x['pid'],[]).append(x)
  assert not normal_first,'Forced low_latency must not execute AlltoallNormal host metadata path'
  if n in [36,41]:
   assert set(initial)<=set(installed) and set(first)==set(initial)
   ranks=set()
   for pid,rows in first.items():
    assert len(rows)==2 and {x['method'] for x in rows}=={'_npu_low_latency_dispatch','_npu_low_latency_combine'}
    for x in rows:
     assert x['HCCL_BUFFSIZE_environment']==x['DEEPEP_HCCL_BUFFSIZE_environment']=='1024'
     assert x['group_size']==32 and x['node']==str(n) and x['original_method_returned'] and x['comm_alg']=='hierarchy' and x['num_experts']==256
     assert x['upstream_sha256']=='1b9fe42cc46425c8d7e97d4f4e8a0501a6465e9b49033522d89b31362d363c46'
     assert x['num_max_dispatch_tokens_per_rank']==128
     assert x['inputs']['topk_idx']['shape']==[128,8] and x['inputs']['topk_idx']['dtype']=='torch.int32'
     assert x['inputs']['x_active_mask']['shape']==[128] and x['inputs']['x_active_mask']['dtype']=='torch.bool'
     assert x['inputs']['topk_weights']['shape']==[128,8] and x['inputs']['topk_weights']['dtype']=='torch.float32'
     if x['method'].endswith('dispatch'):
      assert x['quant_mode'] in [0,2] and x['inputs']['x']['shape']==[128,6144]
      assert x['outputs'][2]['shape']==[8] and x['outputs'][2]['dtype']=='torch.int64'
     else:assert x['comm_quant_mode']==0 and x['outputs']['shape']==[128,6144]
     ranks.add(x['rank'])
   assert ranks==set(range(0,16) if n==36 else range(16,32))
  else:assert not installed and not first
  if n in [34,36]:
   expected=f'Disaggregation warmup requests completed for all {4 if n==34 else 16} DP ranks'
   assert any(expected in l for stream in ['stdout','stderr'] for l in logs[stream+'_capacity_lines'])
   selected={x['path']:x['value'] for x in o['/get_server_info']['selected']}
   for key,value in [('tp_size',32),('ep_size',32),('dp_size',4 if n==34 else 16),('speculative_num_steps',5),('speculative_num_draft_tokens',6),('kv_cache_dtype','bfloat16'),('deepep_mode','normal' if n==34 else 'low_latency')]:assert selected[key]==value,(n,key,selected.get(key))
   assert selected['max_total_num_tokens']==254848 if n==34 else selected['max_total_num_tokens']>=161280,('Insufficient2x80Kcapacity',n,selected['max_total_num_tokens'])
   result.setdefault('KV_tokens_per_DP',{})[str(n)]=selected['max_total_num_tokens']
  errors=[l for stream in ['stdout','stderr'] for l in logs[stream+'_error_lines']]
  bind_pattern=r'^\d{4}-\d{2}-\d{2} [0-9:.]+ ERROR \[\d+-\d+\]\[AccLink acc_tcp_port_util\.cpp:(?:43|62) BindTcpPortV[46]\] bind ipv[46] tcp port failed, port=\d+, errno=98, errstr=Address already in use$'
  preserved_bind=[line for line in errors if re.fullmatch(bind_pattern,line)]
  assert len(preserved_bind)==len(errors),errors
  result.setdefault('startup_control_bind_diagnostics_preserved',{})[str(n)]=len(preserved_bind)
  # Scope: only exact initial bind-attempt diagnostics, after all current
  # initialization/registration/warmup/health/nativeOps proofs above passed.

  result['nodes'][str(n)]={'container_id':model['id'],'native_initialized_workers':16,'model_state_registered_workers':16,'actual_native_ops_workers':len(first),'idle_gauges':gauges,'first_use':first,'source_sha256':sha(p)}
  result['source_evidence_sha256'][str(p.relative_to(ROOT))]=sha(p)
 p=ROOT/'results/sglang-C32-ops-window-startup-v2.json';assert not p.exists();p.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'pass':True,'actual_D_first_use_workers':32,'sha256':sha(p)}))
def sha_bytes(raw):return hashlib.sha256(raw).hexdigest()
if __name__=='__main__':main()
