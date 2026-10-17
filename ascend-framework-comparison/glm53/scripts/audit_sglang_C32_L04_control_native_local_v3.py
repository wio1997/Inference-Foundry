"""Independent SG measurement audit; no model, SDK, SSH or resource calls."""
import argparse,collections,hashlib,json,re
from decimal import Decimal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def read(relative):return json.loads((ROOT/relative).read_text())
def sha(relative):return hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()
def unpack(capture,node):
 path=f'runs/{capture}/node-{node}.json';wire=read(path);assert wire['rc']==0;o=json.loads(wire['stdout']);events=[]
 for item in o['persistent_audit'].values():
  raw=''.join(json.dumps(x,sort_keys=True)+'\n' for x in item['rows']).encode();assert len(raw)==item['bytes'] and hashlib.sha256(raw).hexdigest()==item['sha256'];events.extend(item['rows'])
 # Historical1747 snapshot uses a flat log schema; preserve its original bytes.
 if 'logs' not in o:
  flat=o['model']['logs'];o['logs']={s:{k:flat[s+'_'+k] for k in ['error_lines','error_line_count','full_archive_complete','capacity_lines','native_sdma']} for s in ['stdout','stderr']};o['container_id']=o['model']['id']
 return path,o,events
def sdk_counts(obj,pids):
 result={}
 for snapshot in obj['native_SDK_small_file_snapshots']:
  assert snapshot['rc']==0
  if snapshot['container_id']!=obj['container_id']:continue
  for item in snapshot['files']:
   assert item['complete_line_ending'] and hashlib.sha256(item['text'].encode()).hexdigest()==item['sha256']
   match=re.search(r'ptracer_(\d+)\.dat$',item['path'])
   if not match or int(match.group(1)) not in pids:continue
   pid=int(match.group(1));tags={}
   for line in item['text'].splitlines():
    parts=line.split()
    if len(parts)>8 and parts[2] in ['TP_MMC_PY_BATCH_GET','TP_MMC_LOCAL_BATCH_GET','TP_MMC_LOCAL_BATCH_GET_SIZE','TP_HYBM_SDMA_BATCH_LD_TO_GD']:
     entry=tags.setdefault(parts[2],{'begin':0,'good':0,'bad':0,'on_fly_last':None,'total_value_sum':Decimal(0)});entry['begin']+=int(parts[3]);entry['good']+=int(parts[4]);entry['bad']+=int(parts[5]);entry['on_fly_last']=int(parts[6]);entry['total_value_sum']+=Decimal(parts[7])
   for tag,entry in tags.items():assert entry['begin']==entry['good'] and entry['bad']==0 and entry['on_fly_last']==0
   result[pid]=tags
 return result
def audit(run_id,capture_id,before_id):
 report=read(f'runs/{run_id}/formal-result.remote.json');C=report['concurrency'];assert report['framework']=='sglang' and C==32 and run_id=='20261017T2052-sglang-formal-C32-L04-control-v3' and capture_id=='20261017T2054-sglang-C32-L04-control-native-v3' and before_id=='20261017T2050-sglang-C32-L04-before-control-native-v1'
 import glm53_sglang_cache_annotation_review_local_v1 as cache_review
 reviewed=cache_review.review(report);assert reviewed['pass']
 null_counts={'P':0,'D':0}
 settings=read('config/C32-only-L04-restore/formal-C32-control-v3.json');assert settings['run_id']==run_id
 client_path=f'results/sglang-C32-L04-control-v3-client-audit-v1.json';client=read(client_path);assert client['client_audit_pass'] and client['raw_formal_report_sha256']==sha(f'runs/{run_id}/formal-result.remote.json')
 sources={client_path:sha(client_path)};objects={};runtime={};PD=[];native_bytes=0;read_workers=[];errors={};sessions={};mtp=[];tiers={'P':collections.Counter(),'D':collections.Counter()};storage_backends={'P':collections.Counter(),'D':collections.Counter()}
 for row in report['formal']['rows']:
  result=row['result'];gw=result['terminal']['gateway'];ext=result['native']['extra_details_data']['glm53_native_sglext'];spec=ext['spec_tokens_details'];verify=spec['spec_verify_ct'];proposed=spec['spec_num_proposed_drafts'];correct=spec['spec_num_correct_drafts'];hist=spec['spec_correct_drafts_histogram']
  assert verify>0 and proposed==5*verify and 0<=correct<=proposed and len(hist)==6 and all(type(x) is int and x>=0 for x in hist) and sum(hist)==verify and sum(i*x for i,x in enumerate(hist))==correct
  mtp.append({'verify':verify,'proposed':proposed,'correct':correct})
  for role,extension in [('P',gw['prefill_sglext']),('D',ext)]:
   details=extension['cached_tokens_details'];numeric={k:details[k] for k in ['device','host','storage'] if k in details};annotations={k:v for k,v in details.items() if k not in numeric};tiers[role].update(numeric);null_counts[role]+=int('storage' not in details)
   if 'storage_backend' in annotations:storage_backends[role][str(annotations['storage_backend'])]+=1
 for node in [34,35,36,41]:
  path,o,events=unpack(capture_id,node);before_path,previous,old=unpack(before_id,node);sources[path]=sha(path);sources[before_path]=sha(before_path);objects[node]=o;runtime[str(node)]=o['container_id']
  assert o['container_id']==settings['runtime_containers'][str(node)]['container_id'] and o['provider_phase']=='READY' and o['/health']['status']==200 and {x['namespace_pid'] for x in o['owned_npu_process_rows']}==({7,o['supervisor']['model_pid']}|{x['pid'] for x in events if x['event']=='initialize_complete'}) and not o['profiler_artifacts']
  assert o['actual_model_identity']['namespace_pid']==o['supervisor']['model_pid'] and o['supervisor']['phase']=='MODEL_RUNNING' and 'model_exit_code' not in o['supervisor']
  for body in [o['/metrics'],previous['/metrics']]:assert hashlib.sha256(body['body'].encode()).hexdigest()==body['sha256']
  gauges=[float(x.rsplit(' ',1)[1]) for x in o['/metrics']['body'].splitlines() if x.startswith(('sglang:num_running_reqs{','sglang:num_queue_reqs{'))];assert gauges and all(v==0 for v in gauges)
  assert all(log['full_archive_complete'] for log in o['logs'].values())
  assert all(log['native_sdma']['copy_calls']==0 or log['native_sdma']['copy_status_values']==[0] for log in o['logs'].values())
  initial={x['pid']:x for x in events if x['event']=='initialize_complete'};assert len(initial)==16 and all(x['rc']==0 and x['protocol']=='SDMA' and x['protocol_value']==1 for x in initial.values());sessions.update({x['session']:(node,x['device_id']) for x in initial.values()})
  marker_lines=[line for log in o['logs'].values() for line in log['metadata_marker_lines']]
  first=[json.loads(line.split('GLM53_DEEPEP_METADATA_BATCH_FIRST_USE=',1)[1]) for line in marker_lines if 'GLM53_DEEPEP_METADATA_BATCH_FIRST_USE=' in line]
  if node in [36,41]:
   assert {x['pid'] for x in first}==set(initial) and len(first)==16
   assert all(x['metadata_host_downloads']==1 and x['host_elements']==72 and x['group_size']==32 and x['upstream_normal_strategy_sha256']=='48fbc94dd60c7c37e198935717357d2479150099487ee0d9a749ed41083a7f84' for x in first)
  else:assert not first
  layouts=[x for x in events if x['event']=='model_state_registered'];assert len(layouts)==16
  for item in layouts:
   meta=item['metadata'];assert meta['hidden_kv_layers']==78 and meta['draft_kv_layers']==1 and meta['kv_buf_groups']==3 and len(meta['kv_item_lens'])==237 and len(meta['aux_item_lens'])==11 and meta['page_size']==128 and meta['kv_cache_dtype_str']=='bfloat16'
  existing={json.dumps(x,sort_keys=True) for x in old};new=[x for x in events if json.dumps(x,sort_keys=True) not in existing]
  assert not any(x['event'] in ['write_error','model_state_send_error','forbidden_tcp_model_payload'] for x in new)
  writes=[x for x in new if x['event']=='write_complete']
  for x in writes:assert x['rc']==0 and x['payload_bytes_completed']==x['payload_bytes_requested']>0 and x['pid'] in initial and node in [34,35] and x['context']['kind'] in ['send_kvcache','send_aux'];PD.append(x|{'node':node})
  assert all(x['rc']==0 for x in new if x['event']=='model_state_send_complete')
  current_errors=[x for log in o['logs'].values() for x in log['error_lines']];old_errors=[x for log in previous['logs'].values() for x in log['error_lines']];assert current_errors==old_errors,'New native errors require review'
  assert sum(log['error_line_count'] for log in o['logs'].values())==sum(log['error_line_count'] for log in previous['logs'].values());errors[str(node)]={'preserved_existing_error_lines':len(current_errors),'new_error_lines':0}
  if node in [34,35]:
   delta=sum(log['native_sdma']['copy_descriptor_bytes'] for log in o['logs'].values())-sum(log['native_sdma']['copy_descriptor_bytes'] for log in previous['logs'].values());assert delta==sum(x['payload_bytes_completed'] for x in writes);native_bytes+=delta
   before=sdk_counts(previous,initial);after=sdk_counts(o,initial)
   for pid,tags in after.items():
    if 'TP_MMC_PY_BATCH_GET' not in tags:continue
    deltas={name:{k:entry[k]-before.get(pid,{}).get(name,{}).get(k,0) for k in ['begin','good','bad','total_value_sum']} for name,entry in tags.items()};assert all(v['bad']==0 and v['begin']==v['good'] for v in deltas.values())
    if deltas['TP_MMC_PY_BATCH_GET']['good']>0:read_workers.append({'node':node,'pid':pid,'device_id':initial[pid]['device_id'],'counter_deltas':{name:{k:str(v) if isinstance(v,Decimal) else v for k,v in entry.items()} for name,entry in deltas.items()}})
 assert {(x['node'],x['device_id']) for x in read_workers}=={(node,d) for node in [34,35] for d in range(16)},'All32 P workers must read actual MemCache payload'
 assert sum(x['correct'] for x in mtp)>0
 assert PD and read_workers and native_bytes==sum(x['payload_bytes_completed'] for x in PD)
 actual_D_devices=set()
 for x in PD:
  assert x['remote_session'] in sessions;node,device=sessions[x['remote_session']];assert node in [36,41];actual_D_devices.add((node,device))
 assert actual_D_devices=={(node,d) for node in [36,41] for d in range(16)},'All D SDMA devices must receive real model state'
 assert {x['context']['kind'] for x in PD}=={'send_kvcache','send_aux'}
 gate=objects[34]['actual_formal_cache_gate'];wire=read(f'runs/{run_id}/cache-gate-native.json');payload=json.loads(wire['stdout']);assert wire['rc']==payload['query_rc']==0 and gate['report']==report['cache_gate'] and gate['sha256']==payload['report_sha256']
 cpupath=f'runs/{capture_id}/node-43.json';wire=read(cpupath);assert wire['rc']==0;cpu=json.loads(wire['stdout']);sources[cpupath]=sha(cpupath);metrics=cpu['independent_formal_metric_archive_audit'];assert metrics['all_native_bodies_SHA_and_DP_gauges_recomputed'] and not metrics['errors'] and metrics['sample_pairs']==metrics['complete_pairs'] and metrics['archive_sha256']==client['D_metric_archive_sha256'] and cpu['formal_report_identity']['sha256']==client['raw_formal_report_sha256']
 native={'project_id':'glm53','framework':'sglang','concurrency':C,'run_id':run_id,'pass':True,'runtime_CIDs':runtime,'native_PD_write_batches':len(PD),'native_PD_descriptor_bytes':native_bytes,'PD_descriptor_bytes_by_state':dict((kind,sum(x['payload_bytes_completed'] for x in PD if x['context']['kind']==kind)) for kind in ['send_kvcache','send_aux']),'native_D_devices_receiving_model_state':len(actual_D_devices),'physical_wire_bytes':None,'native_PD_request_ID_join':None,'native_PD_request_ID_join_limitation':'Existing SDK send audit exposes peer sessions and typed state, with room=null; period transfers/all32Ddevices are verified. Logical request success/count/cache/terminal joins are independently verified in AIS/client records. No invented per-request native transfer mapping.','native_MemCache_read_workers':read_workers,'formal_native_cached_token_tiers':{k:dict(v) for k,v in tiers.items()},'storage_field_null_requests':null_counts,'storage_sum_scope':'Only observed numeric storage values summed; actual omitted fields are counted in storage_field_null_requests and remain null. Prefix verification from supplemental native formatter/P-D metadata evidence.','storage_backend_annotation_counts':{k:dict(v) for k,v in storage_backends.items()},'native_cached_token_tiers_scope':'Native API accounting; D HiCache disabled. D storage label does not establish a direct D MemCache read; native SDK/PD evidence reported separately.','formal_MTP':{'requests':len(mtp),'verify_steps':sum(x['verify'] for x in mtp),'proposed_drafts':sum(x['proposed'] for x in mtp),'correct_drafts':sum(x['correct'] for x in mtp),'num_drafts_per_verify':5},'execution_mode':'Native eager; graph disabled in frozen SG startup, no FULL graph claim','independent_complete_D_metrics_archive':metrics,'preserved_connection_diagnostics':errors,'source_evidence_sha256':sources,'cache_annotation_review_sha256':sha(f'results/sglang-C32-L04-control-v3-cache-annotation-review-v1.json'), 'original_client_baseline_accepted':report['baseline_accepted']}
 nativepath=f'results/sglang-C32-L04-control-v3-native-v1.json';(ROOT/nativepath).write_text(json.dumps(native,indent=2)+'\n')
 certificate=client|{'measurement_valid':True,'native_postrun_audit_required':False,'native_postrun_evidence':nativepath,'native_postrun_evidence_sha256':sha(nativepath),'shared_run_id':settings['shared_run_id'],'runtime_CIDs':runtime,'accepted_baseline':False,'accepted_optimization_trial':False,'accepted_C32_restored_L04_control':True,'formal_measurement_accepted_SLA_separate':True,'acceptance_basis':'Independent supplemental native cache optional-field review, complete original client timing/count/input/gate checks and postrun native verification. Original client rejection retained unchanged.','sampled_peak_D_running':metrics['sampled_peak_total_D_running'],'native_proofs':{'actual_allprefix_payload_samples':True,'actual_PD_SDMA_period_state_and_all_D_devices':True,'actual_native_MemCache_reads':True,'actual_MTP5':True,'complete_D16_native_metrics':True,'native_execution_mode':'eager'}}
 path=f'results/sglang-C32-L04-control-v3-measurement-v1.json';(ROOT/path).write_text(json.dumps(certificate,indent=2)+'\n');print(json.dumps({'certificate':path,'sha256':sha(path),'measurement_valid':True,'aggregate_SLA_pass':certificate['aggregate_SLA_pass'],'output_tokens_per_second':certificate['output_tokens_per_second']}));return certificate
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('run_id');parser.add_argument('capture_id');parser.add_argument('before_id');a=parser.parse_args();audit(a.run_id,a.capture_id,a.before_id)
