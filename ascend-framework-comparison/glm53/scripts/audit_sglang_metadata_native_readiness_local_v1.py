"""New-runtime semantic/native cache/SDMA hard gates; local evidence only."""
import hashlib,importlib.util,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads((ROOT/p).read_text())
def outer(p):
 wire=read(p);assert wire['rc']==0;raw=wire['stdout'].encode()
 if 'native_output_transport' in wire:
  transport=wire['native_output_transport'];assert len(raw)==transport['raw_bytes'] and hashlib.sha256(raw).hexdigest()==transport['raw_sha256']
 return json.loads(raw)
def events(o):
 rows=[]
 for item in o['persistent_audit'].values():
  raw=''.join(json.dumps(x,sort_keys=True)+'\n' for x in item['rows']).encode();assert len(raw)==item['bytes'] and hashlib.sha256(raw).hexdigest()==item['sha256'];rows+=item['rows']
 return rows
def main():
 source=ROOT/'scripts/audit_glm53_sglang_formal_native_local_v5.py';assert sha(source)=='0e4eabeeae3c49bf9ea458a90d385bc45a7bdd9b5cf86e30183dbdd3cd0b840e'
 spec=importlib.util.spec_from_file_location('native_count_audit',source);native=importlib.util.module_from_spec(spec);spec.loader.exec_module(native)
 startup=read('results/sglang-metadata-startup-v1.json');short=read('results/sglang-metadata-short-v1.json');capacity=read('results/sglang-metadata-pool-capacity-v1.json')
 assert startup['pass'] and startup['all32_D_actual_first_use_verified'] and short['pass'] and short['successful_backend_inferences']==32 and capacity['pass']
 long_id='20261017T1964-sglang-metadata-long-v1';receipt=read(f'runs/{long_id}/receipt.json');assert receipt['rc']==0 and len(receipt['model_precheck'])==4 and all(x==0 for x in receipt['model_precheck'].values())
 long_path=f'runs/{long_id}/node-43.json';wire=read(long_path);assert wire['rc']==0
 artifacts=json.loads(next(l.split('=',1)[1] for l in wire['stdout'].splitlines() if l.startswith('GLM53_LONG_ARTIFACTS=')))
 report=json.loads(artifacts['semantic-result.json']['text']);phase=json.loads(artifacts['semantic-phase-status.json']['text'])
 assert phase['client_rc']==0 and len(phase['gates'])==2 and all(x['pass'] and x['rc']==0 for x in phase['gates'].values())
 assert report['semantic_pass'] and len(report['cases'])==len(report['gateway'])==6 and report['external_backend_HTTP_attempts']==report['successful_backend_inferences']==12 and report['actual_successful_output_tokens']==2408
 MTP=[]
 for row in report['cases']:
  assert row['count_pass'] and row['pd_lifecycle_pass'] and row['required_input'] in [74400,80000] and row['required_output'] in [1,600]
  if row['required_output']==600:
   assert row['semantic_pass'] and row['observed_answer_fields'][0]==row['expected_answer']
   detail=row['native']['extra_details_data']['glm53_native_sglext']['spec_tokens_details'];verify=detail['spec_verify_ct'];assert verify>0 and detail['spec_num_proposed_drafts']==5*verify and detail['spec_num_correct_drafts']>0;MTP.append(detail)
 assert len(MTP)==4
 for gate in phase['gates'].values():
  meta=gate['metadata'];assert meta['pass'] and meta['close_rc']==0 and meta['known_component_keys']==1743 and meta['last_existing_component_keys']==1743 and meta['object_bytes']==8253063168 and meta['init_bm'] is False
 assert len(report['cache_readback_guards'])==2 and all(x['pass'] for x in report['cache_readback_guards'].values())
 capture='20261017T1966-sglang-metadata-native-qualification-v1';capture_receipt=read(f'runs/{capture}/receipt.json');assert len(capture_receipt['nodes'])==4 and all(x==0 for x in capture_receipt['nodes'].values())
 sessions={};writes=[];reads={};objects={};sources={long_path:sha(ROOT/long_path)}
 for n in [34,35,36,41]:
  path=f'runs/{capture}/node-{n}.json';o=outer(path);objects[n]=o;sources[path]=sha(ROOT/path);current=events(o)
  assert o['container_id']==startup['runtime_CIDs'][str(n)] and o['provider_phase']=='READY' and o['/health']['status']==200 and not o['profiler_artifacts']
  initial={x['pid']:x for x in current if x['event']=='initialize_complete'};assert len(initial)==16 and all(x['protocol']=='SDMA' and x['protocol_value']==1 and x['rc']==0 for x in initial.values())
  assert {x['namespace_pid'] for x in o['owned_npu_process_rows']}==({7,o['supervisor']['model_pid']}|set(initial))
  sessions.update({x['session']:(n,x['device_id']) for x in initial.values()})
  before=outer(f'runs/20261017T1963-sglang-metadata-startup-stable-v1/node-{n}.json');old={json.dumps(x,sort_keys=True) for x in events(before)}
  new=[x for x in current if json.dumps(x,sort_keys=True) not in old]
  assert not any(x['event'] in ['write_error','model_state_send_error','forbidden_tcp_model_payload'] for x in new)
  for x in new:
   if x['event']=='write_complete':assert n in [34,35] and x['rc']==0 and x['payload_bytes_completed']==x['payload_bytes_requested']>0;writes.append(x|{'node':n})
  assert all(x['rc']==0 for x in new if x['event']=='model_state_send_complete')
  current_errors=[l for log in o['logs'].values() for l in log['error_lines']];old_errors=[l for stream in ['stdout','stderr'] for l in before['model']['logs'][stream+'_error_lines']];assert current_errors==old_errors,'New native error requires review'
  for log in o['logs'].values():assert log['full_archive_complete'] and (log['native_sdma']['copy_calls']==0 or log['native_sdma']['copy_status_values']==[0])
  gauges=[float(l.rsplit(' ',1)[1]) for l in o['/metrics']['body'].splitlines() if l.startswith(('sglang:num_running_reqs{','sglang:num_queue_reqs{'))];assert gauges and not any(gauges)
  if n in [34,35]:
   counters=native.sdk_counts(o,initial);positive={pid:tags for pid,tags in counters.items() if tags.get('TP_MMC_PY_BATCH_GET',{}).get('good',0)>0};assert len(positive)>=8,'Actual model-side native cache payload reads required on each P node';reads[str(n)]=positive
 assert writes and {x['context']['kind'] for x in writes}=={'send_kvcache','send_aux'}
 receiving={sessions[x['remote_session']] for x in writes};assert receiving=={(n,d) for n in [36,41] for d in range(16)}
 for name in ['results/sglang-metadata-startup-v1.json','results/sglang-metadata-short-v1.json','results/sglang-metadata-pool-capacity-v1.json','results/sglang-metadata-SDK-readback-v1.json','scripts/glm53_metadata_model_identity_v2.py','scripts/glm53_deepep_metadata_batch_v1.py','scripts/glm53_deepep_metadata_import_hook_v1.py']:sources[name]=sha(ROOT/name)
 result={'project_id':'glm53','framework':'sglang','pass':True,'shared_run_id':'20261017T1954-sglang-metadata-v1','runtime_CIDs':startup['runtime_CIDs'],'all32_D_actual_first_use_verified':True,'all_DP_short_and_80K_semantic_verified':True,'native_cache_and_SDMA_verified':True,'qualification':'Freshruntime all32nativeSDMA/all16D-DPshort/two80Kfacts/cross-PDPmodelSDKreads; formalper-C allprefixpayload gate stillrequired','external_functional_backend_HTTP_attempts':44,'successful_functional_backend_inferences':44,'successful_functional_output_tokens':2680,'native_model_cache_read_counts':reads,'native_D_devices_receiving_poststartup_state':len(receiving),'poststartup_PD_write_batches':len(writes),'poststartup_PD_descriptor_bytes':sum(x['payload_bytes_completed'] for x in writes),'physical_wire_bytes':None,'MTP_requests_checked':len(MTP),'formal_requests':0,'accepted_performance_baseline':False,'startup_bind_diagnostics_preserved':{n:x['preserved_startup_control_port_bind_errors'] for n,x in startup['nodes'].items()},'source_evidence_sha256':sources}
 p=ROOT/'results/sglang-metadata-native-readiness-v1.json';assert not p.exists();p.write_text(json.dumps(result,indent=2,default=str)+'\n');print(json.dumps({'pass':True,'certificate_sha256':sha(p),'native_D_devices':32,'formal_performance_accepted':False}))
if __name__=='__main__':main()
