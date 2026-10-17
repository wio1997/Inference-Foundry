"""Audit already captured native SDMA/full-batch cache/MTP/graph evidence."""
import collections,hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def read(path):return json.loads((ROOT/path).read_text())
def sha(path):return hashlib.sha256((ROOT/path).read_bytes()).hexdigest()
def main():
 rotation_path='runs/20261017T1785-vllm-key-rotation-audit-v1/node-43.json';rotation_outer=read(rotation_path);assert rotation_outer['rc']==0;rotation=json.loads(rotation_outer['stdout']);assert rotation['pass']
 run='runs/20261017T1783-vllm-shared-long-native-v2';before='runs/20261017T1777-vllm-shared-native-checkpoint-v1';client=read('runs/20261017T1779-vllm-shared-long-semantic-v2/semantic-result.local.json');assert client['semantic_pass'];namespace='glm53-sem-vllm-shared-v2-20261017T1779';objects={};source={};all_init=[];pd=[];mc={};runtime={};graphs={};errors={};native_SDK={}
 for node in [30,31,43,45]:
  file=run+f'/node-{node}.json';source[file]=sha(file);outer=read(file);assert outer['rc']==0;o=json.loads(outer['stdout']);objects[node]=o;runtime[str(node)]=o['container_id'];assert o['/health']['status']==200 and o['container_state']['Running'] and len(o['owned_npu_process_rows'])==17 and not o['profiler_artifacts']
  body=o['/metrics']['body'];assert hashlib.sha256(body.encode()).hexdigest()==o['/metrics']['sha256']
  gauges=[float(x.rsplit(' ',1)[1]) for x in body.splitlines() if x.startswith(('vllm:num_requests_running{','vllm:num_requests_waiting{'))];assert gauges and all(x==0 for x in gauges)
  for stream in o['logs'].values():assert stream['full_archive_complete']
  events=[]
  for a in o['persistent_audit'].values():
   raw=''.join(json.dumps(r,sort_keys=True)+'\n' for r in a['rows']).encode();assert len(raw)==a['bytes'] and hashlib.sha256(raw).hexdigest()==a['sha256'];events.extend(a['rows'])
  initialized={r['pid']:r for r in events if r['event']=='initialize_complete'};assert len(initialized)==16 and all(r['protocol']=='SDMA' and r['protocol_value']==1 and r['rc']==0 for r in initialized.values());all_init.extend(initialized.values())
  for r in events:
   if r['event']=='model_state_read_complete' and any(namespace in name for name in r['request_ids']):assert r['rc']==0;pd.append(r|{'node':node})
  if node in [30,31]:
   batches=[r for r in events if r['event']=='native_get_batch_complete'];assert len(batches)==8 and len({r['pid'] for r in batches})==8
   case=0 if node==30 else 1;known=client['metadata_gates'][str(case)]['metadata'];assert known['pass'] and known['close_rc']==0
   order=rotation['cases'][case];assert order['unique_physical_keys']==581 and order['TP_read_orderings'][0]['key_list_sha256']==known['known_keys_sha256']
   for r in batches:
    tp=initialized[r['pid']]['device_id']%8
    assert r['key_count']==r['total_key_count']==581 and r['first_key_index']==0 and r['native_results']==[0]*581 and r['native_call_count']==1 and r['batch_limit_blocks'] is None and r['retry_attempts']==0 and r['fallback'] is False and r['payload_bytes_requested']==7186923520 and r['key_list_sha256']==order['TP_read_orderings'][tp]['key_list_sha256']
   mc[str(node)]={'actual_model_worker_reads':8,'keys_per_full_native_call':581,'native_call_count_per_worker':1,'native_result_codes':[0],'retry_attempts':0,'fallback':False,'canonical_key_list_sha256':known['known_keys_sha256'],'unique_physical_prefix_keys':581,'unique_physical_prefix_bytes':7186923520,'all8_native_TP_rotated_key_orders_verified':True,'descriptor_payload_bytes':sum(r['payload_bytes_requested'] for r in batches),'read_duration_ns':{'min':min(r['terminal_monotonic_ns']-r['begin_monotonic_ns'] for r in batches),'max':max(r['terminal_monotonic_ns']-r['begin_monotonic_ns'] for r in batches)}}
  if node in [43,45]:
   previous=json.loads(read(before+f'/node-{node}.json')['stdout']);seen=set(previous['logs']['stdout']['capacity_lines']);rows=[];new=[]
   for line in o['logs']['stdout']['capacity_lines']:
    match=re.search(r'\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*FULL\s*\|\s*(\d+)\s*\|',line)
    if match:
     nums=list(map(int,match.groups()));assert nums[1] in [6,12,18,24] and nums[3]>0;rows.append(nums)
     if line not in seen:new.append(nums)
   assert rows
   if node==43:assert new,'Long requests did not produce native FULL graph rows'
   graphs[str(node)]={'native_FULL_runtime_table_rows':len(rows),'new_FULL_rows_since_pre_long_checkpoint':len(new),'observed_padded_token_sizes':sorted({x[1] for x in rows}),'scope':'Actual native API graph runtime stats; no Prometheus graph series required. D43 handled long routes0..3; D45 graph execution is independently covered by short requests.'}
  errorlines=[x for stream in o['logs'].values() for x in stream['error_lines']];assert all('smem_trans_entry.cpp:' in x and ('QueryRole' in x or 'OnAddSlices' in x) for x in errorlines);errors[str(node)]={'count':len(errorlines),'lines':errorlines,'scope':'Preserved native connection/rank diagnostic rows, all audited transfer completions successful; exact internal retry cause not established.'}
  counters={}
  for snap in o['native_SDK_small_file_snapshots']:
   assert snap['rc']==0
   if snap['container_id']!=o['container_id']:continue
   for f in snap['files']:
    assert f['complete_line_ending'] and hashlib.sha256(f['text'].encode()).hexdigest()==f['sha256'];pid=int(re.search(r'ptracer_(\d+)\.dat$',f['path']).group(1))
    if pid not in initialized:continue
    for line in f['text'].splitlines():
     parts=line.split()
     if len(parts)>8 and parts[2]=='TP_HYBM_SDMA_BATCH_GD_TO_LD':
      values=list(map(int,parts[3:7]));assert values[0]==values[1] and values[2]==0
      item=counters.setdefault(pid,{'BEGIN':0,'GOOD_END':0,'BAD_END':0,'last_ON_FLY':None});item['BEGIN']+=values[0];item['GOOD_END']+=values[1];item['BAD_END']+=values[2];item['last_ON_FLY']=values[3]
  if node in [43,45]:assert len(counters)==16 and all(x['last_ON_FLY']==0 for x in counters.values())
  native_SDK[str(node)]={'workers_with_PD_SDMA_read_counters':len(counters),'GOOD_END':sum(x['GOOD_END'] for x in counters.values()),'BAD_END':sum(x['BAD_END'] for x in counters.values()),'last_ON_FLY_each':0 if counters and all(x['last_ON_FLY']==0 for x in counters.values()) else None}
 assert len(all_init)==64 and len(pd)==12 and sum(r['payload_bytes_completed'] for r in pd)==90646773760
 assert sum(x['GOOD_END'] for x in native_SDK.values())==48 # short32 + priorfailedhelperattempt4 + currentlong12; no startup attribution.
 artifacts=objects[43]['long_client_artifacts'];index=read('runs/20261017T1779-vllm-shared-long-semantic-v2/artifact-index.json')
 metrics={}
 for name,a in artifacts.items():
  assert hashlib.sha256(a['text'].encode()).hexdigest()==a['sha256'] and len(a['text'].encode())==a['bytes']
  if name in index:assert a['sha256']==index[name]['sha256'] and a['bytes']==index[name]['bytes']
  if name.startswith('metrics-'):metrics[name]=a['text']
 assert len(metrics)==28
 def val(tag,node,name,engine=None,source=None):
  rows=[]
  for line in metrics[f'metrics-{tag}-node{node}.txt'].splitlines():
   if not line.startswith(name+'{'):continue
   if engine is not None and f'engine="{engine}"' not in line:continue
   if source is not None and f'source="{source}"' not in line:continue
   rows.append(float(line.rsplit(' ',1)[1]))
  assert rows,(tag,node,name,engine,source);return sum(rows)
 cache=[]
 for case,node,engine in [(0,30,1),(1,31,3)]:
  a=f'case{case}-prefix-warmup-PD';b=f'case{case}-cross-DP-storage-PD';delta={source:val(b,node,'vllm:prompt_tokens_by_source_total',engine,source)-val(a,node,'vllm:prompt_tokens_by_source_total',engine,source) for source in ['local_compute','local_cache_hit','external_kv_transfer']};assert delta=={'local_compute':5632,'local_cache_hit':0,'external_kv_transfer':74368};cache.append({'semantic_case':case,'P_global_DP':engine,'actual_source_token_delta':delta})
 mtp=[];previous='before'
 for case in [0,1]:
  for label in ['cold-PD','prefix-warmup-PD','cross-DP-storage-PD']:
   tag=f'case{case}-{label}';delta={field:sum(val(tag,n,f'vllm:spec_decode_num_{field}_total')-val(previous,n,f'vllm:spec_decode_num_{field}_total') for n in [43,45]) for field in ['drafts','draft_tokens','accepted_tokens']}
   if label!='prefix-warmup-PD':assert delta['drafts']>0 and delta['draft_tokens']==5*delta['drafts'] and delta['accepted_tokens']>0
   mtp.append({'case':tag,**delta,'actual_accepted_fraction':delta['accepted_tokens']/delta['draft_tokens'] if delta['draft_tokens'] else None});previous=tag
 result={'project_id':'glm53','framework':'vllm','shared_run_id':'20261017T1766-vllm-shared-pool-v4','runtime_CIDs':runtime,'pass':True,'qualification':'TWO_80K_FACTS_NATIVE_FULL_BATCH_KNOWN_KEY_PAYLOAD_PD_SDMA_MTP_FULL_GRAPH_PASSED','MC_full_native_reads':mc,'MC_total_descriptor_payload_bytes':sum(x['descriptor_payload_bytes'] for x in mc.values()),'new_long_PD_worker_read_completions':len(pd),'new_long_PD_descriptor_bytes':sum(r['payload_bytes_completed'] for r in pd),'physical_wire_bytes':None,'SDMA_initialized_workers':64,'cross_P_cache_source_delta':cache,'actual_native_MTP_metric_deltas':mtp,'actual_native_graph_runtime':graphs,'native_SDK_PD_counters_all_actual_HTTP':native_SDK,'preserved_connection_diagnostics':errors,'formal_requests':0,'baseline_accepted':False,'limits':['Two frozen factual guards and short output reference, no broad numerical equivalence or whole-output bitwise gate.','Descriptor bytes include logical worker replication; physical link bytes unknown.','Actual native full-batch reads plus repeatable keys validate model read path; formal every-prefix host payload sampling is independently required before formal clock.','Cross-host wallclock/monotonic clocks are not subtracted; causal request IDs and same-host intervals only.'],'source_evidence_sha256':source}
 source[rotation_path]=sha(rotation_path)
 target='results/vllm-shared-long-native-v2.json';(ROOT/target).write_text(json.dumps(result,indent=2)+'\n')
 source[rotation_path]=sha(rotation_path)
 proofs=['results/vllm-shared-pool-capacity-v4.json','results/vllm-shared-startup-v1.json','results/vllm-shared-short-matrix-v1.json','results/vllm-shared-short-native-v1.json','results/vllm-shared-long-semantic-v2.json',target]
 for path in proofs:assert read(path)['pass']
 cert={'project_id':'glm53','framework':'vllm','shared_run_id':result['shared_run_id'],'runtime_CIDs':runtime,'pass':True,'qualification':'CURRENT_NATIVE_FUNCTIONAL_READINESS_ONLY_FORMAL_ALLPREFIX_GATE_REQUIRED','evidence_sha256':{path:sha(path) for path in proofs},'formal_requests':0,'baseline_accepted':False,'profiling_allowed':False,'native_CLI_proof':'Unchanged SDK/framework/native args reused by source/config/image hashes as explicitly documented in capacity-v4; actual runtime/startup/requests independently current.'}
 (ROOT/'results/vllm-formal-native-readiness-v1.json').write_text(json.dumps(cert,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['preserved_connection_diagnostics','source_evidence_sha256','limits']},indent=2));print('READINESS_SHA',sha('results/vllm-formal-native-readiness-v1.json'))
if __name__=='__main__':main()
