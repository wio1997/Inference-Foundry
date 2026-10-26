"""Accept a completed vLLM measurement only with current native postrun evidence."""
import argparse,collections,hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def read(path):return json.loads((ROOT/path).read_text())
def sha(path):return hashlib.sha256((ROOT/path).read_bytes()).hexdigest()
def unpack(capture,node):
 path=f'runs/{capture}/node-{node}.json';outer=read(path);assert outer['rc']==0;o=json.loads(outer['stdout']);events=[]
 for a in o['persistent_audit'].values():
  raw=''.join(json.dumps(x,sort_keys=True)+'\n' for x in a['rows']).encode();assert len(raw)==a['bytes'] and hashlib.sha256(raw).hexdigest()==a['sha256'];events.extend(a['rows'])
 return path,o,events

def audit(run_id,capture_id,before_id):
 report=read(f'runs/{run_id}/formal-result.remote.json');C=report['concurrency'];namespace=report['namespace'];client_path=f'results/vllm-formal-C{C}-v2-client-audit-v1.json';client=read(client_path);assert client['client_audit_pass'] and report['baseline_accepted'] and report['framework']=='vllm'
 settings=read(f'config/formal-runs-v2/vllm-C{C}.json');assert settings['run_id']==run_id
 allowed={r['logical_request_id']:r for r in report['gateway']};assert len(allowed)==5*C
 coverage={rid:set() for rid in allowed};mc={};pd=[];period_PD_reads=0;source={client_path:sha(client_path)};runtime={};objects={};error_counts={};SDK_counts={};cache_delta={};MTP={};graphs={}
 def metric(body,name,source=None):
  values=[]
  for line in body.splitlines():
   if line.startswith(name+'{') and (source is None or f'source="{source}"' in line):values.append(float(line.rsplit(' ',1)[1]))
  assert values,(name,source);return sum(values)
 for node in [30,31,43,45]:
  path,o,events=unpack(capture_id,node);before_path,previous,old=unpack(before_id,node);source[path]=sha(path);source[before_path]=sha(before_path);objects[node]=o;runtime[str(node)]=o['container_id'];assert o['container_id']==settings['runtime_containers'][str(node)]['container_id'] and o['container_state']['Running'] and o['/health']['status']==200 and len(o['owned_npu_process_rows'])==17 and not o['profiler_artifacts']
  assert o['actual_model_identity']['namespace_pid']==93 and o['provider_phase']=='READY'
  for log in o['logs'].values():assert log['full_archive_complete']
  for body in [o['/metrics'],previous['/metrics']]:assert hashlib.sha256(body['body'].encode()).hexdigest()==body['sha256']
  gauges=[float(x.rsplit(' ',1)[1]) for x in o['/metrics']['body'].splitlines() if x.startswith(('vllm:num_requests_running{','vllm:num_requests_waiting{'))];assert gauges and all(v==0 for v in gauges)
  initial={r['pid']:r for r in events if r['event']=='initialize_complete'};assert len(initial)==16 and all(x['rc']==0 and x['protocol']=='SDMA' and x['protocol_value']==1 for x in initial.values())
  oldset={json.dumps(x,sort_keys=True) for x in old};new=[x for x in events if json.dumps(x,sort_keys=True) not in oldset]
  for event in new:
   if event['event']=='read_complete':assert event['rc']==0 and event['payload_bytes_completed']==event['payload_bytes_requested']>0;period_PD_reads+=1
   if event['event']!='model_state_read_complete':continue
   assert event['rc']==0 and event['payload_bytes_completed']>0 and node in [43,45];ids=event['request_ids'];assert ids
   for native_rid in ids:
    matches=[rid for rid in allowed if rid+'.D' in native_rid];assert len(matches)==1,native_rid;rid=matches[0];coverage[rid].add((node,event['pid']));expected_DP=allowed[rid]['route']['D']['global_dp'];assert expected_DP==(0 if node==43 else 8)+initial[event['pid']]['device_id']//2
   pd.append(event|{'node':node})
  if node in [30,31]:
   batches=[x for x in new if x['event']=='native_get_batch_complete'];assert batches
   for x in batches:assert x['native_call_count']==1 and x['batch_limit_blocks'] is None and x['retry_attempts']==0 and x['fallback'] is False and x['key_count']==x['total_key_count'] and x['native_results']==[0]*x['key_count'] and x['payload_bytes_requested']>0
   mc[str(node)]={'actual_model_batch_reads':len(batches),'complete_key_count_distribution':dict(collections.Counter(x['key_count'] for x in batches)),'native_return_codes':[0],'retry_attempts':0,'fallback':False,'descriptor_bytes':sum(x['payload_bytes_requested'] for x in batches),'same_key_data_may_be_read_by_multiple_TP_workers':True,'physical_wire_bytes':None,'duration_ns':{'min':min(x['terminal_monotonic_ns']-x['begin_monotonic_ns'] for x in batches),'max':max(x['terminal_monotonic_ns']-x['begin_monotonic_ns'] for x in batches)}}
  before_errors=[x for log in previous['logs'].values() for x in log['error_lines']];errors=[x for log in o['logs'].values() for x in log['error_lines']];assert errors==before_errors,'New native error diagnostics require review';error_counts[str(node)]={'preserved_existing_errors':len(errors),'new_error_lines':0}
  if node in [43,45]:
   rows=[line for line in o['logs']['stdout']['capacity_lines'] if re.search(r'\|\s*FULL\s*\|\s*[1-9][0-9]*\s*\|',line)];newrows=[line for line in rows if line not in set(previous['logs']['stdout']['capacity_lines'])];assert newrows;graphs[str(node)]={'new_native_FULL_runtime_stats_rows':len(newrows),'source':'Native CUDAGraph runtime stats after actual formal requests'}
   drafts={k:metric(o['/metrics']['body'],f'vllm:spec_decode_num_{k}_total')-metric(previous['/metrics']['body'],f'vllm:spec_decode_num_{k}_total') for k in ['drafts','draft_tokens','accepted_tokens']};assert drafts['drafts']>0 and drafts['draft_tokens']==5*drafts['drafts'] and drafts['accepted_tokens']>0;MTP[str(node)]=drafts
  for label,name in [('P' if node in [30,31] else 'D','vllm:prompt_tokens_by_source_total')]:
   cache_delta[str(node)]={source:metric(o['/metrics']['body'],name,source)-metric(previous['/metrics']['body'],name,source) for source in ['local_compute','local_cache_hit','external_kv_transfer']}
  def sdk_PD_count(obj,model_pids):
   total=0
   for snap in obj['native_SDK_small_file_snapshots']:
    assert snap['rc']==0
    if snap['container_id']!=obj['container_id']:continue
    for f in snap['files']:
     assert f['complete_line_ending'] and hashlib.sha256(f['text'].encode()).hexdigest()==f['sha256'];match=re.search(r'ptracer_(\d+)\.dat$',f['path'])
     if not match or int(match.group(1)) not in model_pids:continue
     rows=[]
     for line in f['text'].splitlines():
      parts=line.split()
      if len(parts)>8 and parts[2]=='TP_HYBM_SDMA_BATCH_GD_TO_LD':rows.append(list(map(int,parts[3:7])))
     if rows:assert sum(x[0] for x in rows)==sum(x[1] for x in rows) and all(x[2]==0 for x in rows) and rows[-1][3]==0;total+=sum(x[1] for x in rows)
   return total
  SDK_counts[str(node)]={'GOOD_END_delta':sdk_PD_count(o,initial)-sdk_PD_count(previous,initial),'BAD_END':0,'last_ON_FLY':0 if node in [43,45] else None}
 assert all(len(pids)==2 for pids in coverage.values()) and len(pd)==period_PD_reads
 assert sum(x['GOOD_END_delta'] for x in SDK_counts.values())==period_PD_reads
 actualgate=objects[30]['actual_formal_cache_gate'];assert actualgate['report']==report['cache_gate'];gatewire=read(f'runs/{run_id}/cache-gate-native.json');gatepayload=json.loads(gatewire['stdout']);assert gatewire['rc']==gatepayload['query_rc']==0 and actualgate['sha256']==gatepayload['report_sha256']
 metric_audit=objects[43]['independent_formal_metric_archive_audit'];assert metric_audit['all_native_bodies_SHA_and_DP_gauges_recomputed'] and not metric_audit['errors'] and metric_audit['sample_pairs']==metric_audit['complete_pairs'] and metric_audit['archive_sha256']==client['D_metric_archive_sha256'];assert objects[43]['formal_report_identity']['sha256']==client['raw_formal_report_sha256']
 # Native per-source totals cover real warmups and formal, not P+D double-counted inputs.
 input_per_role=C*74400+4*C*80000
 assert sum(sum(cache_delta[str(n)].values()) for n in [30,31])==sum(sum(cache_delta[str(n)].values()) for n in [43,45])==input_per_role
 native={'project_id':'glm53','framework':'vllm','concurrency':C,'run_id':run_id,'pass':True,'runtime_CIDs':runtime,'native_PD_read_batches':len(pd),'PD_logical_request_worker_coverage':sum(len(x) for x in coverage.values()),'PD_logical_backend_requests_covered':len(coverage),'PD_descriptor_bytes':sum(x['payload_bytes_completed'] for x in pd),'PD_descriptor_byte_distribution':dict(collections.Counter(x['payload_bytes_completed'] for x in pd)),'MC_native_reads':mc,'physical_wire_bytes':None,'native_PD_SDK_counters_delta':SDK_counts,'native_source_token_deltas_including_warmup':cache_delta,'native_MTP_deltas_including_warmup':MTP,'native_FULL_graph_execution':graphs,'independent_complete_D_metrics_archive':metric_audit,'every_prefix_actual_payload_gate_and_close_verified':True,'all_formal_input_SHA_count_cache_fullclock_verified_by_client_audit':True,'preserved_connection_diagnostics':error_counts,'source_evidence_sha256':source}
 native_path=f'results/vllm-formal-C{C}-v2-native-v2.json';(ROOT/native_path).write_text(json.dumps(native,indent=2)+'\n')
 summary=client|{'measurement_valid':True,'native_postrun_audit_required':False,'native_postrun_evidence':native_path,'native_postrun_evidence_sha256':sha(native_path),'shared_run_id':settings['shared_run_id'],'runtime_CIDs':runtime,'accepted_baseline':True,'formal_measurement_accepted_SLA_separate':True,'sampled_peak_D_running':metric_audit['sampled_peak_total_D_running'],'native_proofs':{'actual_allprefix_payload_samples':True,'actual_PD_SDMA':True,'actual_native_MemCache_reads':True,'actual_MTP5':True,'actual_FULL_graph_execution':True,'complete_D16_native_metrics':True}}
 certificate=f'results/vllm-formal-C{C}-v2-measurement-v1.json';(ROOT/certificate).write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps({'certificate':certificate,'sha256':sha(certificate),'measurement_valid':True,'aggregate_SLA_pass':summary['aggregate_SLA_pass'],'output_tokens_per_second':summary['output_tokens_per_second'],'native_PD_read_batches':len(pd),'native_PD_bytes':native['PD_descriptor_bytes'],'sampled_peak_D_running':summary['sampled_peak_D_running']},indent=2));return summary
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('run_id');parser.add_argument('capture_id');parser.add_argument('before_id');args=parser.parse_args();audit(args.run_id,args.capture_id,args.before_id)
