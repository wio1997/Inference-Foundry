"""Supplemental SG client audit with source-qualified optional D storage; raw report stays unchanged."""
import argparse,hashlib,json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def percentile(values,p):
 values=sorted(values);at=(len(values)-1)*p;low=math.floor(at);high=math.ceil(at);return values[low]+(values[high]-values[low])*(at-low)
def close(a,b):return math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-8)
def audit(run_id):
 run=ROOT/'runs'/run_id;raw=(run/'formal-result.remote.json').read_bytes();report=json.loads(raw);fw=report['framework'];C=report['concurrency'];namespace=report['namespace'];assert report['project_id']=='glm53' and fw in ['vllm','sglang'] and C in [16,32,64] and report['run_id']==run_id and namespace=='glm53-formal-'+run_id+f'-C{C}'
 expected=json.loads((ROOT/f'runs/20261017T0615-formal-inputs-v3/C{C}-manifest.json').read_text());assert hashlib.sha256((json.dumps(expected['data'],indent=2)+'\n').encode()).hexdigest()==expected['sha256']==report['input_manifest_sha256'];manifest=expected['data'];accepted={(x['prefix_id'],x['variant']):x for x in manifest['formal']}
 assert report['http_connection_policy']=={'upstream_force_close':True,'frontend_force_close':True,'automatic_model_request_retries':False,'revision':'v2'}
 assert fw=='sglang' and report['phase']=='COMPLETE' and len(report['warmup'])==C and len(report['gateway'])==5*C
 import glm53_sglang_cache_annotation_review_local_v1 as cache_review
 review=cache_review.review(report);assert review['pass'] and review['independently_verified_prefix_requests']==4*C
 reviewed={x['request_id']:x for x in review['rows']}
 for index,warm in enumerate(report['warmup']):
  native=warm['native'];request=native['extra_details_data']['glm53_request'];assert warm['pass'] and native['success'] and native['input_tokens']==74400 and native['output_tokens']==1 and request['namespace']==namespace and request['input_sha256_le_u32']==manifest['prefixes'][index]['ids_sha256_le_u32'] and request['native_attempt_setting']==1
  assert warm['request_id']==f'{namespace}-p{index:02d}-warm' and warm['terminal']['gateway']['status']=='D_STREAM_COMPLETE'
 gate=report['cache_gate'];assert gate['pass'] and gate['all_expected_keys_present'] and gate['actual_payload_samples_read'] and gate['prefix_count']==C and gate['SDK_init_bm'] is True and gate['SDK_dram_size']=='0GB' and gate['close_rc']==gate['init_rc']==gate['setup_rc']==0
 assert len(gate['prefixes'])==C and {x['prefix_id'] for x in gate['prefixes']}==set(range(C));payload_reads=0
 for prefix in gate['prefixes']:
  pid=prefix['prefix_id'];assert prefix['prefix_ids_sha256_le_u32']==manifest['prefixes'][pid]['ids_sha256_le_u32']
  keys=581 if fw=='vllm' else 1743;assert prefix['keys']==keys and len(prefix['samples'])==(2 if fw=='vllm' else 6)
  assert {x['key_index'] for x in prefix['samples']}==({0,580} if fw=='vllm' else {0,1,2,1740,1741,1742})
  for sample in prefix['samples']:
   assert sample['read1']['bytes']==sample['read2']['bytes']==sample['expected_bytes'] and sample['read1']['sha256']==sample['read2']['sha256'];assert len(sample['read1']['sha256'])==64;payload_reads+=2
 formal=report['formal'];rows=formal['rows'];events=formal['client_inflight_events'];assert len(rows)==4*C and len(events)==8*C and formal['client_concurrency_limit']==C
 assert {(x['prefix_id'],x['variant']) for x in rows}==set(accepted) and len({x['request_id'] for x in rows})==len(rows)
 assert formal['first_dispatch_perf_counter_ns']==min(x['outer_dispatch_perf_counter_ns'] for x in rows) and formal['last_terminal_perf_counter_ns']==max(x['outer_terminal_perf_counter_ns'] for x in rows)
 seconds=(formal['last_terminal_perf_counter_ns']-formal['first_dispatch_perf_counter_ns'])/1e9;assert seconds>0
 ttft=[];tpot=[];e2e=[];output=0;logical_input=0;noncached_input=0;inflight=0;peak=0;refills=[]
 for event in events:
  inflight+=1 if event['event']=='dispatch' else -1;assert event['event'] in ['dispatch','terminal'] and 0<=inflight<=C and event['client_inflight']==inflight;peak=max(peak,inflight)
 assert inflight==0
 for row in rows:
  native=row['result']['native'];result=row['result'];pid=row['prefix_id'];variant=row['variant'];meta=accepted[(pid,variant)];rid=f'{namespace}-p{pid:02d}-v{variant}';assert row['request_id']==native['uuid']==rid and result['accepted_request_id']==meta['request_id'] and result['input_ids_sha256_le_u32']==meta['ids_sha256_le_u32'] and result['error'] is None
  extra=native['extra_details_data'];usage=extra['glm53_service_usage'];request=extra['glm53_request'];assert native['success'] and native['input_tokens']==usage['prompt_tokens']==request['input_tokens']==80000 and native['output_tokens']==usage['completion_tokens']==request['output_target']==600 and extra['glm53_finish_reason']=='length' and request['native_attempt_setting']==1
  assert request['namespace']==namespace and request['input_sha256_le_u32']==meta['ids_sha256_le_u32'] and result['count_pass'] and result['logical_success'] and result['transport_success'] and reviewed[rid]['logical_prefix_verified']
  gateway=result['terminal']['gateway'];assert gateway['logical_request_id']==rid and gateway['status']=='D_STREAM_COMPLETE' and gateway['prefill_http_status']==gateway['decode_http_status']==200
  valid=native['extra_perf_data']['glm53_valid_output_chunk_times'];assert valid and all(math.isfinite(x) for x in valid) and all(a<=b for a,b in zip(valid,valid[1:]));first=valid[0]-row['outer_dispatch_perf_counter_ns']/1e9;per=(valid[-1]-valid[0])/599;full=(row['outer_terminal_perf_counter_ns']-row['outer_dispatch_perf_counter_ns'])/1e9;assert first>=0 and per>=0 and full>=first and close(first,result['ttft_seconds']) and close(per,result['tpot_seconds']);ttft.append(first);tpot.append(per);e2e.append(full);output+=native['output_tokens'];logical_input+=native['input_tokens']
  if fw=='vllm':
   cached=gateway['prefill_usage']['prompt_tokens_details']['cached_tokens'];assert 74368<=cached<=74400;noncached_input+=80000-cached
  else:noncached_input=None
 for pid in range(C):
  group=sorted((x for x in rows if x['prefix_id']==pid),key=lambda x:x['variant']);assert [x['variant'] for x in group]==[1,2,3,4]
  for a,b in zip(group,group[1:]):assert b['outer_dispatch_perf_counter_ns']>=a['outer_terminal_perf_counter_ns'];refills.append((b['outer_dispatch_perf_counter_ns']-a['outer_terminal_perf_counter_ns'])/1e9)
 warm_terminal=max(x['terminal']['gateway']['gateway_terminal_monotonic_ns'] for x in report['warmup']);assert formal['first_dispatch_perf_counter_ns']>=warm_terminal # Both clocks on CPU43 Linux; no remote P/D clocks used.
 summary=report['summary'];assert close(summary['full_clock_seconds'],seconds) and summary['formal_requests']==4*C and summary['transport_successes']==summary['strict_count_lifecycle_successes']==4*C and summary['cache_verified_strict_successes']==review['original_cache_passes'] and summary['strict_count_lifecycle_failures']==0 and summary['successful_output_tokens']==output==C*4*600 and summary['successful_logical_input_tokens']==logical_input==C*4*80000
 assert close(summary['successful_output_tokens_per_second'],output/seconds) and close(summary['successful_logical_input_tokens_per_second'],logical_input/seconds)
 quantiles={}
 for name,values in [('ttft_seconds',ttft),('tpot_seconds',tpot),('e2e_seconds',e2e)]:
  quantiles[name]={label:percentile(values,p) for label,p in [('P50',.5),('P90',.9),('P99',.99)]};native=summary['quantiles'][name];assert native['measured_requests']==len(rows) and native['missing_requests']==0 and all(close(native[k],v) for k,v in quantiles[name].items())
 sla=quantiles['ttft_seconds']['P50']<4 and quantiles['ttft_seconds']['P90']<12 and quantiles['ttft_seconds']['P99']<30 and quantiles['tpot_seconds']['P50']<.018 and quantiles['tpot_seconds']['P90']<.040;assert summary['aggregate_SLA']['pass']==sla and summary['request_count_and_clock_complete'] and summary['all_formal_prefix_hits_verified']==(review['original_cache_passes']==4*C) and summary['goodput_tokens_per_second'] is None
 assert report['backend_HTTP_attempts']==10*C and len(report['gateway'])==5*C and not report['pending_terminal_ids'];assert report['D_metrics']['sample_count']==report['D_metrics']['complete_sample_count']>0
 steady=summary['native_AISBench_steady'];assert steady['status'] in ['CALCULATED','N/A']
 if steady['status']=='CALCULATED':assert steady['sample_count']==len(steady['sample_request_ids']) and set(steady['sample_request_ids'])<={x['request_id'] for x in rows}
 result={'project_id':'glm53','framework':fw,'concurrency':C,'run_id':run_id,'namespace':namespace,'client_audit_pass':True,'measurement_valid':'PENDING_NATIVE_POSTRUN_AUDIT','formal_requests':len(rows),'warmup_logical_requests':C,'backend_HTTP_attempts':10*C,'successful_backend_inferences':10*C,'all_successful_backend_output_tokens':output+6*C,'formal_successful_output_tokens':output,'full_clock_seconds':seconds,'output_tokens_per_second':output/seconds,'logical_input_tokens_per_second':logical_input/seconds,'noncached_logical_P_input_tokens':noncached_input,'noncached_scope':'Derived as80000-nativePcached tokens; physical recomputation is unknown. Not summed across P and D.','quantiles':quantiles,'aggregate_SLA_pass':sla,'native_AISBench_steady':steady,'actual_client_peak_inflight':peak,'refill_gap_seconds':{'min':min(refills),'max':max(refills),'median':percentile(refills,.5)},'gate_payload_reads':payload_reads,'all_prefix_metadata_and_repeatable_payload_samples_passed':True,'natural_cache_policy_preserved_by_frozen_sources':True,'raw_formal_report_sha256':hashlib.sha256(raw).hexdigest(),'D_metric_archive_sha256':report['D_metrics']['raw_archive_sha256'],'native_postrun_audit_required':True}
 result.update(original_client_baseline_accepted=report['baseline_accepted'],original_client_cache_verified_requests=review['original_cache_passes'],independently_verified_prefix_requests=4*C,cache_review_revision=review['review_revision'],D_storage_null_requests=review['D_storage_null_requests'],raw_report_and_fields_unchanged=True,cache_annotation_review=f'results/sglang-formal-C{C}-v2-cache-annotation-review-v1.json',cache_annotation_review_sha256=digest(ROOT/f'results/sglang-formal-C{C}-v2-cache-annotation-review-v1.json'))
 (ROOT/f'results/{fw}-formal-C{C}-v2-client-audit-v2.json').write_text(json.dumps(result,indent=2)+'\n');return result
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('run_id');args=parser.parse_args();print(json.dumps(audit(args.run_id),indent=2))
