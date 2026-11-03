from pathlib import Path
from collections import Counter,defaultdict
from datetime import datetime,timezone
from decimal import Decimal,InvalidOperation
import json,hashlib,csv,socket,statistics
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
A=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/profile-readonly-20261031')
def pinned(p,sha):
 x=p.stat();b=p.read_bytes();y=p.stat();assert hashlib.sha256(b).hexdigest()==sha;assert (x.st_size,x.st_mtime_ns)==(y.st_size,y.st_mtime_ns);return b
d=json.loads(pinned(A/'ep3-all-compute-origins-v9-v2.json','fdce699c6cf2fdf15b9ba85c5e9e5a5cd7893d2b34a0f69b0321e719a9ef6b9d'))
audit=json.loads(pinned(A/'ep3-origin-source-and-csv-audit-v9-v2.json','7124c7e900ed2bfb4d1ed9a29eb335096a5d8f9a6bd2be85b1f43b9d91453541'))
g=json.loads(pinned(A/'ep3-vector-overlay-and-graph-blocks-v10-v3.json','587b81f07b1b8c3057b9b7486bec6d58ce7aa946909d9385d8d4915bd6012c84'))
assert g['summary']['control_pair_container_check'] and g['summary']['per_block_marker_attention_moe_pass']==440
csv_rows=list(csv.DictReader(pinned(Path(audit['summary']['csv_path']),audit['summary']['csv_sha256']).decode('utf-8-sig').splitlines()))
tm={t['task_record_rowid']:t for t in d['tasks']}
csv_by_task={x['task_record_rowid']:csv_rows[x['csv_row']] for x in audit['csv_compute_matches']+g['vector_overlay']}
metadata=g['metadata_selected_events'];pairs=defaultdict(dict)
for x in metadata:
 if x.get('label')=='NPUModelRunner.execute_model':
  assert x['kind'] in ['call_begin','call_end'] and x['kind'] not in pairs[x['call_id']]
  pairs[x['call_id']][x['kind']]=x
assert all(set(v)=={'call_begin','call_end'} for v in pairs.values())
scheduled_requests={r['request_hmac_sha256'] for v in pairs.values() for r in v['call_begin'].get('schedule',{}).get('requests',[])}
begin_rows=[v['call_begin'] for v in pairs.values()]
initial_eager=[b for b in begin_rows if b.get('schedule',{}).get('scheduled_total') in [8180,4760]]
eager_request_set={r['request_hmac_sha256'] for b in initial_eager for r in b['schedule']['requests']}
scopes={x['ordinal']:x for x in g['host_scope_candidates']};phase_rows=[]
for e in g['graph_executions']:
 candidates=scopes[e['ordinal']]['same_thread_enclosing_pytorch_scopes']
 valid=[]
 for p in candidates:
  prefix='wio_diag:NPUModelRunner.execute_model:'
  if not p['name'].startswith(prefix):continue
  cid=p['name'][len(prefix):]
  if cid not in pairs:continue
  b,z=pairs[cid]['call_begin'],pairs[cid]['call_end'];a=e['CANN_API']
  assert b['pid']==z['pid']==2128 and b['tid']==z['tid']==2128
  assert p['globalTid']==a['globalTid']==(b['pid']<<32)|b['tid']
  assert int(p['startNs'])<=a['startNs']<=a['endNs']<=int(p['endNs'])
  schedule=b.get('schedule',{});rs=schedule.get('requests',[])
  request_ids={r['request_hmac_sha256'] for r in rs}
  decode=bool(rs) and all(r.get('cached_computed_tokens') is not None and r['cached_computed_tokens']>=32768 and r.get('new_computed_tokens') is None and r.get('spec_token_count') is not None and r['scheduled_tokens']==r['spec_token_count']+1 for r in rs)
  valid.append(dict(call_id=cid,annotation_API_rowid=p['api_rowid'],annotation_name=p['name'],schedule_digest_sha256=schedule.get('schedule_digest_sha256'),
    scheduled_total=schedule.get('scheduled_total'),scheduled_requests=len(rs),cached_token_counts=[r.get('cached_computed_tokens') for r in rs],
    scheduled_token_counts=[r['scheduled_tokens'] for r in rs],spec_token_counts=[r.get('spec_token_count') for r in rs],
    request_set_sha256=hashlib.sha256(json.dumps(sorted(request_ids)).encode()).hexdigest(),requests_subset_of_two_eager_cases=request_ids<=eager_request_set,
    all_cached_decode_schedule=decode,metadata_wall_scope_containment=(b['time_ns']<=int(p['startNs']) and int(p['endNs'])<=z['time_ns']),
    association='EXACT_ANNOTATION_CALL_ID_TO_SIDECAR_PLUS_SAME_THREAD_SCOPE_CONTAINS_CANN_SUBMISSION; CANN_TO_DEVICE_CONTROL_PAIR_HAS_DIRECT_CONNECTION'))
 status='BOUND_NONEMPTY_CACHED_DECODE_SCHEDULE' if len(valid)==1 and valid[0]['all_cached_decode_schedule'] else 'BOUND_SCHEDULE_OTHER' if len(valid)==1 else 'SCHEDULE_CONTEXT_UNBOUND'
 phase_rows.append(dict(ordinal=e['ordinal'],status=status,bindings=valid,formal_client_id_attested=False,logical_tokens_not_physical_tensor_rows=True))
eligible={x['ordinal'] for x in phase_rows if x['status']=='BOUND_NONEMPTY_CACHED_DECODE_SCHEDULE'}
def number(x):
 try:
  n=Decimal(str(x).strip());return n if n.is_finite() else None
 except (InvalidOperation,ValueError,TypeError):return None
def proxy(rows,engine,ratio):
 good=[]
 for r in rows:
  c=number(r.get(engine+'_total_cycles'));v=number(r.get(engine+'_'+ratio))
  if c is not None and c>0 and v is not None and 0<=v<=1:good.append((c,v))
 den=sum((c for c,v in good),Decimal(0));num=sum((c*v for c,v in good),Decimal(0))
 return dict(valid_rows=len(good),total_rows=len(rows),cycles=str(den),weighted_work_proxy=str(num),ratio=float(num/den) if den else None)
def union(ts):
 end=None;out=0
 for a,b in sorted((t['startNs'],t['endNs']) for t in ts):
  if end is None or a>end:out+=b-a;end=b
  elif b>end:out+=b-end;end=b
 return out
block_rows=[];outside=[];stream_hist=defaultdict(Counter)
for e in g['graph_executions']:
 for s,signature in e['stream_signatures'].items():stream_hist[s][signature['sha256']]+=1
 assigned=set()
 for block in e['layer_blocks']:
  ids=block['task_record_rowids'];assert not assigned.intersection(ids);assigned.update(ids)
  ts=[tm[i] for i in ids];rs=[csv_by_task[i] for i in ids];moe=[t for t in ts if t['op_type']=='DispatchFFNCombine']
  block_rows.append(dict(execution=e['ordinal'],block_ordinal=block['observed_layer_block_ordinal'],schedule_status=phase_rows[e['ordinal']]['status'],
   records=len(ts),span_ns=block['endNs']-block['startNs'],compute_record_union_ns=union(ts),
   AIC_MAC_counter_proxy=proxy(rs,'aic','mac_ratio'),AIV_VEC_counter_proxy=proxy(rs,'aiv','vec_ratio'),
   corrected_vector_rows=sum(t['op_type'] in ['_quantize_indexer_query_kernel','_prepare_indexer_indices_kernel','_pool_kernel','_cache_update_kernel'] for t in ts),
   type_counts=block['type_counts'],moe_signatures=[{k:t[k] for k in ['input_shapes','output_shapes','input_dtypes','output_dtypes','blockNum','mixBlockNum']} for t in moe]))
 outside_tasks=[tm[i] for i in e['task_record_rowids'] if i not in assigned]
 outside.append(dict(execution=e['ordinal'],schedule_status=phase_rows[e['ordinal']]['status'],records=len(outside_tasks),type_counts=dict(Counter(t['op_type'] for t in outside_tasks)),
   stream_ids=sorted({t['streamId'] for t in outside_tasks}),interval_union_ns=union(outside_tasks),tasks=outside_tasks))
pooled=[]
for k in range(40):
 bs=[b for b in block_rows if b['execution'] in eligible and b['block_ordinal']==k]
 def aggregate(field):
  den=sum(Decimal(b[field]['cycles']) for b in bs);num=sum(Decimal(b[field]['weighted_work_proxy']) for b in bs)
  return float(num/den) if den else None
 pooled.append(dict(observed_layer_block_ordinal=k,scheduled_decode_executions=len(bs),AIC_MAC_counter_proxy=aggregate('AIC_MAC_counter_proxy'),
   AIV_VEC_counter_proxy=aggregate('AIV_VEC_counter_proxy'),median_block_span_ns=statistics.median(b['span_ns'] for b in bs) if bs else None,
   corrected_vector_records=sum(b['corrected_vector_rows'] for b in bs),
   metric_scope='Observed40marker blocks inside scheduled cached-decode calls onEP3. Source-layer identity and full-module membership not independently closed; PMUproxies are not whole-NPU utilization or hardware-peak efficiency.'))
ranked=sorted([x for x in pooled if x['AIC_MAC_counter_proxy'] is not None],key=lambda x:x['AIC_MAC_counter_proxy'])
first=g['graph_executions'][0]
first_groups={k:[b['block_ordinal'] for b in block_rows if b['execution']==0 and k in b['type_counts']] for k in ['VllmAscendQuantLightningIndexerV2','_pool_kernel','_cache_update_kernel']}
summary=dict(created_utc=datetime.now(timezone.utc).isoformat(),source_overlay_sha256='587b81f07b1b8c3057b9b7486bec6d58ce7aa946909d9385d8d4915bd6012c84',
 schedule_context_counts=dict(Counter(x['status'] for x in phase_rows)),bound_graph_ordinals=sorted(eligible),scheduled_model_call_pairs=len(pairs),
 sidecar_total_request_ids=len(scheduled_requests),two_eager_case_request_ids=len(eager_request_set),
 bound_decode_requests_subset_of_two_eager_cases=all(b['requests_subset_of_two_eager_cases'] for x in phase_rows for b in x['bindings'] if x['status']=='BOUND_NONEMPTY_CACHED_DECODE_SCHEDULE'),
 bound_schedule_totals=dict(Counter(str(b['scheduled_total']) for x in phase_rows for b in x['bindings'])),
 marker_blocks=len(block_rows),eligible_schedule_blocks=sum(b['execution'] in eligible for b in block_rows),
 graph_definition_stream_signature_frequencies={s:dict(c) for s,c in stream_hist.items()},observed_indexer_compressor_block_ordinals=first_groups,
 outside_marker_records_per_execution=[x['records'] for x in outside],lowest_observed_AIC_proxy_blocks=ranked[:5],
 whole_module_or_global_lowest_utilization_proved=False,no_target_source_read=True,no_device_or_inference_work=True,whole_modules_closed=0,overall_complete=False,
 limits=['Exact custom annotation/sidecar call_id ties schedule metadata to a containing CPU scope; this is not a direct PYTORCH_API connection or client request_id attestation.',
  'Cached/scheduled metadata proves schedule class, not exact accepted token values, physical valid/padded row masks or routing counts.',
  'Initial graph with no scheduled model scope remains phase-unbound, not automatically dummy/warmup/formal.',
  'PMUcycle-weighted MAC/vector ratios are separate reported counter proxies. They are not comparable to host CPUcores or physical-peak NPUutilization.',
  'Temporal blocks omit root/head/outside work and do not settle every consumer/alias. No latency gain follows from ranking or interval accounting.'])
out=A/'ep3-graph-schedule-and-block-proxies-v10.json';assert not out.exists()
payload=json.dumps(dict(summary=summary,graph_schedule_bindings=phase_rows,block_metrics=block_rows,pooled_block_proxies=pooled,outside_marker_records=outside),ensure_ascii=False,indent=2)+'\n';out.write_text(payload)
print(json.dumps(dict(summary=summary,full_result=str(out),result_sha256=hashlib.sha256(payload.encode()).hexdigest(),
 first_bound_schedules=[x for x in phase_rows if x['bindings']][:2],first_MoE_signature=block_rows[40]['moe_signatures'] if len(block_rows)>40 else []),ensure_ascii=False))
