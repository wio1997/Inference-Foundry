from pathlib import Path
from collections import Counter,defaultdict
import hashlib,json,re,socket
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
P=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/hostprof-20261030T212812Z-360e0c72')
b=(P/'analysis/prepare-field-origins-v12.json').read_bytes();assert hashlib.sha256(b).hexdigest()=='c23b75f3fbd3ae3d3d7152f6882524e899bade53001ea2f1befa09ce943f091f';x=json.loads(b)
rows=[];cohorts=defaultdict(list)
for root in x['prepare_roots']:
 m=root['metadata'];scheduled=m['scheduled_tokens'];cache=m['cached_request_state'];computed={}
 ids=cache.get('req_ids');values=cache.get('num_computed_tokens')
 if isinstance(ids,list) and isinstance(values,list) and len(ids)==len(values):computed.update(zip(ids,values))
 for request in m['new_request_state']:computed[request['req_id']]=request['num_computed_tokens']
 known=all(type(computed.get(k))is int for k in scheduled)
 if not scheduled:phase='ZERO_SCHEDULED_TOKENS'
 elif known and all(computed[k]>=32768 for k in scheduled) and all(v==6 for v in scheduled.values()) and not m['new_request_state']:phase='ALL_SCHEDULED_CONTEXT_GE32768_SPEC5_DECODE_COMPATIBLE'
 elif known and any(computed[k]<32768 for k in scheduled):phase='CONTAINS_PREFILL_CONTEXT_LT32768'
 else:phase='UNRESOLVED_CONTEXT'
 path=x['databases'][root['database_index']]['path'];ep=int(re.search(r'_ep(\d+)_',path).group(1))
 row=dict(root_key=root['root_key'],label=m['label'],pid=m['pid'],ep_rank=ep,data_parallel_rank=m['data_parallel_rank'],phase=phase,total_scheduled_tokens=sum(scheduled.values()),scheduled_tokens=scheduled,computed_tokens={k:computed.get(k) for k in scheduled},selected_name_counts=root['selected_name_counts'])
 if scheduled and known:
  key=hashlib.sha256(json.dumps(dict(dp=m['data_parallel_rank'],state=sorted((k,v,computed[k]) for k,v in scheduled.items())),sort_keys=True).encode()).hexdigest();row['schedule_position_cohort_key']=key;cohorts[key].append(row)
 rows.append(row)
cs=[dict(key=k,roots=len(v),ep_ranks=sorted(r['ep_rank'] for r in v),distinct_ep_ranks=len({r['ep_rank'] for r in v}),phases=sorted({r['phase'] for r in v}),root_keys=[r['root_key'] for r in v],no_repeated_ep=len({r['ep_rank'] for r in v})==len(v)) for k,v in cohorts.items()]
summary=dict(roots=len(rows),phase_counts=dict(Counter(r['phase'] for r in rows)),nonempty_schedule_position_cohorts=len(cs),cohort_size_hist=dict(Counter(c['roots'] for c in cs)),cohorts_with_unique_eight_ep=sum(c['roots']==c['distinct_ep_ranks']==8 for c in cs),
 zero_schedule_roots=sum(r['phase']=='ZERO_SCHEDULED_TOKENS' for r in rows),
 phase_copy_histograms={phase:{name:dict(Counter(r['selected_name_counts'].get(name,0) for r in rows if r['phase']==phase)) for name in ['acl_memcpy_host_to_device','aten::_pin_memory','Event::synchronize']} for phase in sorted({r['phase'] for r in rows})},
 limits=['Cohort key uses exact runtime request IDs+scheduledcounts+CPUcomputed-position snapshots+recordeddp. No nearest timestamp or ordinal join.','CPUposition snapshots can be optimistic;not GPUaccepted-mask/value identity. Repeated same-EP keys remain ambiguous.','Current166population only;does not repair old169other-rank schedules. Empty schedules are not assigned to cross-card cohorts.'],overall_complete=False)
out=P/'analysis/prepare-phase-and-schedule-cohorts-v12.json';assert not out.exists();payload=json.dumps(dict(summary=summary,roots=rows,cohorts=cs),ensure_ascii=False,indent=2)+'\n';out.write_text(payload)
print(json.dumps(dict(summary=summary,full_result=str(out),result_sha256=hashlib.sha256(payload.encode()).hexdigest()),ensure_ascii=False))
