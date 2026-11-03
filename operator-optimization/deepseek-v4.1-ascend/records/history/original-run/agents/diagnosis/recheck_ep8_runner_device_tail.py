#!/usr/bin/env python3
"""Recheck selected saved EP8 scheduler -> runner -> layer -> MC2 evidence; CPU only."""
import hashlib, json
from pathlib import Path
from decimal import Decimal
p=Path(__file__).resolve().parents[2]
inputs={
 'layers': ('diagnostic-20261029/ep8-737-5-layer-mc2.json','afaeab84b04bc49ff371a00c61c07eaf4abb9ea0f7626b0f93bc48f818b2c795'),
 'schedule': ('agents/fusion/f12-selected-scheduler-binding.json','d4798d9751c643d0ef70249dc0d83f41c5a1d89ffbf0821cf1da0dd1883f8016'),
 'flow': ('agents/diagnosis/ep8-l14-l39-flow-intervals.json','8b84dbea06027a71905d0726d252f78fd547c5545db67b064bc72eff4575d80b')}
d={}
for k,(name,h) in inputs.items():
 b=(p/name).read_bytes();assert hashlib.sha256(b).hexdigest()==h,(k,'hash changed');d[k]=json.loads(b)
l=d['layers']; groups=[g for g in d['schedule']['groups'] if g['ep']==8 and g['digest']==l['parent']['schedule_digest']]
assert len(groups)==1
s=groups[0];assert (s['matched_scheduler_events'],s['matched_runner_events'],s['scheduled_total'])==(1,8,8180)
rows=l['rows']; assert len(rows)==40 and [r['layer'] for r in rows]==list(range(40))
assert l['layer_attributed_dispatch_count']==40 and len(l['bad_scopes'])==0 and len(l['ambiguous_or_unattributed'])==0
assert all(r['csv_match_count']==1 for r in rows)
start=Decimal(l['parent']['start_us']);end=Decimal(l['parent']['end_us']); assert end>start
for ix in [14,39]:
 f=d['flow']['layers'][str(ix)];assert f['matched_unique']>0 and f['unmatched']==0 and f['ambiguous']==0
sel=[]
for i in [0,13,14,15,39]:
 r=rows[i]; a=Decimal(r['trace_start_us']);b=a+Decimal(r['trace_duration_us'])
 sel.append(dict(layer=i,task_id=r['trace_task_id'],stream_id=r['trace_physical_stream_id'],host_submission_us_from_runner_begin=str(Decimal(r['host_flow_start_us'])-start),device_start_us_from_runner_begin=str(a-start),device_end_us_from_runner_end=str(b-end),duration_us=r['trace_duration_us']))
nstart=sum(Decimal(r['trace_start_us'])>=end for r in rows)
nend=sum(Decimal(r['trace_start_us'])+Decimal(r['trace_duration_us'])>end for r in rows)
assert (nstart,nend)==(26,27)
for r in rows:assert r['trace_physical_stream_id']=='47'
for x,y in zip(rows,rows[1:]):assert Decimal(x['trace_start_us'])+Decimal(x['trace_duration_us'])<=Decimal(y['trace_start_us'])
print(json.dumps({'input_sha256':{k:v[1] for k,v in inputs.items()},'scheduler_recorded_match':1,'runner_recorded_matches':8,'selected_ep':8,'runner_marker':'737-5','scheduled_total':8180,'scheduler_descriptor_count':len(s['descriptors_sorted_without_identity']),'layers':40,'device_tasks_start_after_runner_end_in_export':nstart,'device_tasks_end_after_runner_end_in_export':nend,'selected':sel,'limits':['Digest equality is content matching within captured records, not unique replay or request identity.','Device rows associated to layer by same-thread scope flow; actual tensor dependency and endpoint critical path unknown.','Host/device timestamp conversion and cross-domain error uncalibrated; apparent cross-domain order is conditional.','Only 40 MC2 tasks in one EP8 worker; other tasks, streams and requests excluded.','Same device trace stream 47 tasks are ordered in exported records; scope correlation and same stream are not full graph dependencies.','Intrusive profile, not production wall time or a forecast of speedup.']},ensure_ascii=False,indent=2))
