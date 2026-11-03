#!/usr/bin/env python3
"""Bounded same-scope intersection audit of already selected HcPost tasks."""
import collections,hashlib,json
from decimal import Decimal as D
from pathlib import Path
R=Path(__file__).resolve().parents[2];p=R/'diagnostic-20261029/target-layer-all-kernels-16rank.jsonl';prior=R/'agents/diagnosis/hc-repeated-scope-anonymous-pairs.json'
assert hashlib.sha256(p.read_bytes()).hexdigest()=='0493115eb0a8c9191297f32ee4f845dc69d5d60ec09fce6deb96ec1faeb9a3c4'
assert hashlib.sha256(prior.read_bytes()).hexdigest()=='6169276fe118ef6803d4f7d91ee6e9c0c245865eb215510699fe77a802c40804' # verify prior selection hash
digests={'db852bff987d2b98d3dc0734e3731ec5528e37dcf6e874860d1ebb4fd12eefc8','1b826883b782ba77814d5bd40fcd3e46c759e380069db2e066ff0e9cb4bce89d'}
keys=('Name','Type','Input Shapes','Output Shapes','Input Data Types','Output Data Types','Input Formats','Output Formats','Accelerator Core','Block Num','Mix Block Num','Stream ID')
prev={g['digest']:g for g in json.loads(prior.read_text())['groups'] if g['name']=='HcPost' and g['digest'] in digests};assert set(prev)==digests
out=collections.defaultdict(list);seen=set()
for line in p.open():
 j=json.loads(line)
 if j.get('kind')!='rank':continue
 for scope in j['data']['layers']:
  dig=scope['scheduler_digest']
  if dig not in digests:continue
  key=(dig,scope['ep'],scope['layer']);assert key not in seen;seen.add(key)
  tasks=[t for group in scope['operator_groups'] for t in group['tasks']];assert len({x['flow_id'] for x in tasks})==len(tasks)
  hc=[t for t in tasks if t['csv']['Type']=='HcPost'];assert len(hc)==2;hc.sort(key=lambda t:D(t['device_start_us']))
  assert all(hc[0]['csv'][k]==hc[1]['csv'][k] for k in keys)
  assert hc[0]['csv']['Input Shapes']=='"1,1023,5120;1,1023,4,5120;1,1023,4;1,1023,4,4"'
  a,b=hc;assert D(a['device_start_us'])+D(a['trace_duration_us'])<=D(b['device_start_us'])
  assert all(t['device_track_pid']==a['device_track_pid'] for t in tasks)
  counts=[]
  for v in hc:
   start=D(v['device_start_us']);end=start+D(v['trace_duration_us']);same=other=0
   for t in tasks:
    if t['flow_id']==v['flow_id']:continue
    s=D(t['device_start_us']);e=s+D(t['trace_duration_us'])
    if s<end and e>start:
     if t['device_track_tid']==v['device_track_tid']:same+=1
     else:other+=1
   counts.append({'same_track':same,'other_track':other})
  out[dig].append({'ep':scope['ep'],'layer':scope['layer'],'scope':scope['layer_scope_id'],
    'first_us':float(a['csv']['Duration(us)']),'second_us':float(b['csv']['Duration(us)']),
    'flow_ids':[a['flow_id'],b['flow_id']],'overlap_counts':counts,'scope_selected_task_count':len(tasks)})
assert len(seen)==640 and set(out)==digests
summary={}
for dig,rows in out.items():
 assert len(rows)==320
 old=prev[dig];assert sorted((r['first_us'],r['second_us']) for r in rows)==sorted((r['first_us'],r['second_us']) for r in old['pairs'])
 assert all(c['same_track']==c['other_track']==0 for row in rows for c in row['overlap_counts'])
 summary[dig]={'scope_count':len(rows),'hcpost_task_count':len(rows)*2,'ranks':dict(collections.Counter(str(r['ep']) for r in rows)),
  'tasks_with_same_track_overlap':sum(c['same_track']>0 for row in rows for c in row['overlap_counts']),
  'tasks_with_other_track_overlap':sum(c['other_track']>0 for row in rows for c in row['overlap_counts']),
  'first_median_us':old['first_median_us'],'second_median_us':old['second_median_us'],'pair_ratio_median':old['ratio_median']}
result={'source_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'prior_sha256':hashlib.sha256(prior.read_bytes()).hexdigest(),'summary':summary,'scope_rows':dict(out),
 'limitations':['Only tasks retained by same-scope host-flow and kernel-CSV matching on same device track PID; no full-device claim.','No direct intersection among selected events is not proof of no contention from other scopes, streams, unrecorded processes, asynchronous activity or internal waits.','Matching physical metadata does not prove tensor identity, equal inputs or executed historical binary.','Task interval difference not end-to-end latency or savings; profile disturbances and replay association remain.']}
target=R/'agents/diagnosis/hcpost-scope-overlap.json';target.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(summary,indent=2));print('output_sha256',hashlib.sha256(target.read_bytes()).hexdigest())
