#!/usr/bin/env python3
"""Summarize already-bound F08 candidates; reads saved data only."""
from pathlib import Path
from decimal import Decimal
import hashlib,json,collections,statistics
root=Path(__file__).resolve().parents[2]
src=root/'diagnostic-20261029/target-layer-all-kernels-16rank.jsonl'
trip=root/'agents/fusion/f08-actual-triples.json'
expected='0493115eb0a8c9191297f32ee4f845dc69d5d60ec09fce6deb96ec1faeb9a3c4'
assert hashlib.sha256(src.read_bytes()).hexdigest()==expected
base=json.loads(trip.read_text());assert base['triples']==base['unique_scopes']==1280
wanted={(r['ep'],r['scope_id'],fid):(r,i) for r in base['records'] for i,fid in enumerate(r['flow_ids'])}
assert len(wanted)==3840
found={}
with src.open() as fp:
 for line in fp:
  record=json.loads(line)
  if record.get('kind')!='rank':continue
  ep=record['data']['ep']
  for scope in record['data']['layers']:
   sid=scope['layer_scope_id']
   for group in scope['operator_groups']:
    for t in group['tasks']:
     key=(ep,sid,t['flow_id'])
     if key not in wanted:continue
     assert key not in found
     r,i=wanted[key];c=t['csv']
     assert c['Stream ID']==r['stream'] and c['Task ID']==r['tasks'][i]
     start=Decimal(t['device_start_us']);assert start==Decimal(r['starts_us'][i])
     dur=Decimal(t['trace_duration_us']);assert dur>0
     found[key]=(start,dur)
assert len(found)==3840
bucket=collections.defaultdict(lambda:collections.defaultdict(list))
for r in base['records']:
 events=[found[(r['ep'],r['scope_id'],fid)] for fid in r['flow_ids']]; k=str(r['rows'])
 for i,(_,dur) in enumerate(events):bucket[k][f'task_{i+1}_duration_us'].append(dur)
 gap1=events[1][0]-sum(events[0]);gap2=events[2][0]-sum(events[1]);bucket[k]['gap1_us'].append(gap1);bucket[k]['gap2_us'].append(gap2)
 bucket[k]['span_us'].append(events[-1][0]+events[-1][1]-events[0][0]);bucket[k]['sum_duration_us'].append(sum(x[1] for x in events))
 bucket[k]['scatter_fraction_of_three_task_sum'].append(events[2][1]/sum(x[1] for x in events))
 bucket[k]['scatter_is_longest'].append(Decimal(int(events[2][1]>max(events[0][1],events[1][1]))))
def dist(vals):
 z=sorted(vals);return {'count':len(z),'min':str(z[0]),'median':str(statistics.median(z)),'max':str(z[-1]),'negative_count':sum(x<0 for x in z),'sum':str(sum(z))}
report={'source_sha256':expected,'triple_sha256':hashlib.sha256(trip.read_bytes()).hexdigest(),'matched_tasks':len(found),'scope_count':len(base['records']),'by_rows':{rows:{k:dist(vals) for k,vals in v.items()} for rows,v in sorted(bucket.items(),key=lambda x:int(x[0]))},'limitations':['profile-on, not baseline latency','same-stream/shape association does not prove shared tensor or causality','gaps are timestamp differences, not idle time','sums are only per scope; not end-to-end saved time','not an upper/lower limit']}
out=root/'agents/fusion/f08-profile-timing-summary.json';out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
