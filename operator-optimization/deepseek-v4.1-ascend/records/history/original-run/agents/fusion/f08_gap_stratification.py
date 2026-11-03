#!/usr/bin/env python3
"""Reproduce association using existing selected triples, no trace replay or NPU."""
import hashlib, json, statistics
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path
R=Path(__file__).resolve().parents[2]
P=R/'agents/fusion'; S=R/'diagnostic-20261029/target-layer-all-kernels-16rank.jsonl'
T=P/'f08-actual-triples.json'; V=P/'f08-scatter-variation.json'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
trip=json.loads(T.read_text()); var=json.loads(V.read_text()); assert sha(S)==trip['source_sha256']==var['source_sha256']; assert sha(T)==var['triples_sha256']
w={(r['ep'],r['scope_id']):r for r in trip['records'] if r['rows']==8180}; assert len(w)==640
vv={(r['ep'],r['scope']):r for r in var['all_samples_sorted']}; assert len(vv)==640 and set(vv)==set(w)
events={}; selected_scopes={}
with S.open() as fp:
 for line in fp:
  x=json.loads(line)
  if x.get('kind')!='rank':continue
  ep=x['data']['ep']
  for s in x['data']['layers']:
   key=(ep,s['layer_scope_id'])
   if key not in w:continue
   if key in ((3,'2128-15'),(3,'2128-36')):selected_scopes[key]=s
   for g in s['operator_groups']:
    for task in g['tasks']:
     if task['flow_id'] in w[key]['flow_ids']:
      k=(key,task['flow_id']);assert k not in events;events[k]=task
assert len(events)==1920
D=Decimal; rows=[]
for key,r in w.items():
 ts=[events[(key,f)] for f in r['flow_ids']];v=vv[key]
 for i,t in enumerate(ts):
  assert r['tasks'][i]==t['csv']['Task ID'] and r['stream']==t['csv']['Stream ID'] and r['starts_us'][i]==t['device_start_us']
  assert abs(D(t['trace_duration_us'])-D(t['csv']['Duration(us)']))<=D('0.001')
 start=[D(t['device_start_us']) for t in ts]; dur=[D(t['trace_duration_us']) for t in ts]
 gap1=start[1]-start[0]-dur[0];gap2=start[2]-start[1]-dur[1]
 assert D(str(v['duration_us']))==dur[2]
 host=D(ts[2]['host_start_us']); previous_end=start[1]+dur[1]
 rows.append({'ep':r['ep'],'digest':r['digest'],'layer':r['layer'],'scope':r['scope_id'],'duration_us':str(dur[2]),'norm_rotary_gap_us':str(gap1),'rotary_scatter_gap_us':str(gap2),'scatter_host_timestamp_minus_rotary_device_end_us':str(host-previous_end),'class':'short' if dur[2]<50 else 'long' if dur[2]>200 else 'middle'})
def summary(a,k):
 vals=sorted(D(x[k]) for x in a)
 if not vals:return None
 return {'n':len(vals),'min':str(vals[0]),'median':str(statistics.median(vals)),'max':str(vals[-1])}
def by_class(a):
 return {c:{'n':len(b:=[x for x in a if x['class']==c]),'rotary_scatter_gap_us':summary(b,'rotary_scatter_gap_us'),'norm_rotary_gap_us':summary(b,'norm_rotary_gap_us'),'host_minus_device_us':summary(b,'scatter_host_timestamp_minus_rotary_device_end_us')} for c in ('short','middle','long')}
groups=defaultdict(list)
for row in rows:groups[(row['ep'],row['digest'])].append(row)
group={f'{ep}|{digest}':by_class(a) for (ep,digest),a in sorted(groups.items())}
sel=next(a for (ep,d),a in groups.items() if ep==3 and d.startswith('db852'));assert len(sel)==40
overlaps={}
for key,s in selected_scopes.items():
 r=w[key];target=events[(key,r['flow_ids'][2])];start=D(target['device_start_us']);end=start+D(target['trace_duration_us']);found=[]
 for g in s['operator_groups']:
  for t in g['tasks']:
   if t['flow_id']==r['flow_ids'][2]:continue
   a=D(t['device_start_us']);b=a+D(t['trace_duration_us']);d=max(D(0),min(b,end)-max(a,start))
   if d:found.append({'name':g['operator_name'],'track_tid':str(t['device_track_tid']),'overlap_us':str(d),'flow_id':t['flow_id']})
 overlaps[key[1]]={'scope_task_count':sum(len(g['tasks']) for g in s['operator_groups']),'matches':found}
out={'source_sha256':sha(S),'triples_sha256':sha(T),'variation_sha256':sha(V),'classification':{'short_lt_us':50,'long_gt_us':200},'source_timing':'device start and trace duration of the selected events, host timestamp as exported; no clock calibration implied','ep3_group':by_class(sel),'ep3_rows':sorted(sel,key=lambda x:x['layer']),'ep3_example_scope_scatter_overlaps':overlaps,'all_by_class':by_class(rows),'all_by_ep_digest':group,'all_selected_rows':rows,'caveats':['descriptive selected samples, thresholds post hoc; not independent requests','same profiler sources and non-unique data contents; no causal attribution','selected event gaps do not establish device idle or absence of other tasks','cross host/device timestamp subtraction is a displayed difference, not calibrated launch latency']}
(P/'f08-gap-stratification.json').write_text(json.dumps(out,indent=2)+'\n')
print('EP3/db852:',json.dumps(out['ep3_group'],indent=2));print('all:',json.dumps(out['all_by_class'],indent=2))
for k,v in group.items():
 if v['short']['n']:print('short group',k, json.dumps(v))
