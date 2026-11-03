#!/usr/bin/env python3
"""Verify 40-layer MC2 per-DP duration distributions from frozen local data."""
import csv,hashlib,json,statistics
from collections import Counter
from pathlib import Path
T=Path(__file__).resolve().parents[2]
paths={'calls':T/'captured-layer-attribution/module-call-metrics.json',
 'timeline':T/'mc2-preceding-analysis/rank_layer0_timeline.csv',
 'mc2':T/'mc2-arrival-analysis-v2/rank_target_calls.csv',
 'stage159':T/'agents/diagnosis/layer0-dp-type-counters.json',
 'self':Path(__file__).resolve()}
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(paths['stage159'])=='0482575955229199d9b2c25e1cff2590d82a27d8a91f89e0d8f3418e7be37fad'
calls=[x for x in json.loads(paths['calls'].read_text()) if x['run_in_model'] in (1,2,3)]
with paths['timeline'].open() as f:timeline=[x for x in csv.DictReader(f) if x['interior_runs_1_2_3']=='True']
with paths['mc2'].open() as f:mc2=list(csv.DictReader(f))
lookup={(r['rank'],int(r['target_call_ordinal'])):r for r in mc2}
forward={(r['rank'],int(r['captured_run_in_model'])):int(r['target_forward_ordinal']) for r in timeline}
assert len(calls)==1920 and len(lookup)==len(mc2) and len(forward)==len(timeline)==48
assert len({(c['rank'],c['run_in_model'],c['layer']) for c in calls})==1920
stage=json.loads(paths['stage159'].read_text()); old={(r['rank'],r['run']):r for r in stage['mc2_by_window']};assert len(old)==48
records=[]
for c in calls:
 rank,run,layer=c['rank'],c['run_in_model'],c['layer']; r=lookup[(rank,forward[(rank,run)]*40+layer)]
 assert r['source_layer']==str(layer) and r['model_id']==c['model_id']=='49' and r['stream_id']==c['main_stream_id']
 assert c['window_start_ns']<=int(r['start_ns'])<=int(r['end_ns'])<=c['window_end_ns']
 assert r['physical_input_rows']=='3' and abs(float(r['duration_us'])*1000-(int(r['end_ns'])-int(r['start_ns'])))<.001
 if layer==0:
  s=old[(rank,run)]; assert s['task_id']==r['task_id'] and s['stream']==r['stream_id'] and abs(s['duration_ns']-float(r['duration_us'])*1000)<.001
 records.append({'rank':rank,'run':run,'layer':layer,'dp':rank.split('_')[0],'duration_us':float(r['duration_us']),
   'aic_mac_ratio':float(r['mac_ratio']),'physical_input_rows':int(r['physical_input_rows'])})
out=[]
for run in (1,2,3):
 for layer in range(40):
  groups=[]
  for dp in ('dp0','dp1'):
   rows=[r for r in records if r['run']==run and r['layer']==layer and r['dp']==dp]
   assert len(rows)==8 and len({r['rank'] for r in rows})==8
   groups.append({'dp':dp,'windows':len(rows),'median_duration_us':statistics.median(r['duration_us'] for r in rows),
    'median_reported_aic_mac_ratio':statistics.median(r['aic_mac_ratio'] for r in rows)})
  out.append({'run':run,'layer':layer,'groups':groups,'difference_of_group_medians_us':groups[1]['median_duration_us']-groups[0]['median_duration_us']})
summary=[]
for run in (1,2,3):
 rows=[r for r in out if r['run']==run];zero=rows[0];others=[r['difference_of_group_medians_us'] for r in rows if r['layer']!=0]
 summary.append({'run':run,'layer0':zero,'other39_min_difference_us':min(others),'other39_max_difference_us':max(others),
   'other39_median_difference_us':statistics.median(others),'layer0_is_max_difference':zero['difference_of_group_medians_us']>max(others)})
assert all(r['layer0_is_max_difference'] for r in summary)
print(json.dumps({'scope':'1920 frozen target MC2 events; each run/layer is 8 DP0 and 8 DP1 per-rank elapsed times',
 'inputs_sha256':{k:sha(v) for k,v in paths.items()},'forward_mapping':{str(k):v for k,v in sorted(Counter((int(r['captured_run_in_model']),int(r['target_forward_ordinal'])) for r in timeline).items())},
 'per_layer_run':out,'run_summary':summary,
 'limits':['difference of group medians, not median of paired differences',
 'each duration from one device clock, no cross-rank timestamp subtraction',
 'run ordinal not validated as same request/collective; M=3 is physical shape not equal effective work',
 'AIC MAC field is reported rounded proxy, no AIV or total peak efficiency',
 'sample distributions not independent, not full request or end-to-end critical path',
 'no communication or other internal causality established'], 'no_device_work':True},ensure_ascii=False,indent=2))
