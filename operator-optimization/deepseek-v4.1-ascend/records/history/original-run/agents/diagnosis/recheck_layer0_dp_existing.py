#!/usr/bin/env python3
"""Join 48 frozen layer0 windows to MC2 task summaries without accessing NPU."""
import csv,hashlib,json,statistics
from pathlib import Path
T=Path(__file__).resolve().parents[2]
files={'calls':T/'captured-layer-attribution/module-call-metrics.json',
 'timeline':T/'mc2-preceding-analysis/rank_layer0_timeline.csv',
 'tasks':T/'mc2-arrival-analysis-v2/rank_target_calls.csv',
 'self':Path(__file__).resolve()}
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
cs={(x['rank'],x['run_in_model']):x for x in json.loads(files['calls'].read_text()) if x['layer']==0 and x['run_in_model'] in (1,2,3)}
with files['timeline'].open() as f:ss=[x for x in csv.DictReader(f) if x['interior_runs_1_2_3']=='True']
with files['tasks'].open() as f:rr=list(csv.DictReader(f))
tasks={(r['rank'],int(r['target_call_ordinal'])):r for r in rr}
assert len(tasks)==len(rr) and len(cs)==len(ss)==48
samples=[]
for s in ss:
 c=cs[(s['rank'],int(s['captured_run_in_model']))];r=tasks[(s['rank'],int(s['target_forward_ordinal'])*40)]
 assert c['model_id']==r['model_id']=='49' and c['window_start_ns']==int(s['layer_start_ns'])
 assert r['source_layer']=='0' and r['physical_input_rows']=='3'
 assert int(r['start_ns'])==int(s['mc2_start_ns']) and int(r['end_ns'])==int(s['mc2_end_ns'])
 assert abs(float(r['duration_us'])-float(s['mc2_duration_us']))<.001
 assert c['window_start_ns']<=int(r['start_ns'])<=int(r['end_ns'])<=c['window_end_ns']
 samples.append({'dp':s['dp'],'rank':s['rank'],'run':int(s['captured_run_in_model']),
  'layer_window_us':c['window_ns']/1000,'mc2_duration_us':float(r['duration_us']),
  'mc2_mac_ratio':float(r['mac_ratio']),'mc2_mte2_ratio':float(r['mte2_ratio']),
  'aic_total_cycles':int(c['aic_cycle_denominator']),'aiv_total_cycles':int(c['aiv_cycle_denominator']),
  'exported_task_union_fraction':c['known_task_union_ns']/c['window_ns'],
  'attributed_row_count':c['attributed_rows'],'physical_mc2_first_dim':3})
def stats(xs):return {'min':min(xs),'median':statistics.median(xs),'max':max(xs)}
fields=['layer_window_us','mc2_duration_us','mc2_mac_ratio','mc2_mte2_ratio','aic_total_cycles','aiv_total_cycles','exported_task_union_fraction','attributed_row_count']
summary=[]
for dp in ('dp0','dp1'):
 for run in (None,1,2,3):
  rows=[x for x in samples if x['dp']==dp and (run is None or x['run']==run)]
  assert len(rows)==(24 if run is None else 8)
  summary.append({'dp':dp,'run':run,'windows':len(rows),'statistics':{k:stats([x[k] for x in rows]) for k in fields}})
print(json.dumps({'inputs_sha256':{k:sha(v) for k,v in files.items()},'samples':samples,'summary':summary,
 'scope':'48 source-validated model49 layer0 windows; per-rank event joins, no cross-rank timestamp subtraction',
 'limitations':['counts and duration distributions do not prove DP causes disparity','M=3 is physical input first dimension, not equal semantic workload','aic/aiv cycle denominators are per-record proxies, not overall utilization','coverage is union of attributed CSV task intervals, not busy/idle or causal critical path','grouped medians of different fields do not describe one sample','task types not decomposed by DP in current local aggregates'],'no_device_work':True},ensure_ascii=False,indent=2))
