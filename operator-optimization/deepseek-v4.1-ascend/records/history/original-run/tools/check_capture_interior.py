"""Check robustness using predeclared common interior runs1,2,3; CPU only."""
from pathlib import Path
from collections import defaultdict
from decimal import Decimal
import bisect,csv,json
T=Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41');O=T/'captured-layer-attribution';p=O/'module-call-metrics.json';assert p.stat().st_size<20*1024*1024
calls=json.loads(p.read_text());selected_runs={1,2,3};byrank=defaultdict(list)
for c in calls:byrank[c['rank']].append(c)
assert len(byrank)==16
for rank,cs in byrank.items():assert selected_runs<=set(c['run_in_model'] for c in cs)
interior=[c for c in calls if c['run_in_model'] in selected_runs]
def rank_layers(cs):
 out=[]
 for layer in range(40):
  rows=[c for c in cs if c['layer']==layer];den=sum((Decimal(c['aic_cycle_denominator']) for c in rows),Decimal(0));num=sum((Decimal(c['aic_mac_cycle_proxy_numerator']) for c in rows),Decimal(0));duration=sum(c['window_ns'] for c in rows)
  out.append({'layer':layer,'calls':len(rows),'rank_count':len(set(c['rank'] for c in rows)),'cycle_weighted_reported_mac_ratio':float(num/den) if den else None,'mean_layer_window_us':duration/len(rows)/1000,'max_layer_window_us':max(c['window_ns'] for c in rows)/1000})
 return sorted(out,key=lambda x:x['cycle_weighted_reported_mac_ratio'] if x['cycle_weighted_reported_mac_ratio'] is not None else 2)
by_run=[]
for run in sorted(set(c['run_in_model'] for c in calls)):
 cs=[c for c in calls if c['run_in_model']==run]
 by_run.append({'run_in_model':run,'lowest':rank_layers(cs)[:3],'sample_rank_count':len(set(c['rank'] for c in cs))})
root=Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-20261028/diagnostic/runtime/diagnostic-traces');aggs=defaultdict(lambda:{'calls':0,'duration_us':Decimal(0),'aic_cycles':Decimal(0),'mac_proxy':Decimal(0),'mte2_proxy':Decimal(0),'max_us':Decimal(0)})
def dec(x):
 try:return Decimal(x)
 except:return Decimal(0)
for rank,cs in byrank.items():
 cs.sort(key=lambda c:c['window_start_ns']);starts=[c['window_start_ns'] for c in cs]
 for row in csv.DictReader((root/rank/'ASCEND_PROFILER_OUTPUT/kernel_details.csv').open()):
  if row['Model ID'] in ['4294967295','N/A','']:continue
  a=int(Decimal(row['Start Time(us)'].strip())*1000);b=a+int(Decimal(row['Duration(us)'])*1000);i=bisect.bisect_right(starts,a)-1
  if i<0 or b>cs[i]['window_end_ns'] or row['Model ID']!=cs[i]['model_id'] or 'metadata' in row['Type'].lower():continue
  c=cs[i];bucket='INTERIOR_RUNS_1_2_3' if c['run_in_model'] in selected_runs else 'CAPTURE_EDGES_OR_OTHER_RUNS';key=(bucket,c['layer'],row['Type']);v=aggs[key];v['calls']+=1;v['duration_us']+=dec(row['Duration(us)']);v['max_us']=max(v['max_us'],dec(row['Duration(us)']));cycles=dec(row['aic_total_cycles']);v['aic_cycles']+=cycles;v['mac_proxy']+=cycles*dec(row['aic_mac_ratio']);v['mte2_proxy']+=cycles*dec(row['aic_mte2_ratio'])
rows=[]
for (bucket,layer,typ),v in sorted(aggs.items()):rows.append({'sample_scope':bucket,'layer':layer,'operator_type':typ,'calls':v['calls'],'task_duration_sum_us':str(v['duration_us']),'max_task_duration_us':str(v['max_us']),'reported_cycle_weighted_mac_ratio':str(v['mac_proxy']/v['aic_cycles']) if v['aic_cycles'] else None,'reported_cycle_weighted_mte2_ratio':str(v['mte2_proxy']/v['aic_cycles']) if v['aic_cycles'] else None,'aic_reported_cycles':str(v['aic_cycles'])})
with (O/'operator_metrics.csv').open('w') as f:
 writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
allranking=rank_layers(calls);inside=rank_layers(interior)
result={'status':'INTERIOR_ROBUSTNESS_ANALYZED','metric':'CSV-reported AIC-cycle-weighted MAC ratio; not whole-device utilization or MFU','interior_rule':'Use runs1,2,3 for every rank, chosen before inspecting this comparison. All original data retained. Run indices are per rank/model, not asserted cross-rank same iteration.','all_capture_lowest':allranking[:5],'interior_lowest':inside[:5],'interior_all_40_ranking':inside,'per_run_lowest':by_run,'all_capture_layer0':[x for x in allranking if x['layer']==0][0],'interior_layer0':[x for x in inside if x['layer']==0][0],'interior_call_count':len(interior),'interior_rank_count':len(byrank),'lowest_identity_stable':allranking[0]['layer']==inside[0]['layer'],'baseline_bottleneck_causality_proven':False,'operator_table':'operator_metrics.csv','layer0_top_task_types':sorted([x for x in rows if x['layer']==0],key=lambda x:Decimal(x['task_duration_sum_us']),reverse=True)[:10],'limitations':['This early-decode sample does not represent the full1024-token steady state','Counter ratios are rounded vendor output and vary with participating cores','Profiler and asynchronous DP boundary effects have not been removed','Captured target scope excludes eager prefill/DSpark and external Engram calls']}
(O/'interior-robustness.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['interior_all_40_ranking','per_run_lowest']},indent=2))
