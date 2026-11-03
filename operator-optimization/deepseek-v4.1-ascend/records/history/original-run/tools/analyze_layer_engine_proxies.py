"""Aggregate multiple AIC/AIV pipe proxies in existing captured-layer windows."""
from pathlib import Path
from collections import defaultdict
from decimal import Decimal,InvalidOperation
import bisect,csv,json
T=Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41')
ROOT=Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-20261028/diagnostic/runtime/diagnostic-traces')
O=T/'captured-layer-engine-proxies';O.mkdir(exist_ok=False)
mapping=json.loads((T/'captured-layer-attribution/module-call-metrics.json').read_text())
fields={'aic_mac_ratio':'aic_total_cycles','aic_mte2_ratio':'aic_total_cycles','aic_scalar_ratio':'aic_total_cycles','aiv_vec_ratio':'aiv_total_cycles','aiv_mte2_ratio':'aiv_total_cycles','aiv_mte3_ratio':'aiv_total_cycles','aiv_scalar_ratio':'aiv_total_cycles'}
def ns(x):return int(Decimal(x.strip())*1000)
def dec(x):
 try:
  a=Decimal(x);return a if a.is_finite() else None
 except (InvalidOperation,TypeError):return None
def bucket():return {'calls':0,'window_ns':0,'known_task_union_ns':0,'rows':0,'metrics':{k:{'num':Decimal(0),'den':Decimal(0),'missing_with_positive_cycles':0,'no_positive_cycles':0} for k in fields},'mc2_us':Decimal(0)}
b=defaultdict(bucket);rankcounts={}
for rp in sorted(ROOT.glob('*_ascend_pt')):
 mods=[x for x in mapping if x['rank']==rp.name and x['run_in_model'] in [1,2,3]];assert len(mods)==120
 mods.sort(key=lambda x:x['window_start_ns']);starts=[m['window_start_ns'] for m in mods];assigned=defaultdict(list)
 for row in csv.DictReader((rp/'ASCEND_PROFILER_OUTPUT/kernel_details.csv').open()):
  if row['Model ID']!='49' or 'metadata' in row['Type'].lower():continue
  st=ns(row['Start Time(us)']);en=st+ns(row['Duration(us)']);i=bisect.bisect_right(starts,st)-1
  if i>=0 and en<=mods[i]['window_end_ns']:assigned[i].append(row)
 rankcounts[rp.name]=len(mods)
 for i,m in enumerate(mods):
  rows=assigned[i];assert len(rows)==m['attributed_rows'],(rp.name,m['layer'],len(rows),m['attributed_rows'])
  q=b[m['layer']];q['calls']+=1;q['rows']+=len(rows);q['window_ns']+=m['window_ns'];q['known_task_union_ns']+=m['known_task_union_ns']
  for row in rows:
   if row['Type']=='DispatchFFNCombine':q['mc2_us']+=Decimal(row['Duration(us)'])
   for key,cyc in fields.items():
    c=dec(row[cyc]);v=dec(row[key]);z=q['metrics'][key]
    if c is None or c<=0:z['no_positive_cycles']+=1;continue
    if v is None or not Decimal(0)<=v<=Decimal(1):z['missing_with_positive_cycles']+=1;continue
    z['den']+=c;z['num']+=c*v
 assert len(rankcounts)<=16
out=[]
for layer,q in sorted(b.items()):
 assert q['calls']==48
 ms={k:{'numerator':str(v['num']),'denominator':str(v['den']),'cycle_weighted_ratio':float(v['num']/v['den']) if v['den'] else None,'missing_with_positive_cycles':v['missing_with_positive_cycles'],'no_positive_cycles':v['no_positive_cycles']} for k,v in q['metrics'].items()}
 out.append({'layer':layer,'calls':q['calls'],'rows':q['rows'],'mean_layer_window_us':q['window_ns']/q['calls']/1000,'mean_mc2_duration_us':float(q['mc2_us']/q['calls']),'known_task_time_union_over_window':q['known_task_union_ns']/q['window_ns'],'metrics':ms})
assert len(out)==40
s={'status':'CAPTURED_TARGET_INTERIOR_MULTI_PIPE_COUNTERS','rank_count':len(rankcounts),'per_rank_calls':rankcounts,'layers':out,'limitations':['Each ratio is weighted by reported engine cycles across attributed rows; missing ratios with positive cycles are excluded and counted.','AIC and AIV denominators differ and pipeline activities may overlap; ratios cannot be added, subtracted, or equated to full-chip utilization, hardware peak efficiency or wait time.','Known task temporal union is coverage of exported task intervals inside the source-inferred window, not proof that device is occupied or idle.','Duration sums and cross-rank means are not end-to-end critical path.','No causal attribution follows from low values alone.'],'no_device_work':True}
assert len(json.dumps(s).encode())<100000
(O/'summary.json').write_text(json.dumps(s,indent=2)+'\n')
print(json.dumps({'status':s['status'],'ranks':s['rank_count'],'layer0':out[0]},indent=2))
