"""Observe EP peer timing associations; does not turn uncalibrated skew into a bound."""
from pathlib import Path
from decimal import Decimal
from collections import defaultdict
import csv,json,math,statistics,bisect
T=Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41');ROOT=Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-20261028/diagnostic/runtime/diagnostic-traces');O=T/'mc2-arrival-analysis-v2';O.mkdir(exist_ok=False)
byrank={}
for rp in sorted(ROOT.glob('*_ascend_pt')):
 rows=[]
 for row in csv.DictReader((rp/'ASCEND_PROFILER_OUTPUT/kernel_details.csv').open()):
  if row['Type']!='DispatchFFNCombine':continue
  s=row['Input Shapes'].strip('"').split(';');w=s[1].split(',');route=s[3].split(',')
  if int(w[0])!=24 or int(route[-1])!=6:continue
  start=int(Decimal(row['Start Time(us)'].strip())*1000);duration=int(Decimal(row['Duration(us)'])*1000)
  rows.append({'rank':rp.name,'start_ns':start,'end_ns':start+duration,'duration_us':float(Decimal(row['Duration(us)'])),'physical_input_rows':int(s[0].split(',')[0]),'model_id':row['Model ID'],'stream_id':row['Stream ID'],'task_id':row['Task ID'],'mac_ratio':row['aic_mac_ratio'],'mte2_ratio':row['aic_mte2_ratio']})
 rows.sort(key=lambda x:x['start_ns']);byrank[rp.name]=rows
counts={k:len(v) for k,v in byrank.items()};assert len(byrank)==16 and all(n%40==0 for n in counts.values()),counts
p=T/'captured-layer-attribution/module-call-metrics.json';assert p.stat().st_size<20*1024*1024;mapping=json.loads(p.read_text());validated=0
for rank,rows in byrank.items():
 ms=sorted([c for c in mapping if c['rank']==rank],key=lambda c:c['window_start_ns']);starts=[c['window_start_ns'] for c in ms]
 for i,row in enumerate(rows):
  row['target_call_ordinal']=i;row['source_layer']=i%40
  if row['model_id']!='4294967295':
   ix=bisect.bisect_right(starts,row['start_ns'])-1;assert ix>=0 and row['end_ns']<=ms[ix]['window_end_ns'] and ms[ix]['layer']==i%40,('Layer ordinal inconsistent with validated graph',rank,i)
   validated+=1
with (O/'rank_target_calls.csv').open('w') as f:
 raw=[x for rows in byrank.values() for x in rows];writer=csv.DictWriter(f,fieldnames=list(raw[0]));writer.writeheader();writer.writerows(raw)
def corr(xs,ys):
 a=statistics.mean(xs);b=statistics.mean(ys);n=sum((x-a)*(y-b) for x,y in zip(xs,ys));den=math.sqrt(sum((x-a)**2 for x in xs)*sum((y-b)**2 for y in ys));return n/den if den else None
dp0={k:v for k,v in byrank.items() if k.startswith('dp0_')};dp1={k:v for k,v in byrank.items() if k.startswith('dp1_')};assert len(dp0)==len(dp1)==8
scores=[];detail=[]
for shift in [-1,0,1]:
 records=[]
 for i in range(min(len(v) for v in dp0.values())):
  j=i+40*shift
  if j<0 or any(j>=len(v) for v in dp1.values()):continue
  rs=[v[i] for v in dp0.values()]+[v[j] for v in dp1.values()];lo=min(x['start_ns'] for x in rs);arr=[(x['start_ns']-lo)/1000 for x in rs];dur=[x['duration_us'] for x in rs];captured=all(x['model_id']!='4294967295' for x in rs)
  rec={'candidate_dp1_forward_shift':shift,'dp0_target_call_ordinal':i,'dp1_target_call_ordinal':j,'layer':i%40,'all_captured_rows3':captured and all(x['physical_input_rows']==3 for x in rs),'start_spread_us':(max(x['start_ns'] for x in rs)-lo)/1000,'end_spread_us':(max(x['end_ns'] for x in rs)-min(x['end_ns'] for x in rs))/1000,'common_interval_overlap':max(x['start_ns'] for x in rs)<=min(x['end_ns'] for x in rs),'start_duration_correlation':corr(arr,dur),'mean_duration_us':statistics.mean(dur),'cross_rank_alignment_proven':False};records.append(rec)
 def agg(cs):
  return {'calls':len(cs),'common_interval_overlap_fraction':sum(x['common_interval_overlap'] for x in cs)/len(cs) if cs else None,'median_start_spread_us':statistics.median(x['start_spread_us'] for x in cs) if cs else None,'median_end_spread_us':statistics.median(x['end_spread_us'] for x in cs) if cs else None,'median_start_duration_correlation':statistics.median(x['start_duration_correlation'] for x in cs if x['start_duration_correlation'] is not None) if cs else None,'median_mean_duration_us':statistics.median(x['mean_duration_us'] for x in cs) if cs else None}
 scores.append({'candidate_dp1_forward_shift':shift,'all':agg(records),'captured_layer0':agg([x for x in records if x['layer']==0 and x['all_captured_rows3']]),'captured_other_layers':agg([x for x in records if x['layer']!=0 and x['all_captured_rows3']])});detail.extend(records)
with (O/'candidate_alignment_calls.csv').open('w') as f:
 writer=csv.DictWriter(f,fieldnames=list(detail[0]));writer.writeheader();writer.writerows(detail)
summary={'state':'CANDIDATE_ALIGNMENT_SENSITIVITY_ONLY','counts':counts,'captured_layer_ordinals_independently_validated':validated,'unequal_counts_preserved':True,'candidate_shift_scores':scores,'alignment_automatically_promoted_to_proof':False,'clock_error_bound_us':None,'reason_for_v2':'First analysis refused equal-count assumption; DP0 contains400target calls andDP1contains440. Both original failure and this sensitivity analysis retained.','scope':'Ordinal layer modulo40 corroborated against every already validated captured graph; relative forward shift remains hypothetical without explicitMC2 sequence IDs/clock error evidence.','no_device_work':True};(O/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
