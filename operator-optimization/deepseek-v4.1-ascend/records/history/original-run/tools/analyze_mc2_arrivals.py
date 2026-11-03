"""Observe EP peer timing associations; does not turn uncalibrated skew into a bound."""
from pathlib import Path
from decimal import Decimal
from collections import defaultdict
import csv,json,math,statistics
T=Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41');ROOT=Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-20261028/diagnostic/runtime/diagnostic-traces');O=T/'mc2-arrival-analysis';O.mkdir(exist_ok=False)
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
counts={k:len(v) for k,v in byrank.items()};assert len(byrank)==16 and len(set(counts.values()))==1 and all(n%40==0 for n in counts.values()),counts
def correlation(xs,ys):
 a=statistics.mean(xs);b=statistics.mean(ys);num=sum((x-a)*(y-b) for x,y in zip(xs,ys));den=math.sqrt(sum((x-a)**2 for x in xs)*sum((y-b)**2 for y in ys));return num/den if den else None
aligned=[]
for i in range(next(iter(counts.values()))):
 rows=[v[i] for v in byrank.values()];start_min=min(x['start_ns'] for x in rows);arr=[(x['start_ns']-start_min)/1000 for x in rows];duration=[x['duration_us'] for x in rows];inputrows=[x['physical_input_rows'] for x in rows]
 aligned.append({'target_call_ordinal':i,'assumed_layer_from_ordinal':i%40,'assumed_target_forward_ordinal':i//40,'rank_count':16,'observed_start_spread_us':(max(x['start_ns'] for x in rows)-min(x['start_ns'] for x in rows))/1000,'observed_end_spread_us':(max(x['end_ns'] for x in rows)-min(x['end_ns'] for x in rows))/1000,'mean_duration_us':statistics.mean(duration),'min_duration_us':min(duration),'max_duration_us':max(duration),'start_duration_correlation':correlation(arr,duration),'min_physical_input_rows':min(inputrows),'max_physical_input_rows':max(inputrows),'all_physical_rows_3':all(n==3 for n in inputrows),'all_captured_target':all(x['model_id']!='4294967295' for x in rows),'alignment_status':'ORDER_BASED_HYPOTHESIS_EP16_NOT_EXPLICIT_HCCL_OPID','cross_rank_clock_error_bound_us':None})
with (O/'aligned_target_calls.csv').open('w') as f:
 writer=csv.DictWriter(f,fieldnames=list(aligned[0]));writer.writeheader();writer.writerows(aligned)
raw=[]
for rank,rows in byrank.items():
 for i,row in enumerate(rows):raw.append({**row,'target_call_ordinal':i})
with (O/'rank_target_calls.csv').open('w') as f:
 writer=csv.DictWriter(f,fieldnames=list(raw[0]));writer.writeheader();writer.writerows(raw)
def summarize(rows):
 return {'calls':len(rows),'median_start_spread_us':statistics.median(x['observed_start_spread_us'] for x in rows) if rows else None,'median_end_spread_us':statistics.median(x['observed_end_spread_us'] for x in rows) if rows else None,'median_mean_duration_us':statistics.median(x['mean_duration_us'] for x in rows) if rows else None,'median_start_duration_correlation':statistics.median(x['start_duration_correlation'] for x in rows if x['start_duration_correlation'] is not None) if rows else None}
uniform=[x for x in aligned if x['all_physical_rows_3'] and x['all_captured_target']];l0=[x for x in uniform if x['assumed_layer_from_ordinal']==0];other=[x for x in uniform if x['assumed_layer_from_ordinal']!=0]
summary={'state':'OBSERVED_ASSOCIATION_NOT_CAUSAL_PROOF','target_call_counts':counts,'all_captured_uniform_rows3_layer0':summarize(l0),'all_captured_uniform_rows3_other_layers':summarize(other),'layer0_call_examples':l0,'alignment_basis':['Standard target identifiable by24local experts andtopk6, distinct from draft8experts/topk3','Equal target call counts and multiples40 per rank','Ordered call matching relies on same EP collective ordering; explicit internalMC2 opId not captured'],'clock_alignment':'CANN-exported epoch timestamps from same host; bounded cross-device clock error not independently established','cannot_claim':['Start spread is proven unavoidable wait','All same ordinal calls are matched by explicit HCCL sequence ID','Communication or scheduling is proven dominant solely from correlation'],'source_mechanism':'MC2 source contains cross-rank readiness polling and cross-core flags; relates observations to competing hypotheses','no_device_work':True}
(O/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
