"""Inspect MC2 in strictly source-invariant draft brackets, CPU-only."""
from pathlib import Path
from collections import defaultdict
from decimal import Decimal
import csv,json,statistics
T=Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41')
ROOT=Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-20261028/diagnostic/runtime/diagnostic-traces')
O=T/'draft-mc2-analysis';O.mkdir(exist_ok=False)
calls=json.loads((T/'eager-draft-mainstream-attribution-v2/draft_call_metrics.json').read_text())
byrank=defaultdict(list)
for c in calls:byrank[c['rank']].append(c)
detail=[]
for rank,cs in sorted(byrank.items()):
 matches=[]
 for r in csv.DictReader((ROOT/rank/'ASCEND_PROFILER_OUTPUT/kernel_details.csv').open()):
  if r['Type']!='DispatchFFNCombine' or r['Model ID']!='4294967295':continue
  r['_s']=int(Decimal(r['Start Time(us)'].strip())*1000);r['_e']=r['_s']+int(Decimal(r['Duration(us)'])*1000);matches.append(r)
 for c in cs:
  rs=[r for r in matches if r['Stream ID']==c['main_stream_id'] and r['Task ID']==c['mc2_task_id'] and c['start_ns']<=r['_s'] and r['_e']<=c['end_ns']]
  assert len(rs)==1,(rank,c['start_ns'],len(rs));r=rs[0]
  cycles=Decimal(r['aic_total_cycles']);ratio=Decimal(r['aic_mac_ratio']);assert cycles>0 and 0<=ratio<=1
  detail.append({'rank':rank,'bracket_start_ns':c['start_ns'],'bracket_end_ns':c['end_ns'],
                 'bracket_duration_us':(c['end_ns']-c['start_ns'])/1000,'mc2_start_ns':r['_s'],'mc2_end_ns':r['_e'],
                 'mc2_duration_us':(r['_e']-r['_s'])/1000,'mc2_aic_num':str(cycles*ratio),'mc2_aic_den':str(cycles),
                 'block_aic_num':c['aic_mac_proxy_numerator'],'block_aic_den':c['aic_cycles_denominator'],
                 'scope':'Strict SparseFlashMla invariant only; model ID and stream from source-invariant bracket'})
assert len(detail)==len(calls)==162
with (O/'strict_draft_calls.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(detail[0]));w.writeheader();w.writerows(detail)
M=sum((Decimal(x['mc2_aic_num']) for x in detail),Decimal(0));D=sum((Decimal(x['mc2_aic_den']) for x in detail),Decimal(0))
B=sum((Decimal(x['block_aic_num']) for x in detail),Decimal(0));BD=sum((Decimal(x['block_aic_den']) for x in detail),Decimal(0))
mc2=[x['mc2_duration_us'] for x in detail];full=[x['bracket_duration_us'] for x in detail]
doc={'status':'STRICT_DRAFT_BRACKET_ONLY','valid_calls':len(detail),'valid_ranks':len(byrank),
     'block_mainstream_mac_counter_proxy':float(B/BD),'mc2_mac_counter_proxy':float(M/D),
     'mc2_mean_us':statistics.mean(mc2),'mc2_median_us':statistics.median(mc2),
     'block_mean_us':statistics.mean(full),'block_median_us':statistics.median(full),
     'sum_mc2_time_over_sum_bracket_time':sum(mc2)/sum(full),
     'scope_limits':['162/288 candidate calls accepted; the omitted126 match no SparseFlashMla in export, and local query count is unobserved.',
                     'Only the same NPU stream is attributed. Timing sums over calls/ranks are not end-to-end critical path.',
                     'Long fused MC2 time and low MAC ratio do not uniquely identify communication waiting, memory, load imbalance, or small matrices.',
                     'Three draft blocks pooled, layer identities unassigned, other modules and streams excluded.'],
     'no_device_work':True}
(O/'summary.json').write_text(json.dumps(doc,indent=2)+'\n');print(json.dumps(doc,indent=2))
