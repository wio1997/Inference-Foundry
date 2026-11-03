"""Describe per-rank device timeline between successive target forwards.

All intervals are within a single rank's profiler clock. Cross-DP forward
pairing is a previously recorded hypothesis; task time outside the selected
interval and unrelated streams are not silently assigned to the model.
"""
from pathlib import Path
from decimal import Decimal
from collections import defaultdict
import bisect, csv, json, statistics

T=Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41')
ROOT=Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-20261028/diagnostic/runtime/diagnostic-traces')
O=T/'inter-forward-gap-analysis-v2'; O.mkdir(exist_ok=False)
timeline=defaultdict(list)
for x in csv.DictReader((T/'mc2-preceding-analysis/rank_layer0_timeline.csv').open()):
    x['target_forward_ordinal']=int(x['target_forward_ordinal']);x['layer_start_ns']=int(x['layer_start_ns'])
    x['interior_runs_1_2_3']=x['interior_runs_1_2_3']=='True';timeline[x['rank']].append(x)
mc2=defaultdict(list)
for x in csv.DictReader((T/'mc2-arrival-analysis-v2/rank_target_calls.csv').open()):
    x['target_call_ordinal']=int(x['target_call_ordinal']);x['start_ns']=int(x['start_ns']);x['end_ns']=int(x['end_ns']);mc2[x['rank']].append(x)
for r in mc2:mc2[r].sort(key=lambda x:x['target_call_ordinal'])
def union(intervals):
    end=None;total=0
    for a,b in sorted(intervals):
        if end is None or a>end:total+=b-a;end=b
        elif b>end:total+=b-end;end=b
    return total
allrows=[];family_rows=[];unmatched=[]
for rank,entries in sorted(timeline.items()):
    kernels=[]
    with (ROOT/rank/'ASCEND_PROFILER_OUTPUT/kernel_details.csv').open() as f:
        for row in csv.DictReader(f):
            start=int(Decimal(row['Start Time(us)'].strip())*1000);end=start+int(Decimal(row['Duration(us)'])*1000)
            kernels.append((start,end,row))
    kernels.sort(key=lambda x:x[0]);starts=[k[0] for k in kernels]
    maxends=[];current=-1
    for _,end,_ in kernels:
        current=max(current,end);maxends.append(current)
    for e in sorted(entries,key=lambda x:x['target_forward_ordinal']):
        o=e['target_forward_ordinal']
        if o==0:continue
        previous=mc2[rank][o*40-1];assert previous['source_layer']=='39'
        begin=previous['end_ns'];end=e['layer_start_ns']
        if end<begin:
            unmatched.append({'rank':rank,'forward':o,'reason':'previous MC2 ends after next layer start'});continue
        # Prefix maximum ends locate every task crossing the window, with no
        # assumed upper bound on duration; clip actual intervals to the gap.
        left=bisect.bisect_right(maxends,begin);right=bisect.bisect_left(starts,end)
        clipped=[];bytype=defaultdict(list);model_ids=defaultdict(int)
        for a,b,row in kernels[left:right]:
            ca=max(a,begin);cb=min(b,end)
            if ca>=cb:continue
            clipped.append((ca,cb));bytype[row['Type']].append((ca,cb));model_ids[row['Model ID']]+=1
        dur=end-begin;busy=union(clipped)
        rec={'rank':rank,'dp':e['dp'],'target_forward_ordinal':o,'interior_runs_1_2_3':e['interior_runs_1_2_3'],
             'previous_target_layer39_mc2_end_ns':begin,'next_target_layer0_start_ns':end,
             'local_gap_us':dur/1000,'known_device_task_union_us':busy/1000,'no_task_in_exported_intervals_us':(dur-busy)/1000,
             'kernel_rows_overlapping_gap':sum(len(v) for v in bytype.values()),'model_ids':json.dumps(dict(model_ids),sort_keys=True),
             'interpretation':'Uncovered time is not automatically device idle, host delay, or causal impact; profiler scope/other activity may be incomplete.'}
        allrows.append(rec)
        for typ,intervals in sorted(bytype.items()):
            family_rows.append({'rank':rank,'dp':e['dp'],'target_forward_ordinal':o,'type':typ,'calls':len(intervals),
                                'clipped_union_us':union(intervals)/1000,
                                'sum_clipped_duration_us':sum(b-a for a,b in intervals)/1000})
with (O/'rank_gaps.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(allrows[0]));w.writeheader();w.writerows(allrows)
with (O/'task_families.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(family_rows[0]));w.writeheader();w.writerows(family_rows)
groups=defaultdict(list)
for r in allrows:groups[r['target_forward_ordinal']].append(r)
paired=[]
for i,rs in sorted(groups.items()):
    if len(rs)!=16:continue
    out={'target_forward_ordinal':i,'all_interior':all(r['interior_runs_1_2_3'] for r in rs)}
    for dp in ['dp0','dp1']:
        rr=[r for r in rs if r['dp']==dp];assert len(rr)==8
        for v in ['local_gap_us','known_device_task_union_us','no_task_in_exported_intervals_us']:
            out[dp+'_'+v+'_mean']=statistics.mean(r[v] for r in rr)
            out[dp+'_'+v+'_min']=min(r[v] for r in rr);out[dp+'_'+v+'_max']=max(r[v] for r in rr)
    paired.append(out)
summary={'status':'RANK_LOCAL_GAPS_OBSERVED','paired_target_forward_indices':len(paired),'paired':paired,
         'rank_gap_records':len(allrows),'family_aggregate_records':len(family_rows),'unmatched':unmatched,
         'assumptions':['Target ordinal pairing is the offset-zero hypothesis, not operation-ID proof.',
                        'Kernel intervals are clipped; union is not an exclusive attribution to current request.',
                        'Gap begins after prior layer39 MC2, not after the prior model forward or its final head.',
                        'Missing exported tasks and concurrent work prevent equating the unoccupied export interval to device idle.'],
         'prior_version_limit':'v1 looked back50ms; this version checks all crossing intervals without a duration assumption.',
         'no_device_work':True}
assert len(json.dumps(summary).encode())<32768
(O/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
