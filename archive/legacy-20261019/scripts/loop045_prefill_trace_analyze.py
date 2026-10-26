#!/usr/bin/env python3
"""Cross-check one rank0 prefill profiler device coverage against CANN step trace."""
import csv,json,collections,glob
from pathlib import Path
root=Path(__file__).resolve().parents[1]
base=root/'evidence/20260925_loop045_boundary/run165'
paths=list((base/'profiler/rank0').glob('*/ASCEND_PROFILER_OUTPUT'))
assert len(paths)==1
p=paths[0]
kernels=list(csv.DictReader((p/'kernel_details.csv').open()))
steps=list(csv.DictReader((p/'step_trace_time.csv').open()))
assert len(steps)==1 and len(kernels)>1000
step=steps[0]
intervals=[]
by_name=collections.defaultdict(lambda:[0,0.0])
for k in kernels:
 start=float(k['Start Time(us)'].strip());dur=float(k['Duration(us)']);end=start+dur
 comm=k['Accelerator Core']=='COMMUNICATION'
 intervals.append((start,end,comm))
 name=k['Name'].split('_')[0]
 by_name[name][0]+=1;by_name[name][1]+=dur
# CANN lists each HCCL logical task alongside its identical AivKernel
# device interval. Drop the duplicate N/A row for category accounting.
comm_pairs={(a,b) for a,b,c in intervals if c}
duplicates=sum(1 for a,b,c in intervals if not c and (a,b) in comm_pairs)
intervals=[(a,b,c) for a,b,c in intervals if c or (a,b) not in comm_pairs]
lo=min(a for a,_,_ in intervals);hi=max(b for _,b,_ in intervals)
events=[]
for a,b,comm in intervals:
 events.append((a,1,1 if comm else 0));events.append((b,-1,-1 if comm else 0))
events.sort();active=communication=0;last=events[0][0];covered=collections.Counter()
for t,dc,dh in events:
 if t>last:
  label='free' if active==0 else ('overlap' if 0<communication<active else ('communication' if communication else 'compute'))
  covered[label]+=t-last
 active+=dc;communication+=dh;last=t
stage_us=float(step['Stage']);free_us=float(step['Free'])
assert abs((hi-lo)-stage_us)<150
assert abs(covered['free']-free_us)<150
assert abs(sum(covered.values())-(hi-lo))<1
rank0=json.loads((base/'boundary/rank0_cohort5.json').read_text())
bench=json.loads((base/'measured12.json').read_text())
assert bench['summary']['success']==12 and bench['summary']['fail']==0
report={'run':'run165','scope':'one rank0 warmed measured prefill _model_forward; profiler explicitly synchronizes and perturbs this call; not formal E2E',
 'kernel_count':len(kernels),'paired_hccl_aiv_duplicates_removed':duplicates,'step_trace_us':{k:float(step[k]) for k in ('Computing','Communication(Not Overlapped)','Overlapped','Communication','Free','Stage','Preparing')},
 'kernel_union_us':dict(covered),'kernel_span_us':hi-lo,'free_fraction':free_us/stage_us,
 'top_kernel_name_prefixes':[{'prefix':n,'calls':v[0],'summed_duration_us':v[1]} for n,v in sorted(by_name.items(),key=lambda x:x[1][1],reverse=True)[:20]],
 'measured_client_duration_s':bench['summary']['duration_s'],
 'rank0_handoff_ns':rank0['handoff_ns'],
 'limits':'One rank, one captured call; CANN Free is no recorded device task and can include Host enqueue latency, dependencies, synchronization/profiler effects. Kernel union across streams avoids double counting; CANN step trace confirms. Cannot infer multi-rank root cause or achievable speedup from this alone.'}
(base/'prefill_trace_analysis.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ('top_kernel_name_prefixes','rank0_handoff_ns')},indent=2))
