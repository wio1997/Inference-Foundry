#!/usr/bin/env python3
"""Correlate one Run165 prefill device gap with trace HostToDevice flows."""
import csv,json,glob,collections
from decimal import Decimal
from pathlib import Path
root=Path(__file__).resolve().parents[1]
base=root/'evidence/20260925_loop045_boundary/run165'
p=list((base/'profiler/rank0').glob('*/ASCEND_PROFILER_OUTPUT'))
assert len(p)==1;p=p[0]
trace=json.loads((p/'trace_view.json').read_text())
flows=collections.defaultdict(lambda:{'s':[],'f':[]})
for event in trace:
 if event.get('cat')=='HostToDevice':
  flows[str(event['id'])][event['ph']].append(Decimal(str(event['ts'])))
submit_at_device_start=collections.defaultdict(list)
for flow in flows.values():
 if len(flow['s'])==1 and flow['f']:
  submit_at_device_start[min(flow['f'])].append(flow['s'][0])
raw=list(csv.DictReader((p/'kernel_details.csv').open()))
assert len(raw)==2825
intervals=[]
for row in raw:
 start=Decimal(row['Start Time(us)'].strip());end=start+Decimal(row['Duration(us)'])
 assert start in submit_at_device_start
 intervals.append((start,end))
intervals.sort()
merged=[]
for start,end in intervals:
 if not merged or start>merged[-1][1]:merged.append([start,end])
 elif end>merged[-1][1]:merged[-1][1]=end
free=host_after_prior_end=post_flow_wait=Decimal(0);gaps=[]
for (_,previous_end),(next_start,_) in zip(merged,merged[1:]):
 gap=next_start-previous_end
 # Earliest launch flow for the next task is a conservative lower
 # estimate of the portion of this gap before Host dispatch.
 flow_start=min(submit_at_device_start[next_start])
 host=max(Decimal(0),min(gap,flow_start-previous_end))
 free+=gap;host_after_prior_end+=host;post_flow_wait+=gap-host
 gaps.append({'gap_us':float(gap),'host_flow_after_previous_end_us':float(host),
  'flow_to_device_us':float(next_start-flow_start),
  'relative_next_start_ms':float((next_start-intervals[0][0])/1000)})
step=list(csv.DictReader((p/'step_trace_time.csv').open()))
assert len(step)==1
assert abs(float(free)-float(step[0]['Free']))<100
assert abs(float((merged[-1][1]-merged[0][0]))-float(step[0]['Stage']))<100
rank0=json.loads((base/'boundary/rank0_cohort5.json').read_text())
bench=json.loads((base/'measured12.json').read_text())
start=min(x['start'] for x in bench['requests'])
measured_calls=[x for x in rank0['calls'] if x['t_ns']/1e9>=start]
assert measured_calls[0]['scheduled_tokens']==83
report={'run':'run166','source':'Run165 one rank0 profiled first measured _model_forward, 83 scheduled tokens',
 'raw_kernel_rows':len(raw),'unique_device_intervals':len(merged),'device_span_us':float(merged[-1][1]-merged[0][0]),
 'free_us':float(free),'host_flow_starts_after_previous_device_task_us':float(host_after_prior_end),
 'remaining_after_host_flow_us':float(post_flow_wait),
 'fraction_of_free_before_host_flow':float(host_after_prior_end/free),
 'host_to_device_flows':len(flows),'matched_kernel_rows':len(raw),
 'largest_gaps':sorted(gaps,key=lambda x:x['gap_us'],reverse=True)[:20],
 'cann_step_trace_us':{k:float(step[0][k]) for k in ('Computing','Communication(Not Overlapped)','Free','Stage')},
 'limits':'HostToDevice flow start is trace-reported launch/dispatch correlation, not proof of exact CPU instruction or guaranteed root cause. Single rank/call with profiler initialization and explicit sync; unprofiled matched 83-token call in Run161 ~0.320s forward versus profiled device stage0.444s. Do not infer 357.6ms removable or formal E2E gain. Flow matching is exact by device timestamp; event duplicates do not change interval union.'}
(base/'prefill_host_gap_analysis.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='largest_gaps'},indent=2))
