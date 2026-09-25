#!/usr/bin/env python3
"""One-card same-stream event Host enqueue cost; not a service speedup."""
import json,statistics,time
from pathlib import Path
import torch
import torch_npu
N=1000;REPS=12
torch.npu.set_device(0)
s=torch.npu.current_stream()
def measure(kind):
 vals=[]
 for _ in range(REPS):
  torch.npu.synchronize()
  a=time.thread_time_ns();w=time.perf_counter_ns()
  if kind=='empty':
   for i in range(N):pass
  elif kind=='record':
   for i in range(N):s.record_event()
  elif kind=='record_wait':
   for i in range(N):s.wait_event(s.record_event())
  c=time.thread_time_ns();d=time.perf_counter_ns()
  torch.npu.synchronize()
  vals.append({'thread_cpu_us_per_pair':(c-a)/N/1000,'wall_us_per_pair':(d-w)/N/1000})
 return vals
# Warm up implementation before timed rows.
for _ in range(10):s.wait_event(s.record_event())
torch.npu.synchronize()
raw={k:measure(k) for k in ('empty','record','record_wait')}
summary={k:{'median_thread_cpu_us_per_pair':statistics.median(x['thread_cpu_us_per_pair'] for x in v),'median_wall_us_per_pair':statistics.median(x['wall_us_per_pair'] for x in v)} for k,v in raw.items()}
result={'run':'run201','environment':'single Ascend 910B3, container torch_npu, service stopped','iterations_per_rep':N,'reps':REPS,'summary':summary,'raw':raw,'derived':{'same_stream_record_wait_minus_empty_thread_cpu_us':summary['record_wait']['median_thread_cpu_us_per_pair']-summary['empty']['median_thread_cpu_us_per_pair'],'illustrative_43_layers_2_pairs_per_forward_ms':(summary['record_wait']['median_thread_cpu_us_per_pair']-summary['empty']['median_thread_cpu_us_per_pair'])*86/1000},'limits':['Isolated eager enqueue, not Run200 full model or8-card critical path.','Profiler Run181 event count includes necessary cross-stream dependencies; this only bounds same-stream no-op opportunity.','No source change, correctness or E2E measurement.']}
Path('evidence/20260925_loop052_prefill_submission/run201/analysis.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'summary':summary,'derived':result['derived']},indent=2))
