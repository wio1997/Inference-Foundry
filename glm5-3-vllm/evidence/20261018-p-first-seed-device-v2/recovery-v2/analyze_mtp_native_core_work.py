"""Reduce only the device tasks directly linked to native MTP opaque-core APIs."""
from pathlib import Path
from bisect import bisect_left,bisect_right
from collections import Counter,defaultdict
import hashlib,json,statistics
r=Path(__file__).resolve().parent
p=Path('/Users/wio/work/glm53-execution-evidence-backup/20261018-p-first-seed-device-v2/recovery-v2/mtp-moe-live1-profile/rank0-trace_view.json')
x=json.loads(p.read_text());events=x['traceEvents'] if isinstance(x,dict) else x;e=[z for z in events if z.get('ph')=='X'];ops=[z for z in e if z.get('name')=='vllm::moe_forward_shared'];main=sorted([z for z in e if z.get('pid')==ops[0]['pid'] and z.get('tid')==ops[0]['tid']],key=lambda z:float(z['ts']));times=[float(z['ts']) for z in main]
dq={(z['name'],z.get('args',{}).get('correlation_id')):z for z in e if z.get('cat')=='dequeue'};api=defaultdict(list);tasks=defaultdict(list)
for z in e:
 if z.get('name','').startswith(('AscendCL@','Node@')):api[z['tid']].append(z)
 if 'Task Type' in z.get('args',{}) and 'connection_id' in z['args']:tasks[z['args']['connection_id']].append(z)
at={}
for tid in api:api[tid].sort(key=lambda z:float(z['ts']));at[tid]=[float(z['ts']) for z in api[tid]]
rows=[];kernel_us=defaultdict(list)
for op in ops:
 t=float(op['ts']);kids=main[bisect_left(times,t):bisect_right(times,t+float(op['dur']))]
 if not any('GroupedMatmul' in z.get('name','') for z in kids):continue
 matched=[];seen=set();enqueues=[z for z in kids if z.get('cat')=='enqueue'];missed=0
 for z in enqueues:
  q=dq.get((z['name'].replace('Enqueue@','Dequeue@'),z.get('args',{}).get('correlation_id')))
  if not q:missed+=1;continue
  t0=float(q['ts']);tid=q['tid'];calls=api[tid][bisect_left(at[tid],t0):bisect_right(at[tid],t0+float(q['dur']))]
  for call in calls:
   for task in tasks.get(call.get('args',{}).get('connection_id'),[]):
    key=(task.get('pid'),task.get('tid'),str(task['ts']),task.get('name'))
    if key not in seen:seen.add(key);matched.append(task)
 compute=[z for z in matched if z['args']['Task Type'] in ['AI_CORE','AI_VECTOR_CORE','MIX_AIC','MIX_AIV']]
 for z in compute:kernel_us[z['name']].append(float(z['dur']))
 rows.append({'opaque_ts':op['ts'],'opaque_cpu_us':op['dur'],'matched_task_types':dict(Counter(z['args']['Task Type'] for z in matched)),'matched_compute_duration_sum_us':sum(float(z['dur']) for z in compute),'matched_compute_tasks':len(compute),'enqueue_records':len(enqueues),'missing_dequeue':missed})
assert len(rows)==100
v=[z['matched_compute_duration_sum_us'] for z in rows]
result={'completed':True,'source_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'native_core_calls':len(rows),'per_call_linked_compute_task_duration_sum_us':{'median':statistics.median(v),'min':min(v),'max':max(v)},'matched_compute_kernels':{k:{'n':len(v),'median_us':statistics.median(v),'sum_us':sum(v)} for k,v in kernel_us.items()},'per_call_rows':rows,'scope':'Rank0 stopped profiler diagnostic only. Task duration sums may overlap and omit unlinked communication/graph tasks. Not elapsed core service time, no claim of a complete critical path or target/draft partition; no multiplication into end-to-end savings. Actual unprofiled formal result remains the optimization verdict.'}
(r/'mtp-native-core-device-work-v1.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:result[k] for k in ['native_core_calls','per_call_linked_compute_task_duration_sum_us','matched_compute_kernels']}))
