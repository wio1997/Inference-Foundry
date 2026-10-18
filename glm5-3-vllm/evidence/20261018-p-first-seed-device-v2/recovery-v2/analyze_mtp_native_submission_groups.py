"""Reduce only the device tasks directly linked to native MTP opaque-core APIs."""
from pathlib import Path
from bisect import bisect_left,bisect_right
from collections import Counter,defaultdict
import hashlib,json,statistics
r=Path(__file__).resolve().parent
p=Path('/Users/wio/work/glm53-execution-evidence-backup/20261018-p-first-seed-device-v2/recovery-v2/mtp-moe-live1-profile/rank0-trace_view.json')
x=json.loads(p.read_text());events=x['traceEvents'] if isinstance(x,dict) else x;e=[z for z in events if z.get('ph')=='X'];ops=[z for z in e if z.get('name') in ['vllm::moe_forward_shared','vllm::mla_forward','vllm::unquantized_gemm']];main=sorted([z for z in e if z.get('pid')==ops[0]['pid'] and z.get('tid')==ops[0]['tid']],key=lambda z:float(z['ts']));times=[float(z['ts']) for z in main]
dq={(z['name'],z.get('args',{}).get('correlation_id')):z for z in e if z.get('cat')=='dequeue'};api=defaultdict(list);tasks=defaultdict(list)
for z in e:
 if z.get('name','').startswith(('AscendCL@','Node@')):api[z['tid']].append(z)
 if 'Task Type' in z.get('args',{}) and 'connection_id' in z['args']:tasks[z['args']['connection_id']].append(z)
at={}
for tid in api:api[tid].sort(key=lambda z:float(z['ts']));at[tid]=[float(z['ts']) for z in api[tid]]
rows=[];kernel_us=defaultdict(list)
for op in ops:
 t=float(op['ts']);kids=main[bisect_left(times,t):bisect_right(times,t+float(op['dur']))]
 if op['name']=='vllm::moe_forward_shared' and not any('GroupedMatmul' in z.get('name','') for z in kids):continue
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
 rows.append({'kind':op['name'],'opaque_ts':op['ts'],'opaque_cpu_us':op['dur'],'matched_task_types':dict(Counter(z['args']['Task Type'] for z in matched)),'matched_compute_duration_sum_us':sum(float(z['dur']) for z in compute),'matched_compute_tasks':len(compute),'enqueue_records':len(enqueues),'missing_dequeue':missed})
groups={}
for name in sorted(set(z['kind'] for z in rows)):
 chosen=[z for z in rows if z['kind']==name];values=[z['matched_compute_duration_sum_us'] for z in chosen]
 groups[name]={'calls':len(chosen),'linked_compute_duration_sum_us':{'median':statistics.median(values),'min':min(values),'max':max(values)},'opaque_cpu_us_median':statistics.median(z['opaque_cpu_us'] for z in chosen),'linked_compute_tasks_per_call_median':statistics.median(z['matched_compute_tasks'] for z in chosen)}
model=defaultdict(list)
for task in e:
 if task.get('args',{}).get('Task Type') in ['AI_CORE','AI_VECTOR_CORE','MIX_AIC','MIX_AIV']:model[str(task['args'].get('Model Id'))].append(task)
models={k:{'compute_tasks':len(v),'compute_duration_sum_us':sum(float(z['dur']) for z in v),'GroupedMatmulSwiglu_tasks':sum('GroupedMatmulSwiglu' in z['name'] for z in v)} for k,v in model.items()}
result={'completed':True,'source_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'opaque_groups':groups,'all_physical_compute_grouped_by_Model_Id':models,'scope':'CPU-only existing rank0 profiler reduction. Opaque MLA events are native MTP attention; unquantized_gemm is labeled by its actual operator name, not assumed to be LMHead until its source/shape is verified. Per-call duration sums may overlap, omit unlinked communication and do not equal critical-path elapsed time. ModelId groups alone do not isolate an entire target/draft service schedule. Do not add sums or multiply into an end-to-end performance forecast.','unknown':['Host supply idle gaps attributable to each component','Unprofiled target verification elapsed time and acceptance per mode','Exact logical layer identity and shapes of unquantized_gemm']}
(r/'mtp-native-submission-groups-v1.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
