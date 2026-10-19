"""CPU-only ownership and host lead from the stopped int32-prefix+native-GMM2+RoPE trace."""
from pathlib import Path
from collections import Counter,defaultdict
from bisect import bisect_left,bisect_right
import argparse,csv,hashlib,json,re,statistics
a=argparse.ArgumentParser();a.add_argument("--mode",choices=["native","adaptive"],required=True);args=a.parse_args()
r=Path(__file__).resolve().parent
base=Path('/Users/wio/work/glm53-execution-evidence-backup/20261019-prefix-count-c4-supply-rank0/artifacts')
b=base/args.mode/'rank0'
p=b/'trace_view.json';c=b/'kernel_details.csv'
x=json.loads(p.read_text());events=x['traceEvents'] if isinstance(x,dict) else x;e=[z for z in events if z.get('ph')=='X']
ops=[z for z in e if z.get('name') in ['vllm::moe_forward_shared','vllm::mla_forward','vllm::unquantized_gemm']]
main=sorted([z for z in e if z.get('pid')==ops[0]['pid'] and z.get('tid')==ops[0]['tid']],key=lambda z:float(z['ts']));mt=[float(z['ts']) for z in main]
dq={(z['name'],z.get('args',{}).get('correlation_id')):z for z in e if z.get('cat')=='dequeue'}
apis=defaultdict(list);tasks=defaultdict(list)
for z in e:
 if z.get('name','').startswith(('AscendCL@','Node@')):apis[z['tid']].append(z)
 if 'Task Type' in z.get('args',{}) and 'connection_id' in z['args']:tasks[z['args']['connection_id']].append(z)
for v in apis.values():v.sort(key=lambda z:float(z['ts']))
at={k:[float(z['ts']) for z in v] for k,v in apis.items()}
compute_types=['AI_CORE','AI_VECTOR_CORE','MIX_AIC','MIX_AIV']
rows=[];chains=[]
for op in ops:
 t=float(op['ts']);end=t+float(op['dur']);kids=main[bisect_left(mt,t):bisect_right(mt,end)]
 if op['name']=='vllm::moe_forward_shared' and not any('GroupedMatmul' in z.get('name','') for z in kids):continue
 matched=[];seen=set()
 for enq in [z for z in kids if z.get('cat')=='enqueue']:
  q=dq.get((enq['name'].replace('Enqueue@','Dequeue@'),enq.get('args',{}).get('correlation_id')))
  if not q:continue
  t0=float(q['ts']);tid=q['tid'];calls=apis[tid][bisect_left(at[tid],t0):bisect_right(at[tid],t0+float(q['dur']))]
  for call in calls:
   for task in tasks.get(call.get('args',{}).get('connection_id'),[]):
    key=(task.get('pid'),task.get('tid'),str(task['ts']),task.get('name'))
    if key in seen:continue
    seen.add(key);matched.append(task)
    if task['args']['Task Type'] in compute_types:
     chains.append({'kind':op['name'],'cpu_op_ts':t,'enqueue':enq['name'],'task':task['name'],
      'enqueue_to_device_us':float(task['ts'])-float(enq['ts']),
      'runtime_to_device_us':float(task['ts'])-float(call['ts']),
      'physical_task_us':task['dur']})
 compute=[z for z in matched if z['args']['Task Type'] in compute_types]
 row={'kind':op['name'],'cpu_ts':t,'cpu_us':float(op['dur']),'linked_compute_tasks':len(compute),'linked_task_types':dict(Counter(z['args']['Task Type'] for z in matched))}
 if compute:
  first=min(float(z['ts']) for z in compute);last=max(float(z['ts'])+float(z['dur']) for z in compute)
  row.update(first_device_compute_ts=first,last_device_compute_end=last,first_device_minus_cpu_begin_us=first-t,first_device_minus_cpu_end_us=first-end,linked_compute_span_us=last-first)
 rows.append(row)
def stats(v):
 v=sorted(v);return {'n':len(v),'min':v[0],'median':statistics.median(v),'p95':v[min(len(v)-1,int(len(v)*.95))],'max':v[-1]} if v else {'n':0}
groups={}
for kind in sorted(set(z['kind'] for z in rows)):
 rr=[z for z in rows if z['kind']==kind]
 groups[kind]={k:stats([z[k] for z in rr if k in z]) for k in ['cpu_us','linked_compute_tasks','first_device_minus_cpu_begin_us','first_device_minus_cpu_end_us','linked_compute_span_us']}
 groups[kind]['per_compute_enqueue_lead_us']=stats([z['enqueue_to_device_us'] for z in chains if z['kind']==kind])
 groups[kind]['calls']=rr
syncs=sorted([z for z in e if z.get('name')=='AscendCL@aclrtSynchronizeStreamWithTimeout'],key=lambda z:float(z['ts']))
replays=sorted([z for z in e if z.get('name')=='AscendCL@aclmdlRIExecuteAsync'],key=lambda z:float(z['ts']))
launches=[]
for replay in replays:
 t=float(replay['ts']);sync=max([z for z in syncs if float(z['ts'])<=t],key=lambda z:float(z['ts']))
 linked=tasks.get(replay['args']['connection_id'],[])
 launches.append({'sync_cpu_ts':sync['ts'],'sync_cpu_us':sync['dur'],'sync_end_to_replay_api_us':t-float(sync['ts'])-float(sync['dur']),
 'replay_cpu_us':replay['dur'],'replay_cpu_ts':replay['ts'],'linked_marker_tasks':linked})
csvrows=list(csv.DictReader(c.open()))
large_slice=[z for z in csvrows if 'Slice' in z['Name'] and '1048576' in z['Input Shapes']]
cumsum=[z for z in csvrows if 'Cumsum' in z['Name'] or 'Cumsum' in z['Type'] or 'CumSum' in z['Type']]

compute=[z for z in e if z.get('args',{}).get('Task Type') in compute_types]
phys=defaultdict(list)
for z in compute:phys[z['name']].append(float(z['dur']))
report={'completed':True,'source_trace_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'source_csv_sha256':hashlib.sha256(c.read_bytes()).hexdigest(),
 'profiled_requests':4,'generated_budget':32,'rope_cached':True,'all16_profiler_stopped':True,'large_rope_slice_count':len(large_slice),'large_slice_sample':large_slice[:3],
 'cumsum_count':len(cumsum),'cumsum_task_types':dict(Counter(z['name']+'|'+z['args']['Task Type'] for z in e if 'Cumsum' in z.get('name','') and 'Task Type' in z.get('args',{}))),'cumsum_by_name_core':dict(Counter(z['Name']+'|'+z['Accelerator Core'] for z in cumsum)),'prefix_enabled_for_profile':args.mode=='adaptive','opaque_groups':groups,'target_sync_replay':launches,
 'compute_top_duration_sums':[{'name':k,'duration_us':stats(v),'sum_us':sum(v)} for k,v in sorted(phys.items(),key=lambda kv:sum(kv[1]),reverse=True)[:12]],
 'scope':'One synchronized4-request D8 admission fragment, '+args.mode+', rank0 only. Metadata observer active and warmed; not original service arrival/timing condition. All timings are profiled diagnostics, not service gain estimates. Per-task and per-call spans may overlap. CPU lead compares exact enqueue/dequeue/API connection_id to physical compute; MODEL_EXECUTE is a launch marker, not target graph completion. No claim that total wait or summed compute equals critical-path time.',
 'unknown':['Cross-rank critical-path attribution','Unprofiled exposed host gaps','Precise owner of long event waits and their downstream device dependency']}
metadata=json.loads((base/args.mode/'metadata.json').read_text());frames=sorted([z for z in metadata if z['rank']==0],key=lambda z:z['monotonic_ns']);mtp_N=[d['N'] for z in frames if z['phase']=='sample_tokens' for d in z['dispatch']];eager_target=sum(len(z['dispatch']) for z in frames if z['phase']=='execute_model');core=sorted(groups.get('vllm::moe_forward_shared',{}).get('calls',[]),key=lambda z:z['cpu_ts']);shape_linked=(len(core)==len(mtp_N) and not eager_target)
if shape_linked:
 for row,N in zip(core,mtp_N):row['observed_MTP_N']=N
report['actual_metadata']={'rank0_frames':len(frames),'MTP_dispatches':len(mtp_N),'eager_target_dispatches':eager_target,'target_graphs':sum(len(z['graphs']) for z in frames),'MTP_cpu_range_shape_order_linked':shape_linked,'MTP_N_counts':dict(Counter(mtp_N))}
report['MTP_by_observed_N']={str(N):{k:stats([z[k] for z in core if z.get('observed_MTP_N')==N and k in z]) for k in ['cpu_us','first_device_minus_cpu_end_us','linked_compute_span_us']} for N in sorted(set(mtp_N))} if shape_linked else {}
book=[z for z in csvrows if re.search(r'Sub|Concat|Slice',z['Name'],re.I) and 'INT64' in z['Input Data Types'].upper() and re.search(r'(^|[^0-9])(255|256)([^0-9]|$)',z['Input Shapes'])]
book_groups=defaultdict(list)
for z in book:book_groups[(z['Name'],z['Accelerator Core'])].append(float(z['Duration(us)']))
report['small_int64_metadata_kernels']=[{'name':k[0],'core':k[1],'duration_us':stats(v),'sum_us':sum(v)} for k,v in book_groups.items()]
report['small_int64_metadata_scope']='Name/dtype/shape-filtered255/256-int64 Sub/Concat/Slice tasks; not alone a complete causal attribution or critical path sum.'
report['physical_task_types']=dict(Counter(z['args']['Task Type'] for z in e if 'Task Type' in z.get('args',{})))

(r/('prefix-count-c4-supply-analysis-'+args.mode+'-v1.json')).write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'large_rope_slice_count':len(large_slice),'cumsum_count':len(cumsum),'cumsum_task_types':dict(Counter(z['name']+'|'+z['args']['Task Type'] for z in e if 'Cumsum' in z.get('name','') and 'Task Type' in z.get('args',{}))),'opaque_groups':{k:{a:b for a,b in v.items() if a!='calls'} for k,v in groups.items()},'sync_durations_us':[z['sync_cpu_us'] for z in launches]}))
