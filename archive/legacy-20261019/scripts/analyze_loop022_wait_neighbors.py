#!/usr/bin/env python3
import bisect,glob,heapq,json,statistics
from collections import Counter,defaultdict
from pathlib import Path
R=Path('/data/wio/Inference_Foundry');trace=glob.glob(str(R/'evidence/20260920_decode_c12_profile/torch_raw/*rank0*/ASCEND_PROFILER_OUTPUT/trace_view.json'))[0]
e=json.loads(Path(trace).read_text());cpu=defaultdict(list);drafts=defaultdict(list);fs=[];ff={};dev=defaultdict(list)
for x in e:
 ph=x.get('ph');cat=x.get('cat');pid=int(x.get('pid',-1));tid=int(x.get('tid',-1));ts=float(x.get('ts',0))
 if ph=='X' and cat=='cpu_op':
  d=float(x.get('dur',0));cpu[(pid,tid)].append((ts,ts+d,d,x.get('name','')))
  if x.get('name')=='draft_token':drafts[(pid,tid)].append((ts,ts+d))
 elif ph=='s' and cat=='async_npu':fs.append((pid,tid,ts,x.get('id')))
 elif ph=='f' and cat=='async_npu':ff[x.get('id')]=(pid,tid,ts)
 elif ph=='X' and cat is None:dev[(pid,tid)].append((ts,float(x.get('dur',0)),x.get('name','')))
for v in drafts.values():v.sort()
for v in dev.values():v.sort()
def inside(key,t):
 v=drafts.get(key,());i=bisect.bisect_right([z[0] for z in v],t)-1;return i if i>=0 and t<v[i][1] else None
owned=defaultdict(list)
for pid,tid,ts,fid in fs:
 i=inside((pid,tid),ts)
 if i is not None:owned[(pid,tid)].append((ts,fid,i))
rows=defaultdict(lambda:{'count':0,'dur':[],'prev':Counter(),'next':Counter(),'next_gap':[]})
for key,flows in owned.items():
 ops=sorted(cpu[key]);flows.sort();heap=[];semheap=[];j=0
 for ts,fid,di in flows:
  while j<len(ops) and ops[j][0]<=ts:
   a,b,d,n=ops[j];heapq.heappush(heap,(d,b,a,n))
   if n.startswith('vllm::') or n=='draft_token':heapq.heappush(semheap,(d,b,a,n))
   j+=1
  while heap and heap[0][1]<ts:heapq.heappop(heap)
  while semheap and semheap[0][1]<ts:heapq.heappop(semheap)
  leaf=heap[0][3] if heap else '';sem=semheap[0][3] if semheap else ''
  fin=ff.get(fid)
  if not fin:continue
  stream=dev.get((fin[0],fin[1]),());k=bisect.bisect_left(stream,(fin[2],-1,''))
  if k>=len(stream) or stream[k][0]!=fin[2]:continue
  task=stream[k]
  if task[2]!='EVENT_WAIT' or sem not in ('draft_token','vllm::moe_forward_shared'):continue
  z=rows[(sem,leaf,fin[1])];z['count']+=1;z['dur'].append(task[1])
  if k:z['prev'][stream[k-1][2]]+=1
  q=k+1
  while q<len(stream) and stream[q][2] in ('EVENT_WAIT','EVENT_RECORD'):q+=1
  if q<len(stream):z['next'][stream[q][2]]+=1;z['next_gap'].append(stream[q][0]-(task[0]+task[1]))
out=[]
for (sem,leaf,tid),z in rows.items():
 out.append({'semantic_parent':sem,'leaf':leaf,'stream':tid,'count':z['count'],'wait_median_ms':statistics.median(z['dur'])/1000,'wait_p90_ms':sorted(z['dur'])[int(.9*(len(z['dur'])-1))]/1000,'previous_tasks':z['prev'].most_common(8),'next_non_event_tasks':z['next'].most_common(8),'next_gap_median_us':statistics.median(z['next_gap']) if z['next_gap'] else None})
out.sort(key=lambda x:(x['semantic_parent'], -x['wait_median_ms']))
p=R/'evidence/20260921_loop022_event_dependency/wait_stream_neighbors_tp0.json';p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps({'source':trace,'method':'exact async flow plus same device stream neighbor lookup','rows':out},indent=2)+'\n');print(json.dumps(out,indent=2))
