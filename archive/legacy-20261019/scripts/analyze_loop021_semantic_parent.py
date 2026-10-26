#!/usr/bin/env python3
"""Attribute TP0 draft-owned async device tasks to innermost CPU op parents."""
import bisect, glob, heapq, json, statistics
from collections import defaultdict
from pathlib import Path
ROOT=Path('/data/wio/Inference_Foundry')
trace=glob.glob(str(ROOT/'evidence/20260920_decode_c12_profile/torch_raw/*rank0*/ASCEND_PROFILER_OUTPUT/trace_view.json'))[0]
events=json.loads(Path(trace).read_text())
cpu=defaultdict(list); drafts=defaultdict(list); flow_s=[]; flow_f={}; device={}
for e in events:
    ph=e.get('ph'); cat=e.get('cat'); pid=int(e.get('pid',-1)); tid=int(e.get('tid',-1)); ts=float(e.get('ts',0))
    if ph=='X' and cat=='cpu_op':
        dur=float(e.get('dur',0)); cpu[(pid,tid)].append((ts,ts+dur,dur,e.get('name','')))
        if e.get('name')=='draft_token': drafts[(pid,tid)].append((ts,ts+dur))
    elif ph=='s' and cat=='async_npu': flow_s.append((pid,tid,ts,e.get('id')))
    elif ph=='f' and cat=='async_npu': flow_f[e.get('id')]=(pid,tid,ts)
    elif ph=='X' and cat is None: device[(pid,tid,ts)]=(float(e.get('dur',0)),e.get('name',''))
for k in drafts:drafts[k].sort()
def draft_index(key,t):
    rows=drafts.get(key,()); starts=[x[0] for x in rows]
    i=bisect.bisect_right(starts,t)-1
    return i if i>=0 and t<rows[i][1] else None
owned=defaultdict(list)
for pid,tid,ts,fid in flow_s:
    di=draft_index((pid,tid),ts)
    if di is not None:owned[(pid,tid)].append((ts,fid,di))
parent_count=defaultdict(int); parent_dur=defaultdict(float); pair_count=defaultdict(int); pair_dur=defaultdict(float); intervals=defaultdict(list)
unmatched_finish=unmatched_device=0
for key,flows in owned.items():
    ops=sorted(cpu[key]); flows.sort(); heap=[]; semheap=[]; j=0
    for ts,fid,di in flows:
        while j<len(ops) and ops[j][0]<=ts:
            a,b,d,n=ops[j]; heapq.heappush(heap,(d,b,a,n));
            if n.startswith('vllm::') or n in ('draft_token','forward','prepare input'): heapq.heappush(semheap,(d,b,a,n))
            j+=1
        while heap and heap[0][1]<ts:heapq.heappop(heap)
        while semheap and semheap[0][1]<ts:heapq.heappop(semheap)
        leaf=heap[0][3] if heap else '<none>'
        sem=semheap[0][3] if semheap else '<none>'
        parent=sem+' | '+leaf
        fin=flow_f.get(fid)
        if fin is None: unmatched_finish+=1; continue
        task=device.get(fin)
        if task is None: unmatched_device+=1; continue
        dur,name=task; parent_count[parent]+=1;parent_dur[parent]+=dur;pair_count[(parent,name)]+=1;pair_dur[(parent,name)]+=dur
        a,b=drafts[key][di]; s=fin[2]; e=s+dur
        if s<b and e>a:intervals[(parent,key,di)].append((max(s,a),min(e,b)))
def union_us(xs):
    if not xs:return 0
    xs=sorted(xs);a,b=xs[0];total=0
    for c,d in xs[1:]:
        if c>b:total+=b-a;a,b=c,d
        else:b=max(b,d)
    return total+b-a
parent_scope=defaultdict(list)
for (p,key,di),xs in intervals.items():parent_scope[p].append(union_us(xs)/1000)
parents=[]
for p,c in parent_count.items():
    vals=parent_scope[p]
    parents.append({'cpu_parent':p,'task_count':c,'task_duration_sum_ms':parent_dur[p]/1000,'scope_count':len(vals),'clipped_union_sum_ms':sum(vals),'clipped_union_median_ms_when_present':statistics.median(vals) if vals else 0})
parents.sort(key=lambda x:x['clipped_union_sum_ms'],reverse=True)
pairs=[]
for (p,n),c in pair_count.items():pairs.append({'cpu_parent':p,'device_task':n,'count':c,'duration_sum_ms':pair_dur[(p,n)]/1000})
pairs.sort(key=lambda x:x['duration_sum_ms'],reverse=True)
out={'source':trace,'method':'async flow CPU start restricted to same-thread draft_token; semantic parent is minimum-duration active vllm:: scope and leaf parent is minimum-duration active cpu_op; flow finish exact-matches device X; intervals clipped to owning draft scope','draft_scopes':sum(map(len,drafts.values())),'owned_flow_starts':sum(map(len,owned.values())),'matched_tasks':sum(parent_count.values()),'unmatched_finish':unmatched_finish,'unmatched_device':unmatched_device,'parents_by_clipped_union':parents[:50],'parent_device_pairs_by_duration':pairs[:100]}
p=ROOT/'evidence/20260921_loop021_device_parent/semantic_leaf_attribution_tp0.json';p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'counts':{k:out[k] for k in ('draft_scopes','owned_flow_starts','matched_tasks','unmatched_finish','unmatched_device')},'parents':parents[:20],'pairs':pairs[:25]},indent=2))
