#!/usr/bin/env python3
import bisect, glob, json, statistics
from collections import Counter, defaultdict
from pathlib import Path
ROOT=Path('/data/wio/Inference_Foundry')
trace=glob.glob(str(ROOT/'evidence/20260920_decode_c12_profile/torch_raw/*rank0*/ASCEND_PROFILER_OUTPUT/trace_view.json'))[0]
events=json.loads(Path(trace).read_text())
starts={}; finishes=[]; device=defaultdict(list); drafts=[]
for e in events:
    ph=e.get('ph'); cat=e.get('cat')
    if ph=='s' and cat=='async_npu': starts[e.get('id')]=(float(e['ts']),int(e['pid']),int(e['tid']))
    elif ph=='f' and cat=='async_npu': finishes.append((e.get('id'),int(e['pid']),int(e['tid']),float(e['ts'])))
    elif ph=='X' and cat is None:
        device[(int(e['pid']),int(e['tid']))].append((float(e['ts']),float(e['dur']),e.get('name','')))
    elif ph=='X' and cat=='cpu_op' and e.get('name')=='draft_token': drafts.append((float(e['ts']),float(e['ts'])+float(e['dur']),int(e['pid']),int(e['tid'])))
for k in device: device[k].sort()
drafts.sort(); by_thread=defaultdict(list)
for i,(a,b,pid,tid) in enumerate(drafts): by_thread[(pid,tid)].append((a,b,i))
def in_draft(pid,tid,t):
    rows=by_thread.get((pid,tid),())
    if not rows:return None
    starts=[x[0] for x in rows]; j=bisect.bisect_right(starts,t)-1
    return rows[j][2] if j>=0 and t<rows[j][1] else None
def match(pid,tid,t):
    xs=device.get((pid,tid),())
    if not xs:return None,None
    i=bisect.bisect_right(xs,(t,float('inf'),chr(0x10ffff)))-1
    cand=[]
    if i>=0:cand.append(xs[i])
    if i+1<len(xs):cand.append(xs[i+1])
    if not cand:return None,None
    x=min(cand,key=lambda q:abs(q[0]-t)); return x,x[0]-t
matched=[]; draft_matched=[]; deltas=[]; missing_start=0
for fid,pid,tid,ft in finishes:
    src=starts.get(fid)
    if src is None: missing_start+=1; continue
    st,spid,stid=src
    x,delta=match(pid,tid,ft); matched.append((st,ft,pid,tid,x,delta))
    if x is not None:deltas.append(delta)
    di=in_draft(spid,stid,st)
    if di is not None:draft_matched.append((di,(st,ft,pid,tid,x,delta)))
absd=sorted(abs(x) for x in deltas)
within=lambda u:sum(x<=u for x in absd)
name_counts=Counter(); dur=[]; per_scope=defaultdict(list)
for di,rec in draft_matched:
    x,delta=rec[4],rec[5]
    if x is not None and abs(delta)<=1.0:
        name_counts[x[2]]+=1; dur.append(x[1]); per_scope[di].append(x)
def union_us(xs):
    z=sorted((x[0],x[0]+x[1]) for x in xs)
    if not z:return 0.0
    a,b=z[0]; total=0.0
    for c,d in z[1:]:
        if c>b:total+=b-a;a,b=c,d
        else:b=max(b,d)
    return total+b-a
per=[union_us(v)/1000 for v in per_scope.values()]
clipped=[]
for i,v in per_scope.items():
    a,b,_,_=drafts[i]
    clipped.append(union_us([(max(x[0],a),max(0.0,min(x[0]+x[1],b)-max(x[0],a)),x[2]) for x in v if x[0]<b and x[0]+x[1]>a])/1000)
out={'source':trace,'flow_starts':len(starts),'flow_finishes':len(finishes),'finish_without_start':missing_start,
'device_x':sum(map(len,device.values())),'matched_same_stream':len(deltas),
'absolute_start_delta_us':{'le_0p001':within(.001),'le_0p01':within(.01),'le_0p1':within(.1),'le_1':within(1.0),'median':statistics.median(absd) if absd else None,'p99':absd[int(.99*(len(absd)-1))] if absd else None},
'draft_scopes_all_trace':len(drafts),'flows_cpu_start_in_draft':len(draft_matched),'draft_exact_device_matches_le_1us':len(dur),
'draft_matched_device_duration_sum_ms':sum(dur)/1000,'draft_scope_matched_device_union_ms':{'count':len(per),'median':statistics.median(per) if per else None,'p90':sorted(per)[int(.9*(len(per)-1))] if per else None},'draft_scope_clipped_causal_device_union_ms':{'count':len(clipped),'median':statistics.median(clipped) if clipped else None,'p90':sorted(clipped)[int(.9*(len(clipped)-1))] if clipped else None},
'draft_top_exact_device_names':name_counts.most_common(20),
'interpretation':'A match requires same device pid/tid and device X start within 1 us of the async_npu flow finish. CPU flow start must lie inside draft_token. Durations can overlap across streams; they are causal attribution evidence, not removable wall time.'}
p=ROOT/'evidence/20260921_dspark_audit/async_flow_attribution_tp0.json';p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
