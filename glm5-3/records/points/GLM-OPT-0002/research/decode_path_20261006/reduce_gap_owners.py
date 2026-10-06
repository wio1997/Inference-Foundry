"""Join exposed pre-enqueue gaps to contemporaneous CPU scopes, not next-op labels.

Existing Run249 files only. Scope wall times are not actual CPU runtime and no
result is an OFF-profile savings prediction. Shortest active scope is a
deterministic overlap convention, not a reconstructed Python call stack.
"""
import bisect
import collections
import gzip
import hashlib
import heapq
import json
from pathlib import Path


def read(path):
    return json.loads(gzip.decompress(path.read_bytes()))


def segments(r, q):
    flow = collections.defaultdict(set)
    for group, connection in q['connection_ids']:
        flow[group].add(connection)
    enqueue = {}
    for x in q['queue']:
        if x[2].startswith('Enqueue@'):
            for c in flow[x[3]]:
                enqueue[c] = x
    deqs = sorted([x for x in q['queue'] if x[2].startswith('Dequeue@')],key=lambda x:int(x[0]))
    starts = [int(x[0]) for x in deqs]
    tasks = sorted(r['tasks'],key=lambda x:x[0])
    embeddings = [x for x in tasks if 'Embedding' in (x[6] or '')]
    copies = [x for x in tasks if x[2] == 45 and x[9] == 'aclrtMemcpyAsyncWithCondition']
    lo, hi = embeddings[0][0], copies[-1][1]
    main = [x for x in tasks if x[2] == 47]
    out = []
    for prev, task in zip(main,main[1:]):
        if task[7] is None:
            continue
        a,b = max(lo,prev[1]), min(hi,task[0],task[7])
        if b <= a:
            continue
        i = bisect.bisect_right(starts,task[7])-1
        assert i >= 0
        d = deqs[i]
        assert int(d[0]) <= task[7] <= int(d[1]) and d[4] == task[10]
        matches = [enqueue[c] for c in flow[d[3]] if c in enqueue]
        assert len(matches) == 1
        b = min(b,int(matches[0][0]))
        if b > a:
            out.append((a,b))
    assert all(a[1] <= b[0] for a,b in zip(out,out[1:]))
    return lo,hi,out


def attribution(r, q):
    lo,hi,gaps = segments(r,q)
    cpu = [(int(x[0]),int(x[1]),x[2]) for x in r['cpu'] if int(x[1]) > lo and int(x[0]) < hi]
    # Every boundary changes the active CPU-scope set or exposed-gap status.
    events = []
    for i,(a,b,name) in enumerate(cpu):
        if b > a:
            events.extend([(max(a,lo),1,i),(min(b,hi),-1,i)])
    for a,b in gaps:
        events.extend([(a,2,-1),(b,-2,-1)])
    events.sort()
    active=set();leaf=[];model=[];inside=0;previous=lo
    leaf_ms=collections.Counter();phase_ms=collections.Counter()
    # Retain exact candidate containment separately from shortest-scope ownership.
    gathers=sorted((a,b) for a,b,n in cpu if n=='c10d::allgather_')
    assert len(gathers)==380
    copies=[];splits=[]
    for a,b,n in cpu:
        if n=='aten::copy_':
            i=bisect.bisect_right(gathers,(a,float('inf')))-1
            if i>=0 and gathers[i][0]<=a and b<=gathers[i][1]:copies.append((a,b))
    for a,b in gathers:
        matches=[(s,e) for s,e,n in cpu if n=='aten::tensor_split' and 0 <= a-e < 100000]
        assert matches
        splits.append(max(matches,key=lambda z:z[1]))
    assert len(copies)==6080 and len(set(splits))==380
    def overlap(intervals):
        points=[]
        for a,b in intervals:points.extend([(a,1),(b,-1)])
        points.sort();merged=[];count=0;start=None
        for t,d in points:
            if count==0:start=t
            count+=d
            if count==0:merged.append((start,t))
        total=0;i=0
        for a,b in merged:
            while i<len(gaps) and gaps[i][1]<=a:i+=1
            j=i
            while j<len(gaps) and gaps[j][0]<b:
                total+=max(0,min(b,gaps[j][1])-max(a,gaps[j][0]));j+=1
        return total/1e6
    for t,kind,i in events:
        if t>previous and inside:
            while leaf and leaf[0][2] not in active:heapq.heappop(leaf)
            while model and model[0][2] not in active:heapq.heappop(model)
            name=leaf[0][3] if leaf else 'NO_EXPORTED_CPU_SCOPE'
            phase=model[0][3] if model else 'OUTSIDE_MLA_MOE_SCOPES'
            leaf_ms[name]+=(t-previous)/1e6;phase_ms[phase]+=(t-previous)/1e6
        previous=t
        if kind==1:
            active.add(i);a,b,n=cpu[i];heapq.heappush(leaf,(b-a,-a,i,n))
            if n in ('vllm::mla_forward','vllm::moe_forward_shared'):
                heapq.heappush(model,(b-a,-a,i,n))
        elif kind==-1:active.discard(i)
        elif kind==2:inside+=1
        else:inside-=1
        assert inside in (0,1)
    total=sum(b-a for a,b in gaps)/1e6
    assert abs(sum(leaf_ms.values())-total)<1e-5 and abs(sum(phase_ms.values())-total)<1e-5
    return dict(rank=r['rank'],db_sha256=r['db_sha256'],window=[lo,hi],
                pre_enqueue_gap_ms=total,segment_count=len(gaps),
                model_scope_coverage_ms=dict(phase_ms.most_common()),
                shortest_active_scope_ms=dict(leaf_ms.most_common()),
                H4_removed_scope_local_gap_intersection_ms=dict(copy=overlap(copies),finalize_split=overlap(splits)),
                limits=[
                    'Disjoint exposed pre-enqueue gaps only. Shortest active scope convention avoids inclusive double counting; no Python stack is exported.',
                    'NO_EXPORTED_CPU_SCOPE means no recorded main-thread operator, not zero CPU work; MLA/MoE scopes may contain unexported Python preparation.',
                    'Scope owner is contemporaneous work. It is not attribution to the next enqueue name, and not a claim that the entire operator is removable.',
                    'The local gap intersection understates propagated savings when earlier host work delays later enqueues; no global patch benefit follows from this overlap.'
                ])


if __name__=='__main__':
    here=Path(__file__).resolve().parent;point=here.parents[1]
    rows=[]
    for rank in (13,15):
        a=point/'jobs/GLM53-DECODE-TRACE-20261006-B/reduced'/('db_rank%d.json.gz'%rank)
        b=point/'jobs/GLM53-DECODE-QUEUES-20261006'/('queues_rank%d.json.gz'%rank)
        out=attribution(read(a),read(b))
        out['input_sha256']={str(p.relative_to(point)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (a,b)}
        rows.append(out)
        print(rank,out['pre_enqueue_gap_ms'],out['model_scope_coverage_ms'],out['H4_removed_scope_local_gap_intersection_ms'])
    (here/'gap_scope_owners.json').write_text(json.dumps(rows,indent=2)+'\n')
