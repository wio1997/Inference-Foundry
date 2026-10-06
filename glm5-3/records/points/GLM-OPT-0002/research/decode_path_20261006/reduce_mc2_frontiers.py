"""Run249 read-only same-call MC2 producer/consumer closure, no torch import.

Partitions scope WALL, and exposed local launch intervals, never CPU runtime or
OFF savings. Each containment/queue relation is asserted; unknown is retained.
"""
import bisect, collections, gzip, hashlib, json
from pathlib import Path
from reduce_gap_owners import segments
from reduce_prepare_criticality import merged, duration, intersection


def read(p): return json.loads(gzip.decompress(p.read_bytes()))
def ns(xs): return sum(b-a for a,b in merged(xs))
def overlap(a,b,c,d): return max(0,min(b,d)-max(a,c))


def partition(a,b,labels,enqueue_start):
    points={a,b}
    clipped=[]
    for s,e,n,priority in labels:
        s=max(a,s);e=min(b,e)
        if e>s:clipped.append((s,e,n,priority));points.update((s,e))
    if a<enqueue_start<b:points.add(enqueue_start)
    points=sorted(points);out=collections.Counter()
    for s,e in zip(points,points[1:]):
        choices=[(p,y-x,n) for x,y,n,p in clipped if x<=s and e<=y]
        n=min(choices)[2] if choices else ('native_unexported_before_enqueue' if s<enqueue_start else 'native_unexported_after_enqueue')
        out[n]+=e-s
    assert sum(out.values())==b-a
    return out


def reduce(data,queue,cann):
    cpu=sorted((int(x[0]),int(x[1]),x[2]) for x in data['cpu']);cs=[x[0] for x in cpu]
    moes=[x for x in cpu if x[2]=='vllm::moe_forward_shared'];assert len(moes)==380
    _,_,gaps=segments(data,queue)
    flow=collections.defaultdict(set)
    for group,c in queue['connection_ids']:flow[group].add(c)
    enq={}
    for q in queue['queue']:
        if q[2].startswith('Enqueue@'):
            for c in flow[q[3]]:
                assert c not in enq or enq[c]==q,'different enqueue for one connection'
                enq[c]=q
    deq=sorted((x for x in queue['queue'] if x[2].startswith('Dequeue@')),key=lambda x:int(x[0]));ds=[int(x[0]) for x in deq]
    previous_deq={};last={}
    for q in deq:
        if q[4] in last:previous_deq[id(q)]=last[q[4]]
        last[q[4]]=q
    api=sorted((int(x[0]),int(x[1]),x[2],x[4]) for x in cann['rows']);aps=[x[0] for x in api]
    tasks={kind:sorted((x for x in data['tasks'] if x[6]==name),key=lambda x:x[0]) for kind,name in [('dispatch','MoeDistributeDispatchV2'),('combine','MoeDistributeCombineV2')]}
    for xs in tasks.values():assert len(xs)==380
    native={kind:[x for x in cpu if x[2]=='npu::npu_moe_distribute_'+kind+'_v2'] for kind in tasks}
    assert all(len(x)==380 for x in native.values())
    embeddings=sorted(x[0] for x in data['tasks'] if 'Embedding' in (x[6] or ''))
    copies=sorted(x[1] for x in data['tasks'] if x[2]==45 and x[9]=='aclrtMemcpyAsyncWithCondition')
    assert len(embeddings)==10 and len(copies)==5
    out=[]
    for i,moe in enumerate(moes):
        a,b,_=moe
        for kind in tasks:
            s,e,name=native[kind][i];assert a<=s<e<=b
            t=tasks[kind][i];hs,he=t[7:9]
            j=bisect.bisect_right(ds,hs)-1;assert j>=0
            dq=deq[j];assert int(dq[0])<=hs<=he<=int(dq[1]) and dq[4]==t[10]
            matches=[enq[c] for c in flow[dq[3]] if c in enq];assert len(matches)==1
            eq=matches[0];e0,e1=int(eq[0]),int(eq[1]);d0,d1=int(dq[0]),int(dq[1])
            assert eq[2].split('@',1)[1]==dq[2].split('@',1)[1]
            assert s<=e0<=e1<=e and e0<=d0
            children=[x for x in cpu[bisect.bisect_left(cs,s):bisect.bisect_right(cs,e)] if x[1]<=e and x[2]!=name]
            apis=[x for x in api[bisect.bisect_left(aps,s):bisect.bisect_right(aps,e)] if x[1]<=e]
            workspace=[x for x in apis if 'MoeDistribute' in x[2] and x[2].endswith('GetWorkspaceSize')]
            assert len(workspace)==1 and workspace[0][1]<=e0
            labels=[]
            for x,y,n,tid in apis:
                assert tid==cann['main_tid']
                category='workspace_query' if 'MoeDistribute' in n and n.endswith('GetWorkspaceSize') else ('allocator_event_query' if n=='aclrtQueryEventStatus' else 'other_exported_CANN')
                labels.append((x,y,category,0))
            for x,y,n in children:
                if n=='empty_tensor' or n.startswith(('aten::empty','aten::empty_like','aten::empty_strided')):labels.append((x,y,'allocation_exclusive',1))
            labels.append((e0,e1,'enqueue_call',0))
            full=partition(s,e,labels,e0)
            exposed=collections.Counter()
            for x,y in gaps:
                x=max(x,s);y=min(y,e)
                if y>x:exposed.update(partition(x,y,labels,e0))
            prev=previous_deq.get(id(dq));assert prev is not None and int(prev[1])<=d0
            # No exported callback between the previous end and this dequeue.
            # This is wall time: not proof the consumer was runnable/on CPU.
            no_callback=[max(s,int(prev[1])),min(e0,d0)]
            no_callback_ns=max(0,no_callback[1]-no_callback[0])
            step=i//76+1;assert embeddings[(step-1)*2]<=t[0]<=copies[step-1]
            out.append(dict(rank=data['rank'],round=step,moe_ordinal=i%76,kind=kind,MoE_scope_ns=[a,b],native_scope_ns=[s,e],workspace_ns=list(workspace[0][:2]),enqueue_ns=[e0,e1],dequeue_ns=[d0,d1],previous_dequeue_ns=[int(prev[0]),int(prev[1])],launch_ns=[hs,he],device_ns=t[:2],connection=t[4],producer_tid=cann['main_tid'],consumer_tid=dq[4],native_wall_partition_ns=dict(full),native_local_gap_partition_ns=dict(exposed),producer_pre_enqueue_ns=e0-s,queue_start_to_dequeue_ns=d0-e0,dequeue_to_launch_ns=hs-d0,no_exported_callback_pre_enqueue_ns=no_callback_ns))
    summaries=[]
    for step in range(1,6):
        for kind in tasks:
            rows=[x for x in out if x['round']==step and x['kind']==kind]
            wall=collections.Counter();exposed=collections.Counter()
            for x in rows:wall.update(x['native_wall_partition_ns']);exposed.update(x['native_local_gap_partition_ns'])
            summaries.append(dict(round=step,kind=kind,calls=len(rows),native_wall_ms=sum(wall.values())/1e6,wall_partition_ms={k:v/1e6 for k,v in wall.items()},local_gap_partition_ms={k:v/1e6 for k,v in exposed.items()},producer_pre_enqueue_ms=sum(x['producer_pre_enqueue_ns'] for x in rows)/1e6,queue_start_to_dequeue_ms=sum(x['queue_start_to_dequeue_ns'] for x in rows)/1e6,dequeue_to_launch_ms=sum(x['dequeue_to_launch_ns'] for x in rows)/1e6,no_exported_callback_pre_enqueue_ms=sum(x['no_exported_callback_pre_enqueue_ns'] for x in rows)/1e6))
    return dict(rank=data['rank'],db_sha256=data['db_sha256'],calls=out,summaries=summaries)


if __name__=='__main__':
    here=Path(__file__).resolve().parent;point=here.parents[1];cannpath=here/'cann_frontend_13_15.json.gz';cann=read(cannpath);result=[]
    for rank in [13,15]:
        paths=[point/f'jobs/GLM53-DECODE-TRACE-20261006-B/reduced/db_rank{rank}.json.gz',point/f'jobs/GLM53-DECODE-QUEUES-20261006/queues_rank{rank}.json.gz']
        row=reduce(*(read(p) for p in paths),next(x for x in cann if x['rank']==rank))
        row['input_sha256']={str(p.relative_to(point)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths+[cannpath]};result.append(row)
        wall=collections.Counter();exposed=collections.Counter()
        for s in row['summaries']:wall.update(s['wall_partition_ms']);exposed.update(s['local_gap_partition_ms'])
        print(rank,'wall',dict(wall),'exposed',dict(exposed))
        print(rank,'queue_totals_ms',{k:sum(x[k] for x in row['summaries']) for k in ['producer_pre_enqueue_ms','queue_start_to_dequeue_ms','dequeue_to_launch_ms','no_exported_callback_pre_enqueue_ms']})
    comparisons=[]
    for left,right in zip(result[0]['calls'],result[1]['calls']):
        assert (left['round'],left['moe_ordinal'],left['kind'])==(right['round'],right['moe_ordinal'],right['kind'])
        comparisons.append(dict(round=left['round'],moe_ordinal=left['moe_ordinal'],kind=left['kind'],D15_minus_D13_enqueue_us=(right['enqueue_ns'][0]-left['enqueue_ns'][0])/1e3,D15_minus_D13_native_entry_us=(right['native_scope_ns'][0]-left['native_scope_ns'][0])/1e3,D15_minus_D13_native_pre_enqueue_duration_us=(right['producer_pre_enqueue_ns']-left['producer_pre_enqueue_ns'])/1e3,D15_minus_D13_device_start_us=(right['device_ns'][0]-left['device_ns'][0])/1e3))
    payload=dict(ranks=result,same_call_comparisons=comparisons,limits=['Existing Run249 profiler-ON same-host clock only. No new model request; no OFF saving prediction.','Native scope wall partition is disjoint. CANN/allocator/queue scopes can contain profiler/scheduling; native unexported time is unknown, not deletable work.','Queue push and scheduler state are not exported. Enqueue-start is an earliest-ready bound, and no-exported-callback wall is not consumer CPU-idle/runnable proof.','Same round/ordinal/kernel/queue-flow linkage is checked. Native scope/start-to-dequeue/dequeue-to-launch durations overlap device and other threads; do not sum into step wall.','Local exposed pre-enqueue intersection misses propagated inter-rank effects. Prepare, gating and between-native work remain outside this native-only partition.'])
    with gzip.open(here/'mc2_frontiers.json.gz','wt') as f:json.dump(payload,f,separators=(',',':'))
    for row in result:del row['calls']
    del payload['same_call_comparisons']
    payload['detail_file']='mc2_frontiers.json.gz'
    (here/'mc2_frontiers_summary.json').write_text(json.dumps(payload,indent=2)+'\n')
