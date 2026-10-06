"""Join MoE explicit event scopes to queue, CANN and hardware stream in Run249."""
import bisect
import collections
import importlib.util
import hashlib
import json
from pathlib import Path


def reduce(r,q,owners):
    cpu=[(int(a),int(b),n) for a,b,n,*_ in r['cpu']]
    moes=sorted((a,b) for a,b,n in cpu if n=='vllm::moe_forward_shared')
    assert len(moes)==380
    flow=collections.defaultdict(set)
    for g,c in q['connection_ids']:flow[g].add(c)
    deqs_by_flow={}
    for x in q['queue']:
        if x[2].startswith('Dequeue@'):
            for c in flow[x[3]]:deqs_by_flow[c]=x
    enqs={}
    for x in q['queue']:
        if x[2] in ('Enqueue@record_event','Enqueue@wait_event'):
            enqs.setdefault(x[2],[]).append(x)
    for xs in enqs.values():xs.sort(key=lambda x:int(x[0]))
    starts={n:[int(x[0]) for x in xs] for n,xs in enqs.items()}
    ts=sorted([x for x in r['tasks'] if x[7] is not None],key=lambda x:x[7])
    task_starts=[x[7] for x in ts]
    selected=[]
    for a,b,n in cpu:
        if n not in ('Event::record','Event::wait'):continue
        i=bisect.bisect_right(moes,(a,float('inf')))-1
        if i>=0 and moes[i][0]<=a and b<=moes[i][1]:selected.append((a,b,n,i))
    result={};examples=[]
    _,_,gaps=owners.segments(r,q)
    for name,count,queue_name,api in [('Event::record',1520,'Enqueue@record_event','aclrtRecordEvent'),('Event::wait',1140,'Enqueue@wait_event','aclrtStreamWaitEvent')]:
        rows=[x for x in selected if x[2]==name];assert len(rows)==count
        streams=collections.Counter();unmatched=[];overlap=0;gidx=0
        for a,b,n,mi in sorted(rows):
            j=bisect.bisect_left(starts[queue_name],a);xs=enqs[queue_name];match=[]
            while j<len(xs) and int(xs[j][0])<b:
                if int(xs[j][1])<=b:match.append(xs[j])
                j+=1
            assert len(match)==1,(a,b,name,len(match))
            enq=match[0];dq=[deqs_by_flow[c] for c in flow[enq[3]] if c in deqs_by_flow]
            assert len(dq)==1
            dq=dq[0];dl,dh=int(dq[0]),int(dq[1])
            k=bisect.bisect_left(task_starts,dl);tasks=[]
            while k<len(ts) and ts[k][7]<=dh:
                t=ts[k]
                if t[10]==dq[4] and t[9]==api:tasks.append(t)
                k+=1
            if not tasks:unmatched.append(dict(scope=[a,b],enqueue=enq,dequeue=dq))
            for t in tasks:streams[t[2]]+=1
            while gidx<len(gaps) and gaps[gidx][1]<=a:gidx+=1
            j=gidx
            while j<len(gaps) and gaps[j][0]<b:
                overlap+=max(0,min(b,gaps[j][1])-max(a,gaps[j][0]));j+=1
            if mi in (0,152,379) and len(examples)<24:
                examples.append(dict(moe_ordinal=mi,name=n,scope=[a,b],enqueue=enq,dequeue=dq,tasks=tasks))
        result[name]=dict(scope_count=len(rows),inclusive_ms=sum(b-a for a,b,*_ in rows)/1e6,
                          hardware_stream_task_counts=dict(streams),no_hardware_task_count=len(unmatched),
                          no_hardware_task_examples=unmatched[:3],
                          no_hardware_task_rows_sha256=hashlib.sha256(json.dumps(unmatched).encode()).hexdigest(),
                          local_pre_enqueue_gap_intersection_ms=overlap/1e6)
        if name=='Event::record':
            assert not unmatched and streams=={47:count},(name,streams,len(unmatched))
        else:
            # Explicit waits have a CANN API but no TASK rows in this capture.
            # Keep that coverage limit rather than filling in a hardware stream.
            assert not streams and len(unmatched)==count
    return dict(rank=r['rank'],db_sha256=r['db_sha256'],explicit_moe_events=result,examples=examples,
                limits=['Queue/CANN/hardware join confirms all explicit records use stream47. Explicit waits have no matched TASK row; their target stream is established by installed source, not invented from a missing hardware row.',
                        'Source dataflow supplies producer/consumer pairing and the disabled stream-switch contract. Targeted original CANN/raw excerpts independently confirm a wait API inside its queue Dequeue. All HCCL/DCP internals remain necessary.',
                        'ON-profile scope totals and local exposure are not OFF-profile savings predictions or mathematical compute floors.'])


if __name__=='__main__':
    here=Path(__file__).resolve().parent;p=here.parents[1]
    s=importlib.util.spec_from_file_location('owners',here/'reduce_gap_owners.py');x=importlib.util.module_from_spec(s);s.loader.exec_module(x)
    rows=[]
    for rank in (13,15):
        r=x.read(p/'jobs/GLM53-DECODE-TRACE-20261006-B/reduced'/('db_rank%d.json.gz'%rank))
        q=x.read(p/'jobs/GLM53-DECODE-QUEUES-20261006'/('queues_rank%d.json.gz'%rank))
        out=reduce(r,q,x);rows.append(out)
        print(rank,{k:{j:v[j] for j in ['scope_count','inclusive_ms','hardware_stream_task_counts','no_hardware_task_count','local_pre_enqueue_gap_intersection_ms']} for k,v in out['explicit_moe_events'].items()})
    (here/'moe_event_streams.json').write_text(json.dumps(rows,indent=2)+'\n')
