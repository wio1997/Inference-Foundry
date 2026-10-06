"""All-rank HCOM and EP readiness from existing Run249, no device work.

Exact HCOM names/groups; EP ordinal correspondence requires the observed same
76-call-per-round model order on every rank. These extents overlap and are not
an exclusive wall budget or a claim that the EP kernels are only transfer.
"""
from pathlib import Path
import collections,gzip,hashlib,json,statistics


def reduce(rows):
    totals=[]
    for key,ops in rows.items():
        device_latest=max(ops,key=lambda x:x['device_start_ns'])
        host_latest=max(ops,key=lambda x:x['host_start_ns'])
        host_end_latest=max(ops,key=lambda x:x['host_end_ns'])
        latest_start=device_latest['device_start_ns']
        final_end=max(x['device_end_ns'] for x in ops)
        assert len(ops)==16 and final_end>=latest_start
        totals.append(dict(key=list(key),latest_device_rank=device_latest['rank'],latest_host_start_rank=host_latest['rank'],latest_host_end_rank=host_end_latest['rank'],
            device_start_skew_us=(latest_start-min(x['device_start_ns'] for x in ops))/1000,
            latest_entry_to_final_end_us=(final_end-latest_start)/1000,
            latest_rank_host_end_to_device_us=(latest_start-device_latest['host_end_ns'])/1000,
            latest_device_row=device_latest))
    by_round=[]
    for round_id in range(1,6):
        sub=[x for x in totals if x['key'][0]==round_id]
        by_round.append(dict(round=round_id,operations=len(sub),latest_host_start_equals_device=sum(x['latest_host_start_rank']==x['latest_device_rank'] for x in sub),latest_host_end_equals_device=sum(x['latest_host_end_rank']==x['latest_device_rank'] for x in sub),
            latest_device_rank_counts=dict(collections.Counter(x['latest_device_rank'] for x in sub)),
            latest_host_start_rank_counts=dict(collections.Counter(x['latest_host_start_rank'] for x in sub)),latest_host_end_rank_counts=dict(collections.Counter(x['latest_host_end_rank'] for x in sub)),
            latest_host_end_to_device_median_us=statistics.median(x['latest_rank_host_end_to_device_us'] for x in sub),
            latest_host_end_to_device_below_200us=sum(x['latest_rank_host_end_to_device_us']<200 for x in sub),
            summed_start_skew_ms=sum(x['device_start_skew_us'] for x in sub)/1000,
            summed_latest_entry_to_final_end_ms=sum(x['latest_entry_to_final_end_us'] for x in sub)/1000,
            examples=sorted(sub,key=lambda x:-x['device_start_skew_us'])[:2]))
    return by_round


def main():
    here=Path(__file__).resolve().parent;point=here.parents[1]
    hcom=collections.defaultdict(list);ep=collections.defaultdict(list);identities=[];other_communication=[]
    for rank in range(16):
        path=point/f'jobs/GLM53-DECODE-TRACE-20261006-B/reduced/db_rank{rank}.json.gz'
        raw=path.read_bytes();d=json.loads(gzip.decompress(raw))
        identities.append(dict(rank=rank,input_sha256=hashlib.sha256(raw).hexdigest(),db_sha256=d['db_sha256']))
        embeddings=sorted(t for t in d['tasks'] if 'aclnnEmbedding_Gather' in (t[6] or ''))
        copies=sorted(t for t in d['tasks'] if t[2]==45 and t[9]=='aclrtMemcpyAsyncWithCondition')
        assert len(embeddings)==10 and len(copies)==5
        windows=[(embeddings[i*2][0],copies[i][1]) for i in range(5)]
        for c in d['comm']:
            if not c[0].startswith('hcom_'):
                other_communication.append(dict(rank=rank,raw_row=c))
                continue
            matches=[i+1 for i,(a,b) in enumerate(windows) if a<=c[2]<=b]
            assert len(matches)==1 and c[8] is not None and c[9] is not None
            hcom[(matches[0],c[1],c[0])].append(dict(rank=rank,device_start_ns=c[2],device_end_ns=c[3],host_start_ns=c[8],host_end_ns=c[9],connection=c[4],count=c[6]))
        for name in ('MoeDistributeDispatchV2','MoeDistributeCombineV2'):
            tasks=sorted(t for t in d['tasks'] if t[6]==name)
            assert len(tasks)==380
            for ordinal,t in enumerate(tasks):
                round_id=ordinal//76+1
                assert windows[round_id-1][0]<=t[0]<=windows[round_id-1][1]
                assert t[7] is not None and t[8] is not None
                ep[(round_id,name,ordinal%76)].append(dict(rank=rank,device_start_ns=t[0],device_end_ns=t[1],host_start_ns=t[7],host_end_ns=t[8],connection=t[4]))
    assert len(hcom)==1901 and len(ep)==760
    result=dict(existing_Run249_only=True,input_identities=identities,other_communication=other_communication,HCOM=reduce(hcom),EP=reduce(ep),
        limits=['All clocks are host167 profile-ON. Exact named HCOM correspondence; EP order is corroborated by identical per-round layer traversal and 76 dispatch/combine calls on all ranks.',
        'Start skew is peer-readiness evidence, not removable transfer time. Latest-entry-to-final-end includes whatever device work the actual kernel performs after latest entry, not a theoretical lower bound.',
        'Sums overlap across streams and operations. They do not partition step wall or predict profiling-OFF265ms/token savings.',
        'Absent event ID/release metadata and Scheduler/Executor entry/exit timestamps prevent reconstructing a complete exported DAG and exact four-layer CPU wall budget. Source supplies necessary ordering, not missing timestamps.'])
    (here/'rendezvous_frontier.json').write_text(json.dumps(result,indent=2)+'\n')
    for kind in ('HCOM','EP'):
        for row in result[kind]:print(kind,json.dumps({k:v for k,v in row.items() if k!='examples'}))


if __name__=='__main__':main()
