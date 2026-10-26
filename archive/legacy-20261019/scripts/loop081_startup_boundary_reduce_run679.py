"""Disjoint Host endpoint partition with explicitly nested diagnostic spans."""
import json,hashlib,sys
from pathlib import Path
from loop081_startup_boundary_admit_run679 import admit
root=Path(sys.argv[1]) if len(sys.argv)>1 else Path('evidence/20260929_loop081_bound/run679/live')
core=[json.loads(x) for x in (root/'boundary/core.jsonl').read_text().splitlines()]
decode={x['req_id']:x for x in core if x['kind']=='socket_add_decoded'}
core_by={}
for x in core:core_by.setdefault((x.get('submit_id'),x['kind']),[]).append(x)
def single(ev,kind,sid):
    v=[x for x in ev if x['kind']==kind and x.get('submit_id')==sid];assert len(v)==1,(kind,sid,len(v));return v[0]['t_ns']*1e-9
def span(ev,begin,end,sid):return single(ev,end,sid)-single(ev,begin,sid)
phases={}
for phase,offset,version in [('off_a',0,1),('on',8,2),('off_b',16,3)]:
    admit(root,phase,offset,version)
    client=json.loads((root/phase/'bench48_1.json').read_text());byclient={x['response_id']:x for x in client['requests']}
    cohorts=[];runtime_intervals=[]
    for c in range(offset+5,offset+9):
        rr=json.loads((root/'runtime'/f'rank0_cohort{c}.json').read_text());ids=set(rr['req_ids'])
        requests=[byclient[decode[r]['external_id']] for r in ids]
        packet=json.loads((root/'boundary'/f'rank0_cohort{c}.json').read_text());ev=packet['events']
        assert packet['time_namespace']==client['time_namespace']
        assert all(x['time_namespace']==client['time_namespace'] for x in core)
        h=[x for x in ev if x['kind']=='handoff_entry'];assert len(h)==1;sid=h[0]['submit_id']
        entries=[x for x in ev if x['kind']=='execute_entry' and ids.intersection({v['req_id'] for v in x.get('new',[])+x.get('cached',[])})]
        assert entries
        points=[('client_start',min(x['start'] for x in requests)),('first_execute',min(x['t_ns'] for x in entries)*1e-9),('handoff',h[0]['t_ns']*1e-9),('runtime_built',single(ev,'runtime_built',sid)),('runtime_begin',single(ev,'runtime_begin',sid)),('runtime_end',single(ev,'runtime_end',sid)),('validation_begin',single(ev,'validation_begin',sid)),('validation_end',single(ev,'validation_end',sid)),('output_ready',single(ev,'output_ready',sid)),('core_result_ready',single(core,'result_ready',sid)),('output_enqueued',single(core,'output_enqueue_end',sid)),('client_complete',max(x['end'] for x in requests))]
        partition={a+'__'+b:y-x for (a,x),(b,y) in zip(points,points[1:])}
        # Any negative interval remains visible; a substantive inversion invalidates the join.
        assert min(partition.values())>=-.002,partition
        assert abs(sum(partition.values())-(points[-1][1]-points[0][1]))<1e-6
        runtime_intervals.append(dict(cohort=c,begin=single(ev,'runtime_begin',sid),end=single(ev,'runtime_end',sid)))
        nested={name:span(ev,begin,end,sid) for name,begin,end in [('certificate','certificate_begin','certificate_end'),('metadata_capture','capture_begin','capture_end'),('validation','validation_begin','validation_end'),('audit_flush','audit_flush_begin','audit_flush_end')]}
        nested['scheduler_update']=span(core,'scheduler_update_begin','scheduler_update_end',sid)
        flush=[json.loads(x) for x in (root/'boundary/rank0_flush.jsonl').read_text().splitlines()];fl=[x for x in flush if x['cohort']==c];assert len(fl)==1
        nested['worker_observer_flush']=(fl[0]['end_ns']-fl[0]['begin_ns'])*1e-9
        details=[]
        for entry in entries:
            q=entry['submit_id'];times={}
            for name,b,e in [('target','target_begin','target_end'),('seed','seed_begin','seed_end')]:
                aa=[x for x in ev if x['kind']==b and x['submit_id']==q];bb=[x for x in ev if x['kind']==e and x['submit_id']==q]
                assert len(aa)==len(bb)
                times[name]=sum((y['t_ns']-x['t_ns'])*1e-9 for x,y in zip(aa,bb))
            details.append(dict(submit_id=q,new=entry.get('new',[]),cached=entry.get('cached',[]),total_scheduled_tokens=entry['total_scheduled_tokens'],host_spans_s=times))
        allrank=[]
        for rank in range(8):
            packet=json.loads((root/'boundary'/f'rank{rank}_cohort{c}.json').read_text());e=packet['events'];assert packet['time_namespace']==client['time_namespace']
            hh=[x for x in e if x['kind']=='handoff_entry'];assert len(hh)==1 and hh[0]['submit_id']==sid
            allrank.append(dict(rank=rank,handoff_s=single(e,'handoff_entry',sid),runtime_begin_s=single(e,'runtime_begin',sid),runtime_end_s=single(e,'runtime_end',sid),output_ready_s=single(e,'output_ready',sid),certificate_s=span(e,'certificate_begin','certificate_end',sid),audit_flush_s=span(e,'audit_flush_begin','audit_flush_end',sid),validation_s=span(e,'validation_begin','validation_end',sid)))
        latest_rank_boundaries={k:max(x[k] for x in allrank) for k in ('handoff_s','runtime_begin_s','runtime_end_s','output_ready_s')}
        cohorts.append(dict(latest_rank_boundaries=latest_rank_boundaries,cohort=c,cycles=rr['cycles'],acceptance=rr['acceptance_window_means'],runtime_record_s=rr['wall_seconds'],host_partition_s=partition,nested_spans_s=nested,ordinary_submits=details,allrank_boundaries=allrank))
    # Exact disjoint Product partition. Cohort request windows may overlap,
    # but the actual rank0 fixed-serving intervals must not overlap.
    start=client['wall_start'];end=client['wall_end'];cursor=start;segments=[]
    for interval in sorted(runtime_intervals,key=lambda v:v['begin']):
        a,b=interval['begin'],interval['end']
        assert start<=a<=b<=end and cursor<=a,(phase,interval,cursor)
        segments.append(dict(kind='outside_runtime',before_cohort=interval['cohort'],begin=cursor,end=a,seconds=a-cursor))
        segments.append(dict(kind='fixed_runtime_host_interval',cohort=interval['cohort'],begin=a,end=b,seconds=b-a))
        cursor=b
    segments.append(dict(kind='outside_runtime_final',begin=cursor,end=end,seconds=end-cursor))
    total=end-start;runtime_total=sum(v['end']-v['begin'] for v in runtime_intervals)
    assert abs(sum(v['seconds'] for v in segments)-total)<1e-6
    assert abs(total-client['summary']['duration_s'])<1e-6
    product=dict(wall_s=total,runtime_host_intervals_s=runtime_total,outside_runtime_remainder_s=total-runtime_total,segments=segments)
    phases[phase]=dict(client_summary=client['summary'],product_disjoint_partition=product,cohorts=cohorts)
assert len(list((root/'runtime').glob('rank*_cohort*.json')))==192
cleanup=dict(x.split('=',1) for x in (root/'cleanup_status.txt').read_text().splitlines());assert set(cleanup.values())=={'0'}
for kind in ('source','scripts'):
    assert (root/(kind+'_before.sha256')).read_bytes()==(root/(kind+'_after.sha256')).read_bytes()
    for line in (root/(kind+'_after.sha256')).read_text().splitlines():
        sha,path=line.split(None,1);assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha,path
out=dict(scope='diagnostic only, no formal promotion',phases=phases,cleanup=cleanup,limits=['Certificate nested inside Runtime; audit flush nested inside validation: do not add nested spans to partition','Output-ready to Core-result-ready includes observer flush, sampling/RPC/serialization/allrank waits; not pure transport','Host spans and rank arrival spread do not measure device service or causally removable time','Each cohort partition has its own client endpoints; cohort intervals can overlap and must not be summed as whole Product wall','Observer flush and instrumentation costs are charged and cannot be subtracted as causal truth','Validation-end to output-ready includes original Runtime JSON serialization/write plus output construction'])
(root.parent/'summary.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
