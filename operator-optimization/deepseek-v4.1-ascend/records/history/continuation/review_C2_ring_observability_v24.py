"""Integer-address and abstract-state analysis only; no kernel implementation."""
from pathlib import Path
from datetime import datetime,timezone
import json

N=Path(__file__).parent;O=N/'profile-first-20261031'
OUT=O/'C2-ring-observability-integer-review-v24.json'

def history_positions(start,used):
    return [p for group in range(start//2,(start+used)//2)
            for p in (2*group,2*group+1) if p<start]

def tail_positions(start,used):
    return list(range(start+used-min(used,32),start+used))

def review_sequence(events):
    full={(page,slot):('initial',page,slot)for page in range(1,4)for slot in range(32)}
    sparse=dict(full)
    reads=updates=0
    for generation,(page,start,used)in enumerate(events):
        before=history_positions(start,used)
        assert all(p%2==0 for p in before)
        for p in before:
            assert full[(page,p%32)]==sparse[(page,p%32)]
            reads+=1
        for p in tail_positions(start,used):
            full[(page,p%32)]=(generation,p)
            if p%2==0:sparse[(page,p%32)]=(generation,p)
        assert all(full[(page,slot)]==sparse[(page,slot)]for page in range(1,4)for slot in range(0,32,2))
        updates+=1
    return reads,updates

def main():
    assert not OUT.exists()
    cases=reads=0
    for start in range(132):
        for used in range(97):
            old=history_positions(start,used)
            assert all(p%2==0 and p%32%2==0 for p in old)
            assert len(old)<=1
            assert len(old)==int(start%2==1 and used>0)
            completed=(start+used)//2-start//2
            endpoints=[p for p in range(start,start+used)if p%2==1]
            assert completed==len(endpoints)
            tail=tail_positions(start,used)
            evens=[p for p in tail if p%2==0]
            assert len(evens)==(start+used+1)//2-(start+used-len(tail)+1)//2
            if len(tail)%2==0:assert len(evens)==len(tail)//2
            cases+=1;reads+=len(old)
    sequences=[]
    sequences.append([(1,6*i,6)for i in range(32)])
    for accepted in range(1,7):
        sequences.append([(1,0,6),(1,accepted,6),(1,accepted+6,6)])
    sequences.append([(1,100,64),(1,133,6),(1,139,6)])
    sequences.append([(1,31,6),(2,0,6),(3,1,6),(1,33,6),(2,1,6),(3,2,6)])
    sequence_totals=[review_sequence(events)for events in sequences]
    # A singleton latest-even retention cannot recover every rollback position.
    written=tail_positions(0,6);latest_even=max(p for p in written if p%2==0)
    required=history_positions(1,1)
    assert latest_even==4 and required==[0] and required[0]!=latest_even
    # State filtering is observational, not whole-buffer byte equality.
    foreign_read=dict(position=1,full_after_update='new-generation-token1',filtered_after_update='old-odd-slot',equal=False)
    u_reads={'K_FP32','S_FP32','metadata'};u_writes={'ring_state'}
    n_reads={'pooled_BF16','norm_weight','epsilon'};n_writes={'latent_BF16','inverse_RMS'}
    assert not u_writes&(n_reads|n_writes) and not n_writes&(u_reads|u_writes)
    sched=json.loads(json.loads((N/'C2-row-domains166-v24.json').read_text())['stdout'])['summary']
    assert sched['conditional_completed_pair_endpoint_rows']==2016 and sched['recorded_physical_C2_norm_rows']==4608
    width=512;pages=17306;bytes_per_written_row=2*width*4
    baseline_rows=2*sched['conditional_completed_pair_endpoint_rows'];filtered_rows=baseline_rows//2
    result=dict(created_utc=datetime.now(timezone.utc).isoformat(),
        scope='Integer address/read-set reasoning and finite abstract-state checks for TOKEN_ALIGNED=True,RATIO=2,CACHE_SIZE=32;no arithmetic kernel/device implementation.',
        source_semantic_proof_steps=[
            'Write s=2q+r with r in{0,1}. Completed group indices are q+j for j>=0.',
            'For j>=1,2(q+j)>s,so neither group member is historical.',
            'For j=0,r=0 has no historical member; r=1 has exactly historical position2q,which is even.',
            'All historical ring value reads therefore use even absolute positions and even slots modulo32.',
            'If initially all even slots agree, keeping identical valid-page/tail/FP32 writes for every even position preserves this invariant after each call.',
            'The invariant supplies identical history values to every later current-helper pool,including rollback calls; unchanged arithmetic and other inputs then produce the same pool outputs in this abstract semantic domain.'],
        conditional_opportunity='Study suppressing odd-position ring maintenance,or a separate16-slot representation,only if no other model/ABI consumer observes odd ring entries and all alias/page/generation contracts are preserved.',
        integer_cases=cases,historical_row_cases=reads,finite_abstract_sequences=len(sequences),finite_abstract_updates=sum(v[1]for v in sequence_totals),finite_abstract_history_reads=sum(v[0]for v in sequence_totals),
        finite_checks_are_not_formal_or_native_validation=True,
        rollback_singleton_retention_counterexample=dict(initial_chunk=[0,6],conditional_next_chunk_start=1,next_used_len=1,required_old_position=required[0],latest_even_position=latest_even,conclusion='Retaining only the latest even token cannot supply older even positions after rollback. Preserve the needed even suffix/generation;this is not an observed acceptance event.'),
        full_state_byte_equality_counterexample=foreign_read,
        pool_update_norm_reorganization=dict(update_reads=sorted(u_reads),update_writes=sorted(u_writes),norm_reads=sorted(n_reads),norm_writes=sorted(n_writes),conditional_dependency_edges=[['pool_reads_old_ring','ring_update'],['pool_output','norm'],['ring_update','future_ring_reader']],
            conclusion='The intermediate ring_update->norm edge is not required by these disjoint abstract read/write sets. Any deferred update still must join before future ring readers and before projected inputs/metadata are reused.',
            preconditions='Source operations have only listed effects;storage sets are disjoint;all aliases,metadata/page generations and asynchronous lifetimes must be bound for native execution.'),
        current_schedule_conditional_accounting=dict(graphs=64,C2_blocks=192,physical_norm_rows=4608,completed_pair_endpoints=2016,endpoint_fraction=0.4375,
            conditional_ring_tail_rows_before=baseline_rows,conditional_even_ring_tail_rows=filtered_rows,bytes_per_ring_row=bytes_per_written_row,
            logical_ring_tail_write_bytes_before=baseline_rows*bytes_per_written_row,logical_even_ring_tail_write_bytes=filtered_rows*bytes_per_written_row,
            qualification='Conditional host-schedule/helper-domain arithmetic across captured rank/graph/source records,not unique user tokens,measured valid device rows,actual HBM traffic or gain.'),
        storage_representation_model=dict(recorded_FP32_shape=[pages,32,2*width],logical_full_bytes=pages*32*2*width*4,conditional16slot_logical_bytes=pages*16*2*width*4,
            qualification='Logical array model only. CircularBufferSpec,page padding,shared cache allocation,packed backend,graph addresses andallreaders mayprevent real allocation savings;not resident memory or a universal lower bound.'),
        remaining=['All ring consumers/aliases andstate-ABI observability','Actual metadata/skip/valid-page/padded-request payloads','Packed/backend/legacy nonTOKEN_ALIGNED domains','Native compiler/purity/lifetime/capture binding','No native fusion implementation or measured gain'],
        no_model_import_or_device_or_service_work=True,overall_complete=False)
    OUT.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(output=str(OUT),integer_cases=cases,abstract_sequences=len(sequences),abstract_updates=result['finite_abstract_updates'],logical_model=result['current_schedule_conditional_accounting'],storage_model=result['storage_representation_model'],overall_complete=False),ensure_ascii=False))

if __name__=='__main__':main()
