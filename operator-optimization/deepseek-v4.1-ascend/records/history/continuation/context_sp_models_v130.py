"""Conditional BF16-512 carrier models; no physical traffic or timing model."""
from itertools import combinations


def carrier_case(C,Q,mq,s,P=8,H=5120,D=512,layers=3,main_bytes=None):
    assert 0<=s<=C and 0<=Q<=P*mq and P>0
    owned=(s+P-1)//P
    base_bytes=2*H*mq;new_bytes=2*D*(mq+owned)
    base_rows=C+Q;new_rows=C-s+owned+mq
    return dict(distributed_context_rows=s,context_owner_rows_padded_per_sender=owned,
        query_projection_padding_rows_over_all_senders=P*mq-Q,
        context_projection_padding_rows_over_all_senders=P*owned-s,
        baseline_query_gather_sender_content_bytes_per_layer=base_bytes,
        new_KV_carrier_sender_content_bytes_per_layer=new_bytes,
        baseline_projection_rows_per_card_per_layer=base_rows,
        new_projection_rows_per_card_per_layer=new_rows,
        baseline_dense_projection_FLOPs_per_card=layers*2*base_rows*H*D,
        new_dense_projection_FLOPs_per_card=layers*2*new_rows*H*D,
        retained_main_feature_gather_sender_content_bytes=main_bytes,
        baseline_combined_sender_content_bytes=main_bytes+layers*base_bytes if main_bytes is not None else None,
        new_combined_sender_content_bytes=main_bytes+layers*new_bytes if main_bytes is not None else None,
        carrier_stage_content_not_above_baseline=new_bytes<=base_bytes,
        logical_ABI_content_not_network_wire_or_latency=True)


def pareto_cases(C,Q,mq,P=8,main_bytes=None):
    allcases=[carrier_case(C,Q,mq,s,P=P,main_bytes=main_bytes)for s in range(C+1)]
    best_by_content={}
    for c in allcases:
        b=c['new_KV_carrier_sender_content_bytes_per_layer'];w=c['new_projection_rows_per_card_per_layer']
        old=best_by_content.get(b)
        if old is None or w<old['new_projection_rows_per_card_per_layer']:best_by_content[b]=c
    frontier=[];best=None
    for b,c in sorted(best_by_content.items()):
        w=c['new_projection_rows_per_card_per_layer']
        if best is None or w<best:frontier.append(c);best=w
    # Every option is dominated or equalled in this two-count model.
    for c in allcases:
        b=c['new_KV_carrier_sender_content_bytes_per_layer'];w=c['new_projection_rows_per_card_per_layer']
        assert any(f['new_KV_carrier_sender_content_bytes_per_layer']<=b and f['new_projection_rows_per_card_per_layer']<=w for f in frontier)
        assert c['carrier_stage_content_not_above_baseline']==((c['distributed_context_rows']+P-1)//P<=9*mq)
    return frontier


def recipient_cases(Pmax=8):
    cases=[]
    for P in range(1,Pmax+1):
        ranks=range(P)
        for nr in range(1,P+1):
            for need in combinations(ranks,nr):
                for nc in range(1,nr+1):
                    for compute in combinations(need,nc):
                        assignment={r:r if r in compute else compute[0]for r in need}
                        deliveries=[(c,r)for r,c in assignment.items()if c!=r]
                        assert len(deliveries)==nr-nc and set(assignment)==set(need)
                        cases.append(dict(P=P,recipient_ranks=list(need),compute_ranks=list(compute),
                            context_row_projection_copies=nc,endpoint_KV_row_deliveries=len(deliveries),
                            endpoint_value_content_bytes=1024*len(deliveries),
                            raw_context_already_available_at_compute_owners_required=True))
    return cases


def writer_witness():
    # Exactly one context and one query update to the same address, across two
    # ordered source calls. No within-call duplicate-scatter semantics assumed.
    original=[('C',1,7,11),('Q',0,7,29)]
    packed_rank_major=sorted(original,key=lambda x:x[1])
    def apply(rows):
        cache={}
        for role,rank,slot,value in rows:cache[slot]=value
        return cache
    assert apply(original)=={7:29}and apply(packed_rank_major)=={7:11}
    role_order=sorted(packed_rank_major,key=lambda x:0 if x[0]=='C'else 1)
    assert apply(role_order)==apply(original)
    return dict(original_C_then_Q=original,rank_major_rows=packed_rank_major,
        source_final=apply(original),unsafe_serialized_packed_final=apply(packed_rank_major),
        role_ordered_final=apply(role_order),native_race_or_actual_slot_values_proved=False)
