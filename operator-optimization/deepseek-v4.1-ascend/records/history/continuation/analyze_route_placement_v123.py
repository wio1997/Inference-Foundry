"""Compose saved token fanout with fixed-slot expert placement, offline only."""
from collections import Counter,defaultdict
from itertools import combinations
from pathlib import Path
import hashlib,json
from finalize_origin_coverage_v9 import N,O,read,write
from route_hypergraph_v123 import records,fit_training,balanced,colocated,score


def pin(p):
    p=Path(p);b=p.read_bytes()
    return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())


def main():
    assert not(O/'REVIEW-CHECKPOINT-v123.json').exists()
    assert not(N/'ssh166-route-v123.sock').exists()
    docs=[N/'route-hypergraph-source-v123/active-routes.bin.gz',N/'route-hyperedges166-v123.json',
        O/'route-hyperedges-source-index-v123.json',O/'route-fanout-saved-records-v122.json',
        N/'MC2-V1-observer-source-v119/decoded-records.jsonl.gz',
        O/'EP-routing-and-placement-contract-v17.json',O/'F21-placement-and-hotspot-variants-v17.json',
        O/'EP-mapping-and-weight-contract-v19.json',O/'F21-interface-branch-ledger-v19.json',
        O/'F21-replica-source-granularity-contract-v19.json',O/'route-fanout-packet-and-return-models-v122.json',
        O/'route-fanout-variants-and-legality-v122.json',O/'route-fanout-local-graph-map-v122.json',N/'route_hypergraph_v123.py']
    write(O/'route-placement-inputs-v123.json',dict(inputs=[pin(p)for p in docs],
        operator_questions=[
            'Does count-balanced expert placement preserve or change token/destination packet fanout?',
            'Can co-selection affinity reduce packet references while worsening receiver assignment load?',
            'How do placement,locality,unchanged expert work/return and metadata-complete packetization compose without double credit?'],
        no_new_device_work=True,no_new_target_source_files=True,overall_complete=False))
    bycohort=defaultdict(list)
    for r in records():
        bycohort[(r['role'],r['layer'],r['observer_ordinal'])].append(r)
    assert len(bycohort)==87 and all(len(c)==16 for c in bycohort.values())
    layers=defaultdict(list)
    for key,c in bycohort.items():
        layers[key[:2]].append((key[2],c))
    variants=['linear','count_balanced','co_selection_capacity','count_balanced_within_TP8','co_selection_within_TP8']
    mappings=[];results=[]
    for (role,layer),samples in sorted(layers.items()):
        samples.sort(key=lambda s:s[0]);training_ordinal,training=samples[0]
        E=training[0]['E_local'];freq,pairs,sender_freq=fit_training(training,E)
        maps={'linear':[e//E for e in range(16*E)],
            'count_balanced':balanced(freq,E),
            'co_selection_capacity':colocated(freq,pairs,sender_freq,E),
            'count_balanced_within_TP8':balanced(freq,E,True),
            'co_selection_within_TP8':colocated(freq,pairs,sender_freq,E,True)}
        for name,owner in maps.items():
            device_experts=[[e for e in range(16*E)if owner[e]==rank]for rank in range(16)]
            physical=[-1]*(16*E)
            for rank,experts in enumerate(device_experts):
                for slot,e in enumerate(experts):
                    physical[e]=rank*E+slot
            assert sorted(physical)==list(range(16*E))
            mappings.append(dict(role=role,layer=layer,variant=name,E_local=E,
                training_ordinal=training_ordinal,training_recorded_rows=sum(len(r['rows'])for r in training),
                trained_on_first_formal_cohort_only=True,owner_by_logical_expert=owner,
                physical_ID_by_logical_expert=physical,device_expert_logical_lists=device_experts,
                same_slot_capacity=True,replicas=0,
                within_TP8_owner_half_invariant=all(owner[e]//8==e//E//8 for e in range(16*E))))
        for ordinal,cohort in samples:
            baseline=score(cohort,maps['linear'])
            for name in variants:
                value=score(cohort,maps[name]);base=baseline['old_format_dispatch_and_return_logical_bytes']
                joint=value['selected_packet_dispatch_and_return_logical_bytes']
                placement_delta=base-value['old_format_dispatch_and_return_logical_bytes']
                packet_after=value['selected_packet_vs_same_mapping_old_format_logical_delta']
                assert base-joint==placement_delta+packet_after
                results.append(dict(role=role,layer=layer,observer_ordinal=ordinal,variant=name,
                    sample_partition='training_first_formal' if ordinal==training_ordinal else 'later_saved_formal',
                    original_linear=baseline,**value,
                    placement_only_logical_byte_delta=placement_delta,
                    packet_after_placement_logical_byte_delta=packet_after,
                    joint_vs_original_linear_old_format_logical_byte_delta=base-joint,
                    naive_independent_delta=placement_delta+baseline['selected_packet_vs_same_mapping_old_format_logical_delta'],
                    count_lower_reference=value['unreplicated_receiver_row_count_lower_reference'],
                    actual_runtime_mapping_or_packet_protocol_changed=False))
    assert len(mappings)==43*5 and len(results)==87*5
    write(O/'route-placement-fixed-mappings-v123.json',dict(mappings=mappings,
        training_policy='Per role/layer first formal saved cohort only;mapping held fixed for later saved cohorts. No later evaluation token/count data used to construct mappings.',
        algorithms=dict(linear='Contiguous recorded baseline.',count_balanced='Descending training expert assignment count;choose lowest loaded rank with fewer than E slots.',
            co_selection_capacity='Greedy pair-affinity clustering into exactly E expert slots;greedy cluster-to-rank labels maximize training local assignment affinity.',
            within_TP8='Restrict expert owner to its original rank-index half of eight. This is a logical partition,not an asserted physical link.'),
        no_global_optimality_or_native_mapping_claim=True,overall_complete=False))
    write(O/'route-placement-cohort-comparisons-v123.json',dict(results=results,
        evaluation_samples='Later saved formal cohorts only are held out. Same archived run/instrumentation,not a separate future production workload.',
        existing_count_placement='v17 already analyzed fixed-count placement. New evidence here is per-token fanout under each selected mapping and its joint packet/return/load consequences.',
        original_native_promotions=0,overall_complete=False))
    aggregates=[]
    for role in ['target','draft']:
        for partition in ['training_first_formal','later_saved_formal']:
            for name in variants:
                rows=[r for r in results if (r['role'],r['sample_partition'],r['variant'])==(role,partition,name)]
                agg=dict(role=role,sample_partition=partition,variant=name,cohorts=len(rows))
                for k in ['active_recorded_rows','assignments_all','token_destinations_all','A_remote','D_remote',
                    'cross_TP8_assignment_references','cross_TP8_token_destination_references',
                    'within_destination_expert_pair_score','old_format_dispatch_and_return_logical_bytes',
                    'selected_packet_dispatch_and_return_logical_bytes','placement_only_logical_byte_delta',
                    'packet_after_placement_logical_byte_delta','joint_vs_original_linear_old_format_logical_byte_delta','naive_independent_delta']:
                    agg[k]=sum(r[k]for r in rows)
                agg['sample_max_receiver_assignment_rows']=max(r['max_receiver_assignment_rows']for r in rows)
                for state,test in [('improved_receiver_max',lambda a,b:a<b),('same_receiver_max',lambda a,b:a==b),('worse_receiver_max',lambda a,b:a>b)]:
                    agg[state]=sum(test(r['max_receiver_assignment_rows'],r['original_linear']['max_receiver_assignment_rows'])for r in rows)
                agg['fewer_D_remote_cohorts']=sum(r['D_remote']<r['original_linear']['D_remote']for r in rows)
                agg['more_D_remote_cohorts']=sum(r['D_remote']>r['original_linear']['D_remote']for r in rows)
                agg['naive_addition_wrong_cohorts']=sum(r['joint_vs_original_linear_old_format_logical_byte_delta']!=r['naive_independent_delta']for r in rows)
                aggregates.append(agg)
    write(O/'route-placement-aggregate-tradeoffs-v123.json',dict(aggregates=aggregates,
        no_runtime_speedup_or_physical_bytes=True,overall_complete=False))
    write(O/'route-placement-nonadditivity-and-surrogate-v123.json',dict(
        exact_identity='Delta_joint=(A_linear-A_pi)*(5152+10240)+(A_pi*5152-D_pi*5216-extra_D_counts). Recompute packet work after placement;do not add the linear-packet saving independently.',
        return_contract='Logical expert assignment count and GMM work are unchanged. Remote return count A_pi can change with placement/locality;same-mapping packetization alone retains return.',
        pair_surrogate_counterexample=dict(K=6,destination_occupancies_a=[3,1,1,1],destination_occupancies_b=[2,2,2],
            pair_scores=[3,3],all_destination_packet_counts=[4,3],saved_duplicate_reference_counts=[2,3],
            conclusion='Pair co-selection affinity is a surrogate. Equal pair scores do not determine exact token/destination packet counts for K>=3;use complete saved hyperedges to score.'),
        theory_state='SYMBOLIC_CEILING',physical_T_lower_or_throughput_upper_numeric=None,
        same_capacity_does_not_mean_same_work_or_critical_path=True,overall_complete=False))
    graph=read(O/'route-fanout-local-graph-map-v122.json')
    new_nodes=['original_logical_topk_ids','chosen_static_map_generation','matching_weight_scale_banks','logical_to_physical_remap']
    graph_nodes=graph['new_nodes']+new_nodes
    graph_edges=graph['edges']+[dict(a=a,b=b,contract=c)for a,b,c in [
        ('original_logical_topk_ids','logical_to_physical_remap','Original logical TopK slots preserved'),
        ('chosen_static_map_generation','logical_to_physical_remap','Bijection and its generation'),
        ('logical_to_physical_remap','effective_native_IDs','Physical expert IDs;retain actual active-mask semantics'),
        ('matching_weight_scale_banks','GMM1_SwiGLU','Matching logical expert function and quantized scales under selected map generation'),
        ('matching_weight_scale_banks','GMM2_return','Matching second projection weights/scales and rounding')]]
    write(O/'route-placement-legality-and-graph-v123.json',dict(
        existing_static_custom_map_source='v19 already documents expert_map_path/log2phy. No wholly absent static mapping capability is invented. Post-load weight-map reconciliation remains unbound.',
        selected_domain='Bijective expert relocation only:16 ranks,exactly E original expert functions per rank,no replicas,no change of logical TopK order/probabilities or assigned function.',
        candidate_graph=['original logical TopK and probabilities','fixed per-layer mapping generation','paired weights/all FP and packed scales','physical-ID remap with original TopK-slot identity',
            'new token/destination grouping and original assignment-map descriptors','selected packet/count protocol','receiver original assignment expansion or alternative indirect iterator',
            'expert GMM and all original rounding','one original result per logical assignment','inverse mapping and original weighted-unpermute order'],
        local_joint_graph=dict(id='MC2_STATIC_PLACEMENT_PACKET_V123',nodes=graph_nodes,edges=graph_edges,
            parent='MC2_TOKEN_DEST_DISPATCH_V122',state='CONDITIONAL_CANDIDATE_NOT_OBSERVED_RUNTIME',
            original_mapping='Original router logical result remains;only physical IDs and paired expert owners change. New receiver-row/descriptor translations must retain original logical assignment identities.',
            readiness='Configuration agreement and required expert/tile data readiness,not an invented full-layer/full-weight-copy barrier before every remap.'),
        lifetime='All ranks must agree on map/weight/scale/capture generations;old queued dispatch/remote pulls/GMM/return readers retire before reuse. New/old weights/maps/packet stages can overlap during transition;fixed steady slot counts do not prove allocation or migration fits.',
        logical_equivalence='For bijection pi, set physical expert function F_physical[pi(e)]=F_logical[e], preserve original TopK slot k and its probability p_k, and send slot k to pi(e_k). Each returned slot then denotes the same expert function. Ordered weighted combine is algebraically identical. Permuting IDs without paired weights/scales changes the function.',
        numerics='ID relabeling is an algebraic logical-function equivalence only. Physical sort/order,receiver capacities,GMM row order/tiling,quantizer partition,rounding and combine order must preserve the declared numeric contract. No native bit identity is proved.',
        capacity='Check actual receiving A counts and original clipping policy per mapping;all saved selected mapping counts stay below recorded C65536 only if verified,not because E slots are unchanged.',
        symbolic_bound='After necessary actual physical work/cuts and typed capacity upper bounds are established,T>=max_r(W_r/C_r_upper). Packet and row count ratios are not speedup or MFU.',
        no_new_global_observed_graph_nodes=True,new_native_promotions=0,new_device_work=False,
        implementation_or_deployment=False,overall_complete=False))
    print(json.dumps(dict(status='ANALYZED',mappings=len(mappings),cohort_variant_comparisons=len(results),
        training_layer_domains=len(layers),new_device_work=False,
        heldout=[{k:r[k]for k in ['role','variant','cohorts','D_remote','sample_max_receiver_assignment_rows',
            'improved_receiver_max','worse_receiver_max','joint_vs_original_linear_old_format_logical_byte_delta']}
            for r in aggregates if r['sample_partition']=='later_saved_formal']),ensure_ascii=False))


if __name__=='__main__':
    main()
