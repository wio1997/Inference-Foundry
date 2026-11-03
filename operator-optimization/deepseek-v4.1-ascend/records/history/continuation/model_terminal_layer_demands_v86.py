"""Conditional owner envelopes and typed dense-algorithm work; not measured speedup."""
from math import ceil
from finalize_origin_coverage_v9 import O,read,write

def quota(N,U,P=8,H=5120):
    assert N%P==0 and 0<=U<=N
    M=N//P;qlo=ceil(U/P);qhi=min(U,M)
    return dict(N=N,U=U,P=P,M=M,minimum_feasible_owner_max=qlo,maximum_feasible_owner_max=qhi,
      equal_count_published_row_slots_min=P*qlo,equal_count_published_row_slots_max=P*qhi,
      per_rank_ring_send_value_bytes_min=2*(P-1)*qlo*H,
      per_rank_ring_send_value_bytes_max=2*(P-1)*qhi*H,
      baseline_localized_dense_ring_send_value_bytes=2*(P-1)*M*H,
      ideal_variable_count_group_total_send_value_bytes=2*(P-1)*U*H,
      padded_equal_count_group_total_send_value_bytes_min=2*P*(P-1)*qlo*H,
      padded_equal_count_group_total_send_value_bytes_max=2*P*(P-1)*qhi*H,
      maximum_quota_zero_payload_reduction_possible=(qhi==M),
      actual_U_or_owner_distribution_or_latency_attested=False)

def main():
    assert not(O/'terminal-layer-owner-and-work-models-v86.json').exists()
    profile=read(O/'terminal-layer-native-frontier-v86.json');models=[]
    for regime in profile['target_head_row_regimes']:
        n,k=regime['normalized_rows'],regime['head_rows'];H=5120;p=8;u=k
        q=quota(n,u);q.update(head_K=k,recorded_root_count=regime['roots'],
          scenario='Illustration ONLY: U=distinct(I)=K, no additional J; runtime I,J and U not attested. Prompt/full-output observers may force U=N.',
          group_original_collapse_row_evaluations=p*n,group_owner_selected_collapse_row_evaluations=u,
          group_receiver_selected_Norm_row_evaluations=p*u,group_owner_selected_Norm_row_evaluations=u,
          original_head_GEMM_rows_per_rank_kept=k,
          conditional_localized_merge_Post_C_Norm_private_interface_union_group_bytes=96*u*H,
          source_group_dense_BF16_KV_projection_ops_kept=2*p*n*H*512,
          selected_source_group_Q_A_BF16_ops=2*u*H*1280,
          selected_source_group_Q_B_INT8_ops=2*u*1280*32768,
          selected_TP_group_WO_A_BF16_ops=2*p*u*4096*1024,
          selected_TP_group_WO_B_BF16_ops=2*p*u*1024*H,
          selected_source_group_router_FLOAT_ops=2*u*H*384,
          selected_source_group_shared_INT8_dense_ops=2*u*H*(4608+2304),
          selected_source_group_routed_INT8_dense_ops_reference=2*6*u*H*(4608+2304),
          routed_assignments_from_this_source_group_reference=6*u,
          shared_input_DQ_row_scales=u,Q_B_input_DQ_row_scales=u,
          actual_MC2_receiver_rows_or_active_experts_or_startup_or_gain=None)
        models.append(q)
    assert len(models)==5 and sum(m['recorded_root_count']for m in models)==88
    # Extremal examples retain distinct global row identity; actual runtime owner counts remain unknown.
    examples=[quota(24,u)for u in[0,1,3,8,12,24]]
    for e in examples:
        u=e['U'];m=e['M'];p=e['P'];balanced=[u//p+(i<u%p)for i in range(p)]
        concentrated=[];remaining=u
        for i in range(p):value=min(m,remaining);concentrated.append(value);remaining-=value
        assert sum(balanced)==sum(concentrated)==u and max(balanced)<=m
        assert max(balanced)==e['minimum_feasible_owner_max']and max(concentrated)==e['maximum_feasible_owner_max']
        e.update(balanced_owner_counts=balanced,concentrated_owner_counts=concentrated)
    write(O/'terminal-layer-owner-and-work-models-v86.json',dict(models=models,small_owner_examples=examples,
      owner_envelope_proof='Given U distinct rows in P disjoint owner ranges of capacity M=N/P: sum(u_p)=U,0<=u_p<=M, so ceil(U/P)<=q=max(u_p)<=min(U,M). Balanced and concentrated constructions attain the two endpoints. Equal-count AllGather sends q slots/rank, not U/P unless exactly balanced.',
      source_of_rule='v65 already recorded8max(U_p). Here its endpoint envelopes, concrete recorded-shape scenarios and joint stage costs are added; no new row-selector readiness proof.',
      variable_count_reference='If every rank must receive arbitrary normalized BF16 values generated only at row owners, point-to-point replication has aggregate(P-1)*2UH value delivery. Equal-count ring reference pads toPq and costs(P-1)*2PqH. This fixed value-transport domain excludes recomputation/other layouts and metadata; not a universal physical topology or latency bound.',
      ordered_rows='Compute unique union U once only if same parameter/numerical generation; restore each ordered I/J consumer including duplicates and pad rows. Optional head index padding can duplicate index0. K alone does not prove U=K or U=N when K=N.',
      compute_domain='Conventional dense dot products, MAC=2ops, preserved per-row quantization and BF16 barriers. Counts are conditional algorithm work references, not all-algorithm universal lower bounds; HcPre HF32, attention selected-key work, quant/scalar processing, metadata and cache writes are separate.',
      projection_domain='Existing TP8 group ownership: Q_A/Q_B source-owner rows total U; WO_A/WO_B each of8 weight owners evaluates Uglobal rows. Exact CP/SP mapping and original query metadata must be supplied. Other ownership variants have different work/weights.',
      routed_domain='6assignments per retained source row is conventional current target topK reference. Source TP8 groups feed EP16 receivers; all source groups and actual expert maps must be accumulated to price each receiver. Dispatch padding, internal quant/combine and EPLB/state observations cannot be inferred as6U per device.',
      retained_KV_domain='Existing full-N KV projection on each TP rank retained as a baseline branch. This is NOT a universal necessary replicated work bound: F32/F33 may move projection before gather under their own contract. Query pruning alone earns no KV projection/state-write credit.',
      conditional_resource_lower_bound='Only with necessary work in the declared implementation domain and valid capacity uppers: T>=max_p{sum_d lambda_d*W_p,d/beta_Cube,p(lambda),Q_GMread,p/Bread,p,Q_GMwrite,p/Bwrite,p,Q_linkcut/Bcut,L_true_dependencies}. Account retained full-KV/HcPre and receiver expert work. BF16/INT8/FLOAT/HF32 counts are not interchangeable capacities.',
      actual_capacities_or_numerical_new_shape_parity_or_speedup_proved=False,overall_complete=False))
    weights=dict(Q_A_BF16_bytes=2*5120*1280,Q_B_INT8_raw_bytes=1280*32768,
      WO_A_local_G1_BF16_bytes=2*4096*1024,WO_B_BF16_bytes=2*1024*5120,
      router_FLOAT_bytes=4*5120*384,shared_gate_up_INT8_raw_bytes=5120*4608,
      shared_down_INT8_raw_bytes=2304*5120,routed_one_expert_INT8_raw_bytes=5120*(4608+2304))
    write(O/'terminal-layer-weight-and-dependency-proof-v86.json',dict(weight_planes=weights,
      cold_plane_gate='Any positive selected-row GEMM can still traverse its entire conventional uncompressed weight plane; row reduction alone does not proportionally reduce weight service. Quant scales/metadata, cache residency and packing are additional domains.',
      routed_active_expert_gate='Active expert union depends on exact retained router outputs. Under6distinct experts/row, union size is6..min(384,6U) forU>0, not automatically6U; actual receiver placement and cached weights remain unknown.',
      dependency_cut='A last-layer output can depend only on selected query results while last-layer KV/cache publication still demands all required scheduled rows of the preceding layer. Thus output-only demand cannot justify deleting previous-layer FFN/attention rows needed by that KV branch.',
      fusion_vs_deletion='It can remain possible to fuse preceding-layer output directly into full-demand KV/HcPre consumers and avoid an interface plane. That changes materialization, not the set of necessary values or preceding-layer arithmetic.',
      state_guards=['Keep all3pre-layer aux means/context consumers; they are not final normalized rows.',
        'Preserve raw snapshot/output external and registered observers and capture aliases/lifetime.',
        'Keep original global row/request/position/query length/causal/sink/cache-slot maps; compact ordinals are not positions.',
        'Keep all future KV/compressor/index state writers,accepted/rejected rollback and publication epochs.',
        'EPLB load/heat/counter/weight updates may observe omitted routing/receiver work. Extra guards or retained side-effect computations are required.',
        'Empty query owners must still participate in required cache writers and rank-uniform collectives. Unknown metadata/zero/pad outputs are not free.'],
      no_new_consumer_closure=True,implementation=False,native_test=False,overall_complete=False))
    print(dict(models=5,root_count=88,small_owner_examples=6,new_rule=False,new_conditional_stage_cut=True,actual_speedup=False))

if __name__=='__main__':main()
