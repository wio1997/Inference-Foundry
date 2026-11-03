"""Build an explicit, incomplete candidate relation ledger and joint cost witnesses.

No discovery by name, no inference that unreviewed pairs are independent, no target work.
"""
from collections import Counter
from itertools import combinations
from pathlib import Path
import hashlib
from finalize_origin_coverage_v9 import N, O, read, write


def main():
    assert not (O / 'candidate-relation-coverage-v73.json').exists()
    previous = read(O / 'candidate-profile-ledger-v72.json')
    candidates = {c['id']:c for c in previous['candidates']}
    assert len(candidates) == 35
    # Every clause below is attached to an existing operator contract, not inferred
    # from two candidate titles sharing a word or a tensor shape.
    refs = {
        'projection':'projection-fusion-variants-and-interactions-v61.json',
        'HC':'HC-fusion-split-reorganization-variants-v59.json',
        'shared':'shared-expert-fusion-variants-and-interactions-v57.json',
        'state':'state-engram-fusion-variants-v62.json',
        'router':'router-vector-fusion-variants-v63.json',
        'heads':'heads-sampling-fusion-variants-v65.json',
        'comm':'communication-fusion-variants-v66.json',
        'init':'embedding-initialization-operator-contract-v67.json',
        'pad':'pad-fusion-variants-v72.json',
        'plan':'planner-source-algorithm-contract-v71.json',
        'plan_options':'planner-fusion-variants-v71.json',
    }
    source_index = []
    for tag, fn in refs.items():
        p = O/fn
        d = read(p)
        assert d
        source_index.append(dict(tag=tag,path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),scope='Existing saved operator facts; no new framework/source exploration.'))
    clauses = []
    def edge(a,b,kind,statement,tags,scope):
        assert a in candidates and b in candidates and a != b
        a,b = sorted([a,b])
        clauses.append(dict(id=f'REL73-{len(clauses)+1:02d}',a=a,b=b,relation=kind,operator_claim=statement,
                            evidence=[refs[t] for t in tags],applicable_domain=scope,
                            conditional_compatibility=True,all_runtime_gates_proved=False,
                            speedup_additive=False,critical_path_gain_us=None))
    edge('F02','F03','COMPOSE_WITH_LIVE_RESIDUAL_FANOUT','Next collapse/norm may consume rounded HcPost tiles, but current coefficient, future post, aux/Engram and tail readers keep residual materialization live. Whole4MhH removal is not implied.',['HC'],'Current rounded BF16 HcPost→next delayed collapse; all readers/row generation must match.')
    edge('F02','F04','SHARED_COLLAPSE_NORM_EDGE','Y collapse+RmsNormCast can remove one4MH Y interface; the FFN rounded BF16 plane and widened FLOAT semantics remain. Splitting Y/coefficient branch alone does not remove arithmetic.',['HC','router'],'Same FFN row/Y generation, all BF16/FLOAT/rstd observers retained.')
    edge('F03','F23','OVERLAPPING_MERGE_INTERFACE','SingletonAdd, down-projection Add epilogue, and merge into HcPost are alternative cuts through the same merge. Preserve Down BF16 then merge BF16 before FP32 HC arithmetic.',['HC','shared'],'Shared+routed merge at a specific layer and row generation; not every Post boundary.')
    edge('F02','F28','COMPOSE_ONLY_WITH_UNCHANGED_HC_DOMAIN','Direct local HC4 initialization preserves copy values but does not remove full4H HC coefficient normalization/projection. Initialization and first-pre producer fusion share the same initialized plane and need a joint all-reader cost.',['HC','init','pad'],'Draft initialization only; HcPre full-row numerics/parameter generation retained.')
    edge('F04','F20','COMPOSE_WITH_ROUNDED_BF16_FANOUT','Router widening and full-E router epilogue may remove distinct FLOAT/logit interfaces; shared/routed experts still need rounded BF16 normalized values. Do not delete all normalized storage.',['router','shared'],'Current FLOAT router branch; native nonlinear/ranking/tie/bias/weight and all reader contracts.')
    edge('F04','F22','SHARED_INPUT_REPRESENTATION_GATE','Shared quantization consumes rounded normalized BF16. Removing the router FLOAT plane is compatible only if this BF16 consumer remains; changing norm precision or reusing pre-round values is not implied.',['HC','router','shared'],'Identical norm generation and shared input row map; no private quantizer identity inferred.')
    edge('F20','F21','PAD_ABI_AND_EXISTING_FUSION_GATE','Full constant-pad replacement may remove private helpers; virtual padding/row pruning additionally changes MC2 dispatch counts/workspace/mask ABI. Existing dispatch/FFN/combine fusion is the baseline.',['router','pad','shared'],'Observed800→816 prepare domain; unchanged MC2 EP/count protocol unless explicitly replaced.')
    edge('F21','F22','COMMON_QUANTIZER_REQUIRES_NEW_ABI','Shared/routed common INT8 producer needs same rounded input, row map, mask, scales, quantizer and lifetime. Current MC2 BF16 input is not already the common prequantized ABI.',['shared'],'Same input generation only; identical dtype is insufficient.')
    edge('F22','F23','READINESS_BUFFER_AND_ROUNDING_TRADEOFF','DownAdd may extend the shared branch producer through merge; it must wait for routed tiles, retain both BF16 rounds and account for changed overlap/buffering. Generic Add removal is counted once.',['shared','HC'],'Matching rows/current scaling and full fanout; finite Cube/Vector/HBM competition.')
    edge('F05','F09','OVERLAPPING_SHARED_QR_PRODUCER','QA norm/quant and Q_B/IndexQ sharing refer to one QR quantized plane. Count its write/read once; distinct QB/IndexQ weights, outputs and row scales remain.',['projection'],'Same QR generation and exact quantizer; M3 interface examples are not whole-request savings.')
    edge('F06','F09','JOINT_PROJECTION_AND_QUANT_REUSE','Q_B+IndexQ input reuse and joint output tiling can compose only with distinct output/scales/packing. The15384B example already includes the11532B quant-producer budget.',['projection'],'Declared M3 QR1280, QB32768 and IndexQ4096; packed stride/layout support conditional.')
    edge('F06','F10','COMPOSE_WITH_SEPARATE_NORM_DOMAIN','QA+IndexWeight joint input must keep separate1280 and32 outputs. QA RMS cannot become RMS over1312 concatenated values.',['projection'],'Same query hidden generation; preserve weight BF16→FP32 scale→FP16 rounding.')
    edge('F06','F32','ROW_DOMAIN_AND_PARTITION_DEPENDENCY','Local QA and global KV cannot be concatenated at identical M by shape convenience. Project-before-gather requires role-specific projected planes and actual ownership/readiness.',['projection','comm'],'Current QA3 vs KV24 example; global fallback/local-row repartition distinct.')
    edge('F08','F32','COMPOSE_ONLY_AFTER_GLOBAL_ROW_READY','KV projection/norm/RoPE/cache fusion can stream complete incoming rows or move compatible projections before gather. Global cache publication/full512 norm and reader/slot ownership remain.',['projection','comm'],'Current BF16 round boundaries and row-generations; no free communication removal.')
    edge('F11','F19','PRE_ROPE_LATENT_FORK_WAR','IndexKey projection reads pre-RoPE compressor latent; fusing its long-KV norm/RoPE/cache path must retain the fork and finish old latent readers before in-place stores.',['projection','state'],'Compatible C1/C2 long-cache branch; key scale and physical slot publication coherent.')
    edge('F17','F32','PROJECTED_PLANE_COMMUNICATION_COST','C2 projected-row exchange may share local input cast/tiles, but both FLOAT WKV/WGATE outputs remain distinct and global pool state cannot be replaced by independent local pool.',['projection','state','comm'],'C2 direct dense FLOAT arithmetic; local/global row plan and input feature quantization fixed.')
    edge('F17','F18','ROW_POOL_NORM_DEPENDENCY','Joint WKV/WGATE projections feed pool then BF16 round/norm; separate output fields and pooling max/exp/sum order remain. Joint producer savings do not erase required pooled output work.',['projection','state'],'C2 pool domain and FLOAT parameter arithmetic; no reassociation across nonlinearity.')
    edge('F18','F19','POOL_NORM_FORK_INTERFACE','Pool+norm and norm/RoPE/cache fusion overlap the normalized latent interface; pre-RoPE IndexKey readers prevent unconditional in-place deletion or assigning the same norm interface saving twice.',['state'],'BF16 pool/norm boundaries, all slot masks and latent readers retained.')
    edge('F18','F34','FORK_OR_FUSE_WITH_OLD_STATE_BARRIER','Pool may fork norm and ring maintenance; one pool/update kernel needs all old observable ring reads before overwrite, physical page ownership and cross-program synchronization. Even-only maintenance is not half current projections or half unchanged allocation.',['state'],'Ratio2 observable ring domain, rollback/generation/all future observers retained.')
    edge('F24','F25','PARTITION_ALTERNATIVES_AND_DEDUP_COMPOSITION','F24 owner/dedup/cache reduces lookup work jointly on Uowned. Row-exchange WKV and head-partial WKV are alternative partitions; compare full downstream gate/norm/partial communication, not two independent TP factors.',['state','comm'],'Active table format/version, head/shard/owner and post-lookup exact numerics required.')
    edge('F26','F35','OVERLAPPING_HEAD_ROW_SELECTION','Tail row selection and lazy target-head/rejection pruning must use the union of all observer demands and one baseline. The native head already has row selection; removing norm/head rows cannot be counted twice.',['heads','comm'],'Target tail/head only; raw/probability/entropy/recovery/logprob/all future observers included.')
    edge('F30','F35','DISTINCT_TARGET_AND_DRAFT_RECURRENCES','Draft Markov serial token recurrence and target rejection/head rows are different computations. A confidence-derived budget must be known before a pruned prefix; it does not make noncausal decoder queries or raw hidden consumers dead.',['heads'],'Frozen five-step Markov and target rejection contracts; RNG/global IDs/readiness preserved.')
    edge('F14','F15','PACK_INTERFACE_CHANGES_WITH_PARTITION','Direct attention send-layout writes may remove pack only for the chosen restore protocol. Moving WA before A2A changes the transmitted representation and weights; cannot additionally count elimination of the old pack twice.',['projection','comm'],'BF16 attention/inverse-RoPE boundaries and actual rank/head/row layout.')
    edge('F15','F16','NESTED_PARTITION_ALTERNATIVES','WA-before-A2A and token-owner WA+WB are different whole output partitions; full owner extends compute/weight replication and replaces communication. Count chosen partition payload and weights, not sum both communication savings.',['comm'],'Current M=A condition; native per-group BF16 rounds and reduction policy retained.')
    edge('F16','F03','COMPOSE_WITH_ONE_PUBLISHED_PARTIAL','WO_B→RS and RS→HCPost can tile pipeline, but each partial must undergo native BF16 reduction before HCPost. Same output/consumer buffer interface is removed once and both Cube stages share capacity.',['projection','comm','HC'],'Exact TP group, row padding, native reduction order/all reader frontier.')
    edge('F31','F35','FULLWRITE_ACCEPT_STATE_FRONTIER','Rejection full-write/count epilogue can feed accepted-state update. Preserve literal predicates/INT64 count vs INT32 destination and preload old values before in-place gather-map stores.',['heads'],'Same B2/4 snapshot/generation and mixed-greedy/RNG ownership, not arbitrary per-row concurrent update.')
    edge('F13','F31','PLAN_REUSE_DOES_NOT_IMPLY_ATTENTION_VALUE_REUSE','Equal normalized planner keys can reuse a scheduling plan; those fields do not encode Q/K/V contents. New KV/query values still require attention. Cached plan identity is not activation identity.',['plan','plan_options'],'SWA saturated effective lengths/QSL/attrs fixed; actual cache bank/current token values may change.')
    edge('F12','F31','PLAN_METADATA_NOT_SCORE_OR_INDEX_IDENTITY','A schedule key based on lengths/QSL/attrs does not establish equal index scores or selected indices. Reusing topK/indices needs their own source-value/key/generation and all-reader proof.',['plan','projection'],'Planner-defined fields only; no pointer/shape equality used for value CSE.')
    selected_pairs = {tuple(sorted([r['a'],r['b']])) for r in clauses}
    assert len(selected_pairs) == len(clauses) == 28
    touched = set(x for p in selected_pairs for x in p)
    pairs = []
    for a,b in combinations(sorted(candidates),2):
        selected = [r['id'] for r in clauses if (r['a'],r['b']) == (a,b)]
        pairs.append(dict(a=a,b=b,scope_state='SCOPED_OPERATOR_RELATION_REVIEWED_CONDITIONAL' if selected else 'NOT_YET_RECONCILED_HERE;PRIOR_CLAUSES_MAY_EXIST;NO_INDEPENDENCE_CONCLUSION',clauses=selected,whole_pair_all_variants_closed=False,new_critical_path_gain_us=None))
    assert len(pairs) == 595
    # Higher-order composition is explicit and incomplete, not inferred from a clique.
    groups = [
        dict(id='HG73-MOE',members=['F04','F20','F21','F22','F23','F03'],contract='Keep rounded BF16 shared/routed plane, separate FLOAT router/logit/removable interfaces; choose native/virtual padding ABI and quantizer before cost; preserve both merge rounds/HC consumers.'),
        dict(id='HG73-Q',members=['F05','F06','F09','F10','F32'],contract='One QR quant producer, distinct query/globalKV domains and QA1280 norm. Joint packed layouts and projected-feature exchange must use the same role/row plan.'),
        dict(id='HG73-C2',members=['F17','F18','F19','F34','F32'],contract='Distinct FLOAT projection fields→global row pool→rounded BF16 norm→preRoPE IndexKey fork/long-KV publication; old-ring WAR and observable-history retention survive new partitions.'),
        dict(id='HG73-OUTPUT',members=['F14','F15','F16','F03'],contract='Select one send/compute partition, preserve attention/WA/WB/reduction BF16 boundaries, price selected weights and all Cube/network/Vector resources jointly.'),
        dict(id='HG73-TAIL',members=['F26','F30','F31','F35'],contract='Target observer row-demand closure and separate draft recurrence, rawhidden/confidence, global best/RNG and accepted-state snapshot publication. A smaller row set is not blanket decoder liveness.'),
    ]
    for g in groups:
        g.update(scope_state='SCOPED_COMPOSITION_CONSTRAINTS_RECORDED;ALL_VARIANTS_AND_HIGHER_ORDER_CLOSURE_OPEN',all_subsets_analyzed=False,legal_joint_fusion_proved=False)
    # Exact joint accounting in the declared INT8 lookup reference model.
    A, Ao, Uo, Ug = 96,12,5,40
    lookup = dict(A=A,A_owned=Ao,U_owned=Uo,U_global=Ug,bytes_per_INT8_decoded_row_reference=288,
                  baseline_bytes=288*A,owner_only_bytes=288*Ao,dedup_only_bytes=288*Ug,joint_bytes=288*Uo,
                  owner_only_saved=288*(A-Ao),dedup_only_saved=288*(A-Ug),joint_saved=288*(A-Uo),
                  invalid_sum_of_standalone_saved=288*(A-Ao)+288*(A-Ug),
                  valid_ordered_decomposition=[288*(A-Ao),288*(Ao-Uo)],
                  physical_host_or_UVA_traffic_proved=False,actual_A_or_U_histogram_proved=False,
                  scope='Constructed feasible owner/ID population, not measured hit ratio.288=256 INT8 data+8 FP32 scales; actual format/cache/page traffic is conditional.')
    assert sum(lookup['valid_ordered_decomposition']) == lookup['joint_saved']
    assert lookup['invalid_sum_of_standalone_saved'] > lookup['baseline_bytes']
    # Feasibility witness: 12 owned occurrences with5 distinct IDs,84 unowned with35.
    owned = list(range(5))+[0]*7
    unowned = list(range(5,40))+[5]*49
    assert len(owned+unowned)==A and len(set(owned+unowned))==Ug and len(owned)==Ao and len(set(owned))==Uo
    # Observer union, not intersection: an extra raw/logprob/confidence observer may
    # keep rows outside a greedy prefix alive. This is an illustrative common-atom
    # model, not evidence of extra rows in the already selected native head.
    universe=set(range(24));first=set(range(12));prefix=set(range(6));extra={10,20}
    demand=prefix|extra
    row_witness=dict(hypothetical_baseline_rows=24,first_proposal_rows=12,greedy_prefix_rows=6,
                     standalone_first_saved=12,standalone_prefix_saved=18,invalid_standalone_sum_saved=30,
                     joint_prefix_only_saved=18,all_observer_union_rows=sorted(demand),all_observer_saved=24-len(demand),
                     extra_observer_outside_first_proposal=20,need_reopen_first_row_filter=True,
                     native_head_already_selected=True,actual_prunable_row_histogram_known=False)
    assert not demand <= first and prefix <= first <= universe and 30>24
    # Fixed conventional dense BF16 GEMM domain, cold uncompressed one-weight-load
    # reference; row count controls work but does not remove W.
    H,Vp=5120,16160
    W=2*H*Vp
    head_models=[]
    for m in [2,6,12,24]:
        f=2*m*H*Vp; q=W+2*m*(H+Vp)
        fc=f/376000000; bw=q/1600000
        head_models.append(dict(rows=m,dense_direct_FLOPs=f,weight_bytes=W,input_output_bytes=2*m*(H+Vp),cold_materialized_reference_bytes=q,Cube_376TF_reference_us=fc,HBM_1p6TB_reference_us=bw,joint_reference_us=max(fc,bw),dominant_reference='HBM',measured=False))
        assert bw>fc
    assert W==165478400
    # Planner identity counterexample in the exact mathematical attention formula.
    # Same2keys/QSL/query count/attrs, all logits0; change only one V channel.
    old_output=(0+0)/2
    new_output=(0+1)/2
    assert old_output==0 and new_output==0.5
    plan_witness=dict(same_QSL=[0,1],same_effective_K=2,same_zero_logits=[0,0],old_values=[0,0],new_values=[0,1],old_attention=old_output,new_attention=new_output,conclusion='Same scheduling key/output is insufficient for attention output CSE; this finite exact-value witness does not execute the native kernel.')
    # Exhaustive set algebra for small universes, including all observer unions.
    set_cases=0
    for n in range(9):
        u=set(range(n))
        for ax in range(1<<n):
            a={i for i in u if ax>>i&1}
            for bx in range(1<<n):
                b={i for i in u if bx>>i&1}
                assert len(a|b)==len(a)+len(b)-len(a&b)
                assert (u-a)&(u-b)==u-(a|b)
                set_cases+=1
    assert set_cases==87381
    resource = dict(status='SYMBOLIC_CEILING',bound='For one fixed legal joint graph in the declared direct algorithm/precision domain: T >= max_r(Q_r_necessary(graph)/C_r_upper). Distinct work on one resource is summed once, independent resources take max; only proved unavoidable dependency paths may add time.',
                    identity_key=['producer semantic value','consumer role','row set','representation/dtype','parameter/state generation','storage level'],
                    composition='Build live observer closure, choose partition/representation, recompute retained and new work, then derive a joint bound. Union of eliminated-interface atoms is at most a preliminary reference budget, not a proof that every atom is actually deletable.',
                    new_work_must_be_priced=['key/dedup/owner lookup and metadata','gather/scatter/pack/unpack','extra copies and materialization','cross-core reductions and queues','weight replication and cold loads','changed full-row readiness and launch/buffer costs'],
                    no_additive_runtime_gain_claim=True,no_universal_GEMM_MAC_lower_bound_claim=True,
                    typed_Cube_Vector_HBM_L2_UVA_link_AICPU_CPU_capacities_separate=True,
                    missing_actual_capacities=['FLOAT/HF32/A2','Vector special/sort/index','INT64 remainder','L2/UVA host path','network algorithm/topology/injection','AICPU/launch/control'],
                    actual_request_numeric_ceiling=None)
    write(O/'candidate-relation-source-index-v73.json',dict(files=source_index,new_remote_source_read=False,new_target_source_read=False,overall_complete=False))
    write(O/'candidate-relation-clauses-v73.json',dict(clauses=clauses,higher_order_groups=groups,clause_count=len(clauses),selected_pair_count=len(selected_pairs),touched_candidates=sorted(touched),no_new_maincandidate=True,overall_complete=False))
    write(O/'candidate-relation-coverage-v73.json',dict(nodes=[dict(id=k,name=candidates[k]['name'],participates_in_selected_relation=k in touched) for k in sorted(candidates)],pairs=pairs,model_candidates=35,pair_universe=595,scoped_conditional_pairs=len(selected_pairs),remaining_pair_relations=595-len(selected_pairs),untouched_candidate_nodes=sorted(set(candidates)-touched),higher_order_groups=len(groups),all_pair_variants_closed=False,all_higher_order_combinations_analyzed=False,scope='595 is only the pair index for35 existing candidates.567 pairs are not centrally reconciled in this artifact; prior scoped clauses may exist and are not invalidated. No entry without a clause proves independence, impossibility or no opportunity. Neither35 candidates nor five groups prove complete whole-model opportunity discovery.',whole_modules_closed=0,overall_complete=False))
    write(O/'joint-candidate-work-models-v73.json',dict(Engram_owner_dedup=lookup,head_row_observer_union=row_witness,head_cold_dense_reference_models=head_models,plan_reuse_not_attention_value_reuse=plan_witness,joint_resource_model=resource,all_examples_conditional_or_constructed=True,new_device_or_profile_work=False,actual_runtime_speedup=None,overall_complete=False))
    write(O/'candidate-composition-math-validation-v73.json',dict(set_union_observer_cases=set_cases,lookup_population_feasible=True,lookup_joint_saved_bytes=lookup['joint_saved'],invalid_lookup_standalone_sum_bytes=lookup['invalid_sum_of_standalone_saved'],row_filter_needs_all_observer_union=True,head_reference_shapes=[x['rows'] for x in head_models],head_weight_bytes=W,all_four_cold_references_HBM_dominated=True,same_plan_different_attention_exact_value_witness=True,no_Torch_or_native_execution=True,actual_alias_reader_or_new_implementation_proof=False,overall_complete=False))
    print(dict(candidate_nodes=35,pair_universe=595,conditional_pair_relations=28,pending_pair_relations=567,touched_candidates=len(touched),untouched=sorted(set(candidates)-touched),higher_order_groups=5,union_cases=set_cases,lookup_joint_saved=lookup['joint_saved'],head_weight_bytes=W,new_candidates=0,overall_complete=False))


if __name__=='__main__':
    main()
