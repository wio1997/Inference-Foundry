"""Close selected relation-index gaps and repair Q_B/head-RMS/RoPE applicability.

Saved profile/source clauses and finite mathematical witnesses only.
"""
from collections import Counter, defaultdict
from copy import deepcopy
from fractions import Fraction
from pathlib import Path
import hashlib, math, struct
from finalize_origin_coverage_v9 import N,O,W,read,write
from mirror_saved_profile_core_v53 import CACHE
from analyze_projection_ceiling_v61 import stat


def f32(v):
    return struct.unpack('f',struct.pack('f',v))[0]


def bf16_rne(v):
    u=struct.unpack('I',struct.pack('f',v))[0]
    u=(u+32767+((u>>16)&1))&0xffff0000
    return struct.unpack('f',struct.pack('I',u))[0]


def head_norm_work(m):
    rows=m*64;values=rows*512
    return dict(M=m,head_rows=rows,head_values=values,
                square_multiplies=values,sum_reduction_adds=rows*511,
                mean_scale_ops=rows,epsilon_adds=rows,rsqrt_ops=rows,
                output_scale_multiplies=values,
                BF16_to_FLOAT_values=values,FLOAT_to_BF16_values=values,
                QB_INT8_direct_GEMM_ops=2*m*1280*32768,
                full512_reduction_precedes_every_rotated_tail=True,
                gamma_one_multiply_present_in_native_API_unknown=True,
                domain='Declared direct unweighted RMS scalar reference;no universal instruction lower bound or native accumulation/throughput identity.',
                actual_Vector_special_function_capacity=None)


def main():
    assert not (O/'remaining-fusion-edge-costs-v74.json').exists()
    previous=read(O/'candidate-relation-clauses-v73.json')
    oldpairs={(r['a'],r['b']) for r in previous['clauses']}
    refs={
        'HC':'HC-fusion-split-reorganization-variants-v59.json',
        'projection':'projection-fusion-variants-and-interactions-v61.json',
        'projection_contract':'projection-operator-numerical-contract-v61.json',
        'router':'router-vector-fusion-variants-v63.json',
        'rotary_contract':'router-vector-numerical-contract-v63.json',
        'context':'index-context-operator-contract-v64.json',
        'context_options':'index-context-fusion-variants-v64.json',
        'context_costs':'index-context-fusion-costs-v64.json',
        'context_partition':'F27-F29-row-feature-and-KV-reorganization-v34.json',
        'C1':'F33-consumer-and-numerical-contract-v27.json',
        'C1_algebra':'F33-projection-algebra-and-layout-v27.json',
        'C1_options':'F33-joint-projection-variants-v27.json',
        'selected':'selected-input-def-use-and-guard-proof-v70.json',
        'comm':'communication-fusion-variants-v66.json',
        'head':'heads-sampling-fusion-variants-v65.json',
        'planner':'planner-source-algorithm-contract-v71.json',
        'support':'prior-support-file-binding-reconciliation-v71.json',
        'pad':'pad-owner-API-namespace-reconciliation-v72.json',
    }
    sources=[]
    for tag,fn in refs.items():
        p=O/fn;assert read(p)
        sources.append(dict(tag=tag,path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    sroot=W/'frozen-source-snapshot/vllm-ascend/vllm_ascend'
    pins=[
        (N/'projection-operator-source-v61/vllm_ascend/attention/context_parallel/dsa_cp.py','7e8dd93eeda988866138342523ecfa7cc9c42ce67669d4b1507ee8c5f3452305',[1825,1866,2028,2054]),
        (sroot/'device/device_op.py','01ce13ea60e46f901ad69ad4c71e4be0e31cd69fbc56ebb7d898f957d5e54f67',[724,737]),
        (sroot/'models/deepseek_v41/model.py','101f96b38e9fc47e7be147ddfe56976aee896867c73a33a7d95c3e3c833a6e91',[405,406,630,640]),
        (W/'sampler-source-supplement/frozen-source-snapshot/vllm-ascend/vllm_ascend/ops/layernorm.py','d2fda42ad99935725c98b9505a8d2e94696d9cdb2906e5cb2f1e37bf21fb27bc',[49,58,80,105]),
    ]
    source_clauses=[]
    for p,sha,lines in pins:
        assert hashlib.sha256(p.read_bytes()).hexdigest()==sha,p
        source_clauses.append(dict(path=str(p),sha256=sha,selected_lines=lines,previously_pinned_source_reused=True))
    cp_source=pins[0][0].read_text()
    assert 'q = DeviceOperator.apply_dsa_q_rms(q, self.eps, self.q_norm_without_weight)' in cp_source
    assert cp_source.index('q = DeviceOperator.apply_dsa_q_rms')<cp_source.index('torch.ops._C_ascend.inplace_partial_rotary_mul',cp_source.index('q = DeviceOperator.apply_dsa_q_rms'))
    # Audit native visibility in existing3200 graph QB/query-RoPE block sequences.
    # Ordering is used only to inspect visible stages, NOT to prove tensor identity.
    qr=read(O/'router-vector-regime-models-v63.json')['regimes']
    qregs=[r for r in qr if r['role']=='QUERY_ROPE_BEFORE_ATTENTION']
    assert len(qregs)==2 and sum(r['executions'] for r in qregs)==3200
    bydb_rope=defaultdict(list)
    for r in qregs:
        for db,tid in r['native_TASK_keys']:
            bydb_rope[db].append(tid)
    chains=read(O/'projection-graph-chains-v61.json')['graph_projection_chains']
    assert len(chains)==3200
    bydb_qb=defaultdict(list)
    for c in chains:bydb_qb[c['database_index']].append(c)
    visible=[];between_ops=Counter();between_types=Counter();head_rms=0
    for db in sorted(bydb_qb):
        data=read(CACHE/f'analysis/native-execution-identity-v14/db-{db:02d}.json')
        tasks={t['task_record_rowid']:t for t in data['TASK_records']}
        defs={d['globalTaskId']:d for d in data['COMPUTE_definitions']}
        qs=sorted(bydb_qb[db],key=lambda c:c['stages']['Q_B']['startNs'])
        rs=sorted((tasks[t] for t in bydb_rope[db]),key=lambda t:t['startNs'])
        assert len(qs)==len(rs)
        for c,rope in zip(qs,rs):
            qb=c['stages']['Q_B'];qt=tasks[qb['task_record_rowid']]
            assert qt['streamId']==rope['streamId'] and qt['endNs']<=rope['startNs']<c['stages']['WO_A']['startNs']
            middle=[t for t in data['TASK_records'] if t['streamId']==qt['streamId'] and t['startNs']>=qt['endNs'] and t['endNs']<=rope['startNs'] and t['task_record_rowid']not in [qt['task_record_rowid'],rope['task_record_rowid']]]
            stages=[]
            for t in middle:
                d=defs.get(t['globalTaskId']);op=d['opType_text'] if d else None
                between_types[t['task_type_name']]+=1
                if op:between_ops[op]+=1
                if op=='RmsNorm' and '512' in d['inputShapes_text']:head_rms+=1
                stages.append(dict(TASK_rowid=t['task_record_rowid'],task_type=t['task_type_name'],operator=op,input_shapes=d['inputShapes_text'] if d else None))
            visible.append(dict(database_index=db,graph_ordinal=c['graph_ordinal'],block_ordinal=c['block_ordinal'],Q_B_TASK_rowid=qt['task_record_rowid'],query_RoPE_TASK_rowid=rope['task_record_rowid'],same_stream=True,between_native_stages=stages,device_interval_ns=rope['startNs']-qt['endNs'],scope='Existing graph/source role population ordered bracket; no pointer or loaded-source identity proof.'))
    assert len(visible)==3200
    # Build source-profile applicability as two separately priced domains.
    contract=dict(source_files=source_clauses,evidence_refs=[refs[x] for x in ['projection_contract','router','rotary_contract']],
                  current_pinned_path=['Q_B GEMM BF16 output','apply_dsa_q_rms over512/head','partial RoPE64/head with FLOAT C/S','BF16_RINT query','attention'],
                  QR_norm1280_is_a_distinct_earlier_activation=True,
                  source_requires_full512_for_head_RMS=True,
                  preceding_model_defines_unweighted_head_RMS512=True,
                  historical_loaded_call_path_attested=False,
                  native_standalone_head_RMS_visibility_known=False,
                  current_source_is_not_historical_graph_attestation=True,
                  correction='Earlier F07-QB-ROPE/QB-ROPE descriptions omitted the explicit head-RMS call in this pinned CP path. Their direct two-stage description is superseded for this path. A no-head-RMS path needs separate applicability evidence; otherwise preserve the full head RMS and both BF16 boundaries in a triple fusion, or attach RoPE to its real norm producer.',
                  retained_alias_readers_rstd_generation_and_capture_conditions=True,
                  new_remote_source_read=False,local_pinned_operator_clauses_read=True,overall_complete=False)
    write(O/'QB-head-RMS-RoPE-applicability-v74.json',contract)
    write(O/'QB-to-query-RoPE-native-visibility-v74.json',dict(graph_sequences=visible,sequence_count=3200,visible_middle_ComputeInfo_ops=dict(between_ops),visible_middle_TASK_types=dict(between_types),visible_standalone_RmsNorm512=head_rms,interval_ns=stat([r['device_interval_ns'] for r in visible]),conclusion='No standalone512-head RMS identity is established by this saved bracket. This does not prove RMS absence, fusion inside a recorded native kernel, or applicability of the newly inspected current caller to historical graphs.',native_alias_or_tensor_identity_proved=False,missing_PMU_or_phase_binding_resolved=False,overall_complete=False))
    # Full-head RMS cannot be computed from just the unchanged rotated tail.
    eps=Fraction(1,1000000)
    denom_a=Fraction(64,512)+eps
    denom_b=Fraction(64+448*4,512)+eps
    assert denom_a!=denom_b
    fullhead_witness=dict(H=512,tail=64,tail_values='all1 in both inputs',prefix_a='448 zeros',prefix_b='448 twos',epsilon=str(eps),normalized_tail_squared_a=str(1/denom_a),normalized_tail_squared_b=str(1/denom_b),conclusion='Identical64-tail values have different head-RMS results when the other448 values change. Full512 reduction is necessary in the declared source RMS semantics.')
    def norm(x):
        ss=0.
        for v in x:ss=f32(ss+f32(v*v))
        inv=f32(1/math.sqrt(f32(f32(ss/512)+1e-6)))
        return [bf16_rne(f32(v*inv)) for v in x]
    cs=f32(math.sqrt(.5))
    def rotate(x):
        y=list(x)
        for i in range(448,512,2):
            y[i]=bf16_rne(f32(f32(x[i]*cs)-f32(x[i+1]*cs)))
            y[i+1]=bf16_rne(f32(f32(x[i+1]*cs)+f32(x[i]*cs)))
        return y
    x=[0.]*448+[.125,3.]*32
    original=rotate(norm(x));reordered=norm(rotate(x))
    assert original[-2:]==[-2.703125,2.953125] and reordered[-2:]==[-2.71875,2.9375]
    order_witness=dict(input_prefix='448 zeros',input_tail_pair=[.125,3.],tail_pairs=32,cos_sin_FLOAT=cs,epsilon=1e-6,original_norm_then_RoPE_last_pair=original[-2:],reordered_RoPE_then_norm_last_pair=reordered[-2:],scope='Finite scalar FP32 operations with explicit BF16 nearest-even boundaries. Not a native RMS accumulation/precision attestation; disproves unconditional stage interchange in this declared arithmetic model.')
    # Role-labelled stage cost algebra. F=full head values; R=rotated tail values.
    # Every value-byte count is a reference model, not automatically actual HBM.
    traffic=[]
    for m in [2,3,800,816,830,834]:
        f=m*64*512;r=m*64*64
        traffic.append(dict(M=m,heads=64,full_values=f,tail_values=r,
                            QB_RoPE_without_head_RMS_reference_bytes=2*f+4*r,
                            QB_head_RMS_RoPE_reference_bytes=6*f+4*r,
                            separate_QB_fused_head_RMS_RoPE_reference_bytes=6*f,
                            joint_QB_head_RMS_RoPE_reference_bytes=2*f,
                            head_RMS_RoPE_consumer_budget_bytes=4*r,
                            all_three_joint_budget_bytes=4*f+4*r,
                            attention_inverse_RoPE_full_pack_reference_bytes=6*f+4*r,
                            standalone_inverse_RoPE_pack_consumer_reference_bytes=6*f,
                            attention_inverse_RoPE_direct_pack_reference_bytes=2*f,
                            native_full_pack_for_this_M_proved=False,
                            cos_sin_loads_aux_rstd_and_queue_costs_not_priced=True,
                            actual_physical_HBM_or_latency_gain=None))
    c1=[]
    for m in [24,6400,6528,6668]:
        weights=2*5120*1024;f=2*m*5120*1024
        old=weights+4*m*5120+2*m*1024
        joint=weights+2*m*5120+2*m*1024
        assert old-joint==2*m*5120
        c1.append(dict(M=m,direct_dense_FLOPs=f,two_distinct_BF16_weight_bytes=weights,separate_materialized_reference_bytes=old,joint_materialized_reference_bytes=joint,shared_input_reference_budget_bytes=old-joint,unchanged_compute_reference_us=f/376000000,separate_HBM_reference_us=old/1600000,joint_HBM_reference_us=joint/1600000,separate_reference_us=max(f/376000000,old/1600000),joint_reference_us=max(f/376000000,joint/1600000),separate_512_norms_and_preRoPE_fork_retained=True))
    assert c1[1]['separate_reference_us']==c1[1]['joint_reference_us']
    # Restore each ordered selected row from three independently retained sources.
    row_cases=0
    for n in range(1,17):
        a=[[100*j+i for i in range(n)] for j in range(3)]
        for indices in [list(range(n)),list(reversed(range(n))),[0,n-1,0]]:
            concat_then_select=[tuple(a[j][i] for j in range(3)) for i in indices]
            select_then_concat=list(zip(*[[source[i] for i in indices] for source in a]))
            assert concat_then_select==select_then_concat
            row_cases+=1
    local_retention=[]
    for local_m in [3,800,816,834]:
        local_retention.append(dict(local_rows=local_m,three_BF16_mean_planes_bytes=3*2*local_m*5120,three_HC4_planes_if_retained_instead_bytes=3*2*local_m*4*5120,all_three_mean_values_remain_distinct=True,allocated_VRAM_reclaimed=False,late_selection_I_known_only_after_target_sampling=True))
    # Extend the central relation ledger, keeping all previous28 clauses verbatim.
    new=[]
    def edge(a,b,kind,claim,tags,domain):
        a,b=sorted([a,b]);assert (a,b)not in oldpairs
        new.append(dict(id=f'REL74-{len(new)+1:02d}',a=a,b=b,relation=kind,operator_claim=claim,evidence=[refs[x] for x in tags]+['QB-head-RMS-RoPE-applicability-v74.json'] if 'QB' in claim else [refs[x] for x in tags],applicable_domain=domain,conditional_compatibility=True,all_runtime_gates_proved=False,speedup_additive=False,critical_path_gain_us=None))
    edge('F01','F27','SHARED_HC_INPUT_NOT_EQUAL_OUTPUT','HC mean may share HcPre loads at fixed checkpoints, but weighted Y is not the BF16 arithmetic mean. New coefficient outputs and aux mean stores/readiness remain.',['HC','context','context_options'],'Same checkpoint37/38/39 source generation without Engram mutation; all mean/Pre observers.')
    edge('F01','F02','SPLIT_COEFFICIENT_AND_Y_READINESS','Delayed external-premix collapse/Y norm and current coefficients can be scheduled separately; joining mean/Y outputs must not silently delay earlier auxiliary or attention consumers.',['HC','context'],'Current HC full4H coefficient norm/HF32 and all BF16 stage boundaries; new output/event ABI conditional.')
    edge('F05','F07','DISTINCT_QR1280_AND_HEAD512_NORMS','QB path may require head512 RMS after QR1280 norm and QB BF16 output. Different activations, widths and rstd; a shared earlier scale cannot replace head RMS.',['projection_contract','context'],'Pinned CP source path requires full512/head; historical source/native binding remains open.')
    edge('F06','F07','JOINT_PROJECTION_DOES_NOT_ELIDE_HEAD_NORM','QB/IndexQ shared quant/input tiling does not remove a QB head512 RMS. Preserve distinct outputs, BF16 boundaries and the head-reduction dependency before tail64 RoPE.',['projection','router'],'Exact QR quantizer/row generation; source path with vs without explicit head norm priced separately.')
    edge('F07','F13','CONSUMER_FUSION_RETAINS_QUERY_TRANSFORMS','QB/head-RMS/RoPE into attention may retain intermediates onchip, but must reproduce full512 normalization and BF16 query values before score computation, with shared Cube/Vector resources.',['projection','router'],'No input-transform deletion or same-value assumption from graph adjacency.')
    edge('F07','F14','FORWARD_AND_INVERSE_ROTATE_DIFFERENT_VALUES','QB query RoPE changes attention scores, whereas inverse RoPE acts on attention output. They are not inverse functions on one unchanged tensor and cannot mutually cancel.',['rotary_contract','comm'],'Preserve BF16 attention result before inverse RoPE; direct pack requires new output layout ABI.')
    edge('F27','F29','CHOOSE_ROW_OR_FEATURE_PARTITION_BEFORE_COST','Main15360→5120 context path requires full5120 RMS, then three distinct512 KV maps/norms. Complete-weight row ownership and feature-partial RS/AG are alternatives with different weights/padding/retained-bank costs.',['context','context_options','context_partition'],'Main and each KV BF16 rounding, actual I/positions/groups/slots and query-weight readers retained.')
    edge('F26','F27','DISTINCT_CHECKPOINT_AND_TAIL_GENERATIONS','Final HC collapse/norm and three earlier auxiliary checkpoint means are distinct values. Equal row shape or a reused buffer is insufficient for cross-layer activation CSE.',['context','head'],'All checkpoint means and final/raw tail observers retain their own semantic generation.')
    edge('F27','F31','ORDERED_SELECTOR_AND_OLD_SOURCE_LIFETIME','Late selective gather and indirect main projection share one ordered I plan, including duplicates. I arrives after target sampling; retain required checkpoint means and snapshot/last-reader state until then.',['context','context_options','selected'],'Bounds/negative indexing/error and capture bank publication preserved; native prep budgets counted once.')
    edge('F29','F31','SLOT_INLINE_AND_CSE_ALTERNATIVES','Slot pair inline production and CSE remove overlapping metadata work. Reuse needs identical group bank/blocksize/slot generation; different layer caches and per-layer KV writes remain.',['context','context_options','context_partition'],'All negative slots map invalid, INT32/INT64 policy and duplicate-slot write order unchanged.')
    edge('F08','F33','SWA_CHAIN_AND_JOINT_C1_SHARE_PROJECTION','C1/SWA joint projection overlaps the SWA producer in its norm/RoPE/cache fusion. Keep two distinct512 weights/norms and both cache publications; a single1024 RMS is not equivalent.',['C1','C1_algebra','C1_options','projection'],'C1 source layer/role only, BF16 boundaries/packing/readiness retained.')
    edge('F11','F33','C1_PRE_ROPE_KEY_FORK_WAR','Joint C1/SWA projection and IndexKey full-chain fusion still require reading the C1 normalized latent before in-place long RoPE overwrites overlapping rows/tiles. Index publication is not automatically a universal whole-operator barrier.',['C1','C1_options','context_options'],'All key/scale/long/SWA generation consumers retained; no tensor identity inferred by cache shape.')
    edge('F19','F33','C1_LATENT_FORK_OVERLAPPING_INTERFACES','C1 part of compressor latent-fork fusion and C1/SWA joint projection touch the same C1 latent interfaces. Count shared loads once and keep pre-RoPE IndexKey versus post-RoPE long-KV outputs distinct.',['C1','C1_options','projection'],'C1 layer-specific graph; C2 pool/state branch remains a separate mode.')
    edge('F32','F33','PROJECT_THEN_EXCHANGE_TWO_ROLE_PLANES','C1/SWA may project local rows and exchange two512 BF16 planes rather than5120 hidden features. The5x content ratio requires equivalent replicated weights, all global readers, row/slot/position plan and new publication layout.',['C1_options','C1_algebra','comm'],'Feature content only, not5x wire/time speedup; no extra ratio from already-sharded queries.')
    edge('F17','F33','DIFFERENT_COMPRESSOR_MODE_AND_DTYPE','C1 BF16 single projection and C2 two FLOAT projections with gate/pool are distinct roles, even with512-wide outputs. Cross-mode weight/output/normalization CSE is unsupported; choose the relevant per-layer mode.',['C1','projection','context'],'C1 and C2 modes can coexist at different model layers; not mutually exclusive whole-model alternatives.')
    assert len(new)==15 and len({(r['a'],r['b'])for r in new})==15
    clauses=deepcopy(previous);clauses['clauses']+=new
    clauses['clause_count']=clauses['selected_pair_count']=len(clauses['clauses'])
    clauses['touched_candidates']=sorted({x for r in clauses['clauses']for x in [r['a'],r['b']]})
    assert clauses['clause_count']==43 and len(clauses['touched_candidates'])==35
    groups=[
        dict(id='HG74-Q-ATTENTION',members=['F05','F06','F07','F13','F14'],contract='QR1280 and post-QB head512 norms are distinct; preserve all query/attention/inverse-rotary BF16 boundaries and price one selected materialization/pack graph.'),
        dict(id='HG74-AUX-CONTEXT',members=['F01','F02','F27','F29','F31'],contract='Same HC input may serve distinct mean/coefficient/Y outputs; choose early/full vs late/selected aux work, full5120 global norm/three512 KV norms, and exact ordered slot-bank publication.'),
        dict(id='HG74-C1',members=['F08','F11','F19','F32','F33'],contract='Two distinct512 BF16 projections/norms; C1 pre-longRoPE IndexKey WAR/fork and both cache publications remain under local projection/new row exchange.'),
    ]
    for group in groups:group.update(scope_state='SCOPED_COMPOSITION_CONSTRAINTS_RECORDED;ALL_VARIANTS_AND_HIGHER_ORDER_CLOSURE_OPEN',all_subsets_analyzed=False,legal_joint_fusion_proved=False)
    clauses['higher_order_groups']+=groups
    graph=deepcopy(read(O/'candidate-relation-coverage-v73.json'));lookup={(r['a'],r['b']):r for r in new}
    for row in graph['pairs']:
        if (row['a'],row['b']) in lookup:row.update(scope_state='SCOPED_OPERATOR_RELATION_REVIEWED_CONDITIONAL',clauses=[lookup[(row['a'],row['b'])]['id']])
    for row in graph['nodes']:row['participates_in_selected_relation']=True
    graph.update(parent_relation_index='candidate-relation-coverage-v73.json',scoped_conditional_pairs=43,remaining_pair_relations=552,untouched_candidate_nodes=[],higher_order_groups=8,scope='All35 existing candidates now have at least one selected relation;this is not full pair/variant/whole-model discovery closure.552pairs not centrally reconciled here;prior clauses may exist. No absence/independence conclusion from blank rows.',overall_complete=False)
    write(O/'candidate-relation-clauses-v74.json',clauses)
    write(O/'candidate-relation-coverage-v74.json',graph)
    write(O/'remaining-fusion-edge-costs-v74.json',dict(QB_and_output_materialization_models=traffic,head_RMS_source_direct_work_models=[head_norm_work(x['M']) for x in traffic],C1_joint_projection_models=c1,aux_late_selection_retention=local_retention,source_version_and_branches_explicit=True,reference_peak_domain='Cold uncompressed materialized payload and direct dense BF16 GEMM, one weight read;376TF/s and1.6TB/s reference capacities assumed applicable. Actual HBM/L2/Vector/control capacity unproved.',joint_theory='T>=max(Qnecessary_resource/Cupper_resource) on one selected legal graph. Extra all-head reductions, cos/sin/rstd/queues/events and changed resource concurrency cannot be assumed free.',theory_status='CONDITIONAL_CEILING',physical_numeric_full_joint_ceiling_proved=False,actual_runtime_gain=None,overall_complete=False))
    write(O/'remaining-fusion-edge-math-v74.json',dict(full512_head_RMS_tail_only_counterexample=fullhead_witness,BF16_stage_order_counterexample=order_witness,ordered_three_source_selection_cases=row_cases,inverse_cancellation_counterexample=dict(query=[0,0],keys_and_values=[[1,0],[0,1]],attention=[.5,.5],inverse90=[.5,-.5],deleting_both_rotaries=[.5,.5],scope='Exact real/representable toy demonstrating different input/output tensors;not a claim actual captured rotary table contains90degrees.'),native_tests=False,new_Torch_model_NPU_imports=False,overall_complete=False))
    write(O/'remaining-fusion-edge-source-index-v74.json',dict(saved_contracts=sources,local_operator_files=source_clauses,new_remote_source_read=False,new_source_files_obtained=0,local_pinned_operator_clauses_read=True,scope='Only norm/rotary/projection/HC/slot value and dependency clauses,not framework lifecycle.',overall_complete=False))
    print(dict(previous_pairs=28,new_pairs=15,total_pairs=43,central_pending=552,all35_candidates_touched=True,higher_order_groups=8,graph_visibility_sequences=3200,visible_middle_ops=dict(between_ops),head_RMS_visibility=head_rms,C1_reference_rows=[24,6400,6528,6668],ordered_selection_cases=row_cases,source_files=4,overall_complete=False))


if __name__=='__main__':main()
