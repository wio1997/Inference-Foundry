"""Operator-specific conditional transformations, rounding and resource cuts."""
from pathlib import Path
import hashlib, struct
from finalize_origin_coverage_v9 import N, O, read, write


def f32(value): return struct.unpack('>f',struct.pack('>f',float(value)))[0]


def bf16(value):
    bits = struct.unpack('>I',struct.pack('>f',f32(value)))[0]
    assert (bits & 0x7f800000) != 0x7f800000, 'Finite witnesses only; no native NaN claims.'
    bits = (bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff0000
    return struct.unpack('>f',struct.pack('>I',bits))[0]


def main():
    assert not (O/'DownAdd-epilogue-variants-v82.json').exists()
    primary = read(O/'quant-epilogue-public-source-index-v82.json')
    details = read(O/'quant-epilogue-primary-details-v82.json')
    for record in primary['files']+details['files']:
        assert hashlib.sha256(Path(record['local_path']).read_bytes()).hexdigest() == record['sha256']
    source_facts = [
        dict(id='ENTRY',file='quant_batch_matmul_v3.cpp',lines=[[18,28],[126,161],[431,574]],
             statement='220-class entry has BF16/per-token TBE, OPT and BASIC <REDACTED_CREDENTIAL>, including perTokenScale input. Compile/format/transpose/template gates matter; not the no-perToken MC2 third-party header.'),
        dict(id='OPT',file='quant_batch_matmul_v3_pertoken_opt.h',lines=[[160,166],[407,415],[541,593],[625,650]],
             statement='AIC INT32 GetTensorC writes GM workspace. AIV waits for C event, reads INT32 to UB, does FP32 processing and final yType CAST_RINT in UB, then writes yGm. When yType=BF16, the final output producer is AIV. Pre-scale/bias arithmetic is branch-specific and must not be reassociated.'),
        dict(id='TBE',file='quant_batch_matmul_v3_pertoken.h',lines=[[477,498],[513,546],[550,565]],
             statement='GetTensorC returns GM workspace consumed as INT32 by DequantCompute, followed by per-token multiplication, optional pre-cast bias, yType CAST_RINT and UB-to-GM output. The BF16 cast can precede a subsequent Vector operation without a BF16 GM interface in a transformed graph.'),
        dict(id='BASIC',file='quant_batch_matmul_v3_pertoken_basic.h',lines=[[227,278]],
             statement='AIC computes INT32 workspace, AIV waits then invokes EpilogueDequant. This is a Cube-to-Vector bridge plus a Vector output epilogue, not direct AIC BF16 ownership.'),
        dict(id='BASIC-EPILOGUE',file='quant_batch_matmul_v3_basic_epilogue.h',lines=[[196,255],[259,290]],
             statement='DequantCompute reads INT32 GM, dequantizes and applies per-token scale/bias; CopyOutToGM performs yType CAST_RINT in UB before GM copy. For BF16 output the legal splice point is after that cast, preserving the exact dequant arithmetic.')]
    write(O/'quant-epilogue-source-contract-v82.json',dict(repo=primary['repo'],commit=primary['commit'],
        source_facts=source_facts,grade='DOCUMENTED_PINNED_PUBLIC_OPERATOR_SOURCE',
        loaded_CANN_9p2_binary_or_tiling_key_attested=False,
        saved_profile_support='22 BF16-output regimes /9671 executions have positive AIV work; INT32 GateUp control9/4472 has none. This is supporting evidence, not source-to-binary proof.',
        supersedes_unconditional_owner_reading_of_v80='v80 Cube-owned BF16->Vector interface remains a valid conditional model. It is not an established ownership fact for these per-token quantized outputs.',
        new_target_source_files=0,overall_complete=False))
    # The scalar witnesses check the stated finite IEEE-like f32/BF16-RNE model,
    # not device instructions, special values, reduction order or tolerance.
    pre_cast = dict(dequant_FP32=257.0,routed_BF16=-256.0,
        original=bf16(f32(bf16(257.0)+bf16(-256.0))),
        pre_cast_bias_replacement=bf16(f32(257.0+bf16(-256.0))))
    split_k = dict(INT32_partials=[257,-256],routed_BF16=0.0,
        original_full_K_then_BF16=bf16(f32(bf16(f32(257-256))+bf16(0))),
        BF16_partial_atomics_reference=bf16(f32(bf16(257)+bf16(-256))))
    assert (pre_cast['original'],pre_cast['pre_cast_bias_replacement']) == (0.0,1.0)
    assert (split_k['original_full_K_then_BF16'],split_k['BF16_partial_atomics_reference']) == (1.0,0.0)
    cases = []
    for x in [0,1,1.0078125,257,-257,0.00390625,65536]:
        for r in [0,1,-1,-256,0.00390625]:
            rounded_s = bf16(x);rounded_r = bf16(r)
            original = bf16(f32(rounded_s+rounded_r))
            local_splice = bf16(f32(rounded_s+rounded_r))
            assert struct.pack('>f',original) == struct.pack('>f',local_splice)
            cases.append(dict(dequant_FP32=x,routed=r,rounded_S=rounded_s,rounded_R=rounded_r,result=original))
    write(O/'epilogue-rounding-witnesses-v82.json',dict(pre_cast_bias_counterexample=pre_cast,
        K_split_BF16_atomic_counterexample=split_k,finite_model_consistency_examples=cases,
        conditional_identity='Z=BF16_RNE(f32(float(BF16_RNE(d))+float(R_BF16))). Store/load of the already-rounded S can be removed; the first cast cannot be cancelled.',
        proof_scope='Constructive finite scalar counterexamples and an identity in the explicitly stated f32/BF16-RNE model; not formal proof or native instruction testing.',
        current_foreach_precision_or_atomic_precision_newly_attested=False,overall_complete=False))
    common = [
        'Use the exact current dequant branch/arithmetic, complete INT32 K sum, channel and per-token scales, then the original BF16 S rounding before original Add rounding.',
        'Bind each routed tile after its pre-add reduction/scaling/output transform and token/feature trim, with matching rank,row,pad,layout and generation. Do not substitute raw MC2 M816/834 for merge M800.',
        'Audit all S/R readers, hooks, aliases and observable mutation/error/capture contracts. Keep required post-add reduction, truncation, ZeroExpertRouter and HC boundaries in original order.',
        'Readiness is tile-local: a new R-ready edge must be acyclic. A persistent shared epilogue cannot wait on routed work queued behind it on the same stream or occupy all resources needed by that producer.',
        'Price changed stream/event joins, extended INT32/FP32/BF16 liveness, UB and GM buffering, finite Cube/Vector contention, graph publication and all-rank progress. No free overlap or removed-launch gain is inferred.'
    ]
    variants = [
        dict(id='F23-VECTOR-FRESH',state='CONDITIONAL_OPERATOR_REORGANIZATION_REVIEWED',
            graph='AIC full INT32 Down -> GM -> AIV original dequant -> original BF16 S cast in UB -> original Add with ready BF16 R -> BF16 Z -> GM -> remaining hooks/HC',
            preserves_R=True,requires_R_last_reader=False,requires_no_other_S_materialization_reader=True,
            benefit='Avoid S BF16 GM store and reload (4n value bytes) if every S reader is contained or otherwise served. Raw INT32 Cube->Vector GM bridge remains.',
            extra_storage_reference='A simple explicit FP32 merge plan adds R_BF16 2t+R_FP32 4t+S_widen_FP32 4t =10t tile bytes; reusing a dead dequant FP32 buffer can make this 6t. Neither is a universal storage floor or current-tiling fit proof.',
            assumptions=common,legal_native_fusion_proved=False,critical_path_gain_us=None),
        dict(id='F23-BF16-ATOMIC-INPLACE',state='CONDITIONAL_API_SUPPORT_WITH_NUMERICAL_AND_ALIAS_GATES',
            graph='Original BF16 S in AIV UB -> one BF16 atomic add into fully ready existing R GM -> writer-completion fence -> original downstream consumers',
            preserves_R=False,requires_R_last_reader=True,requires_one_full_K_contribution_per_element=True,
            assumptions=common+['Destination is initialized to exact ready R, not unconditionally zeroed; each element has one S contribution. Disjoint M/N tiles are allowed, BF16 K partial accumulation is not equivalent in general.',
                'Public A3 SetAtomicAdd supports BF16 UB->GM, but native add precision/rounding, NaN payload/operand order, signed zero, saturation mode and completion visibility must match original Add. No-contention is not a throughput proof.',
                'Scope atomic state and DisableDmaAtomic so later stores are unaffected. Device-level BF16 saturation mode is not changed or queried in this analysis.'],
            atomic_source_transport_bytes_per_element=2,destination_RMW_value_bytes_per_element=4,
            source_transport_and_RMW_are_not_both_HBM_traffic=True,critical_path_gain_us=None),
        dict(id='F23-COPY-R-THEN-ATOMIC',state='CONDITIONAL_FRESH_OUTPUT_VARIANT_WITH_COPY_COST',
            graph='Copy R to fresh Z, then atomically add already-rounded S; preserve old R for other readers',
            reference_copy_bytes_per_element=4,destination_RMW_value_bytes_per_element=4,
            net_reference_value_byte_saving_vs_baseline=0,critical_path_gain_us=None,
            assumptions=common+['Copy completion before atomic and atomic completion before readers are separate edges. Atomic resource traffic/rate is unknown.']),
        dict(id='F23-ROUTED-DIRECT-DEST-ATOMIC',state='JOINT_F21_F23_PRODUCER_AND_OUTPUT_BANK_CONDITION',
            graph='Routed producer writes final rounded/transformed R directly to final Z bank; one rounded S atomic contribution follows',
            avoids_R_copy_only_if='All old-R readers/aliases, routed transform/trim, bank ownership and generation permit this producer rewrite.',
            assumptions=common,not_additive_with_VectorFresh_or_MergePost=True,critical_path_gain_us=None),
        dict(id='F23-RAW-INT32-FIXPIPE-BF16-ATOMIC',state='NOT_ESTABLISHED_BY_PUBLISHED_A3_FIXPIPE_API',
            exclusion_scope='Public A3 Fixpipe V220 type table has INT32->INT8/UINT8/HALF/INT32, not INT32->BF16. FLOAT->BF16 exists, but current INT8 dot is INT32 and per-token processing remains. Newer 950 INT32->BF16 support must not be imported into A3.',
            not_a_hardware_impossibility_proof=True,not_a_drop_in_current_ABI=True,critical_path_gain_us=None)
    ]
    write(O/'DownAdd-epilogue-variants-v82.json',dict(variants=variants,common_gates=common,
        preferred_analysis_frontier='Post-BF16-cast Vector epilogue with fresh output: fewer alias/atomic-specific assumptions than in-place atomic. This is a conditional candidate, not an implementation design or speedup claim.',
        native_precision_or_aliases_or_all_reader_closure_proved=False,
        numerical_counterexamples='epilogue-rounding-witnesses-v82.json',
        primary_API_semantics=dict(SetAtomicAdd='A3 BF16 and UB/L0C/L1->GM documented; no automatic zero, atomic mode scope, device-level 2201 BF16 saturation.',
            Fixpipe='A3 V220 conversion table; Fixpipe floating saturation CTRL mode is distinct from atomic device-level mode. Published support is not combined cast-before-add proof.'),
        no_new_maincandidate=True,implementation=False,native_benchmark=False,overall_complete=False))
    models = []
    for rows in [2,3,800,816,834]:
        elems = rows*5120
        models.append(dict(M=rows,H=5120,K=2304,elements=elems,
            conventional_dense_INT8_ops=2*elems*2304,
            common_raw_INT32_GM_write_bytes=4*elems,common_raw_INT32_GM_read_bytes=4*elems,
            separate_S_store_S_read_R_read_Z_write_value_bytes=8*elems,
            VectorFresh_R_read_Z_write_value_bytes=4*elems,
            removable_S_BF16_interface_reference_bytes=4*elems,
            base_total_value_bytes_including_raw_bridge=16*elems,
            VectorFresh_total_value_bytes_including_raw_bridge=12*elems,
            base_GM_read_bytes=8*elems,base_GM_write_bytes=8*elems,
            VectorFresh_GM_read_bytes=6*elems,VectorFresh_GM_write_bytes=6*elems,
            inplace_atomic_destination_RMW_value_bytes=4*elems,
            inplace_atomic_rounded_source_transport_bytes=2*elems,
            fresh_copy_then_atomic_value_reference_bytes=8*elems,
            reference_752TOP_dense_time_us=2*elems*2304/752e6,
            reference_1p6TB_total_base_us=16*elems/1.6e6,
            reference_1p6TB_total_VectorFresh_us=12*elems/1.6e6,
            reference_1p6TB_removable_interface_us=4*elems/1.6e6,
            actual_HBM_traffic_or_time_gain=None))
    write(O/'quant-epilogue-boundary-models-v82.json',dict(models=models,
        domain='Fixed A3 INT32 Cube->GM->Vector dequant path, each raw value written/read once, no compression/recomputation; already-rounded S has no unserved external reader; R starts in GM and final Z must be published to GM. Internal common scales/weights/quant input and later HC are outside this boundary.',
        symbolic_time_bound='For n=M*H, a conventional full dense Down and a fresh Vector merge: T >= max(2*K*n/C_INT8, 6*n/B_GMread, 6*n/B_GMwrite, 12*n/B_shared_aggregate_if_proved, L_original_Vector_sequence, L_true_dependencies). Each C/B must be a valid capacity UPPER and each counted transfer necessary in the declared domain; do not sum independent resource times.',
        symbolic_effective_down_throughput_upper='Q=2*K*n; Q/T <= min(C_INT8, (K/3)*B_GMread, (K/3)*B_GMwrite, (K/6)*B_shared_aggregate_if_proved, Q/L_Vector, Q/L_dependencies) over available valid positive bounds. Unknown terms remain symbolic.',
        hardware_rates_are_published_references_not_actual_upper_bounds=True,
        read_write_or_aggregate_capacity_upper_values=None,
        no_universal_algorithm_lower_bound=True,GM_value_bytes_are_not_HBM_bytes=True,
        time_difference_not_speedup_or_critical_path_gain=True,overall_complete=False))
    extensions=[]
    for prior in read(O/'V41-QB-RoPE-work-and-bound-models-v75.json')['models']:
        rows=prior['M'];q=rows*32768;tail=rows*64*64
        extensions.append(dict(M=rows,Q_B_elements=q,tail_elements=tail,
            conditional_common_INT32_CV_bridge_value_bytes=8*q,
            final_full_BF16_query_output_bytes=2*q,
            RoPE_tail_interface_removable_value_bytes=4*tail,
            original_v75_tail_budget=prior['tail_interface_reference_budget_bytes'],
            equal_to_prior=True,raw_INT32_bridge_eliminated=False))
        assert 4*tail==prior['tail_interface_reference_budget_bytes']
    write(O/'Q_B-index-epilogue-extension-v82.json',dict(Q_B_models=extensions,
        Q_B_scope='Active V4.1 target/draft query has no post-QB head RMS512. Full512 output remains; only tail64 RoPE is composed after original BF16 rounding. Frozen exact arithmetic/C-S/readiness/position/layout still apply.',
        Q_B_partner_gate='AIV tile must co-own matching 32+32 halves of each 64-wide tail and retain unrotated coordinates. Retile/buffer/exchange costs may be required; no full M3 query+CS fit in one UB is assumed.',
        binding='1167 DIRECT CPU source-chain owners from v75 retained. 3200 graph bindings remain template/source grade and 16 graphs phase-unbound. New profile evidence does not upgrade invocation/alias/phase identity.',
        index_Q_scope='If the same BF16 Vector output ownership applies, append original BF16 RoPE then per-head128 quant after the first BF16 cast. Keep rounded FP16 scale/div_rn/min2^-24/nearbyint, row amax and all output readers. Current native tiling/output owner not attested.',
        index_Q_max_single_BF16_plane_removable_value_reference='4*M*4096 only when all original BF16 plane readers are absorbed; RoPE intermediates/quant scale/state costs must be reconciled before whole-subgraph savings.',
        no_post_QB_head_RMS_512_in_active_case=True,all_extensions_native_legality_proved=False,
        additional_actual_gain=None,overall_complete=False))
    print(dict(conditional_DownAdd_variants=len(variants),shape_models=len(models),public_source_files=5,
        M3_S_interface_bytes=4*3*5120,M800_S_interface_bytes=4*800*5120,
        raw_INT32_bridge_remains=True,numerical_counterexamples=2,overall_complete=False))


if __name__ == '__main__': main()
