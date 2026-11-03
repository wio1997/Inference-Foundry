"""Pure-projection scheduling/memo domains and HC precision/resource boundaries."""
from datetime import datetime,timezone
from urllib.request import Request,urlopen
from math import exp,sqrt,floor,copysign,isfinite
import struct,hashlib
from capture_A3_datapath_evidence_v80 import Extract
from finalize_origin_coverage_v9 import O,read,write


def f32(value):
    try:return struct.unpack('>f',struct.pack('>f',float(value)))[0]
    except OverflowError:return copysign(float('inf'),value)


def bf16(value):
    value=f32(value);assert isfinite(value)
    bits=struct.unpack('>I',struct.pack('>f',value))[0]
    bits=(bits+0x7fff+((bits>>16)&1))&0xffff0000
    return struct.unpack('>f',struct.pack('>I',bits))[0]


def main():
    assert not(O/'Engram-WKV-hoist-and-memo-variants-v84.json').exists()
    url='https://docs.pytorch.org/docs/2.8/generated/torch.split.html'
    blob=urlopen(Request(url,headers={'User-Agent':'OperatorLifetimeEvidence/1.0'}),timeout=15).read()
    page=O/'torch-split-primary-source-v84.html';page.write_bytes(blob)
    parser=Extract();parser.feed(blob.decode());plain=' '.join(parser.parts)
    assert 'Each chunk is a view of the original tensor' in plain
    write(O/'torch-split-primary-evidence-v84.json',dict(url=url,version='PyTorch2.8publicAPI,notinstalledserverversion/binaryattestation',
        snapshot_path=str(page),snapshot_bytes=len(blob),snapshot_sha256=hashlib.sha256(blob).hexdigest(),
        retrieved_UTC=datetime.now(timezone.utc).isoformat(),fact='Split chunks are views of the original tensor; model source key/value split can therefore retain combined KV backing in the declared view-backed storage domain.',
        actual_NPU_addresses_or_allocator_liveness_attested=False,no_browser_used=True,overall_complete=False))
    gate=f32(1/(1+exp(-sqrt(2**-14))));assert gate==0.501953125
    hidden=[256.0,-256.0,0.0,0.0];value=2.0;add=f32(gate*value)
    unrounded=[f32(x+add)for x in hidden];rounded=[bf16(x)for x in unrounded]
    ordered=lambda values:f32(f32(f32(values[0]+values[1])+values[2])+values[3])
    original=bf16(ordered(rounded));bypassed=bf16(ordered(unrounded))
    assert (original,bypassed)==(5.0,4.0)
    a,b,c=2.0**100,2.0**30,2.0**-30
    original_product=f32(f32(a*b)*c);preweighted=f32(a*f32(b*c))
    assert original_product==float('inf')and preweighted==a
    write(O/'Engram-HC-rounding-witnesses-v84.json',dict(
        gate_before_HC_collapse=dict(hidden_BF16=hidden,gate_FP32=gate,shared_value_BF16=value,external_pre_mix=[1,1,1,1],
            unrounded_gate_outputs=unrounded,original_BF16_gate_outputs=rounded,
            original_HC_Y_BF16=original,skip_gate_BF16_round_HC_Y=bypassed,
            qualification='Finite scalar f32/BF16-RNE model. Gate chosen as f32(sigmoid(sqrt(2^-14))), consistent with positive scalar gate formula. Not actual frozen weight outputs or native gate/HcPre instruction parity.'),
        preweight_key_reassociation=dict(rotated_hidden_FP32=a,channel_weight_FP32=b,key_BF16=c,
            original_left_associated='Inf',preweighted_key_result=preweighted,
            original_expression='f32(f32(a*b)*c)',changed_expression='f32(a*f32(b*c))',
            qualification='Finite input FP32 overflow counterexample to unconditional reassociation; native saturation/precision remains a separate contract.'),
        exact_scalar_hoist='Cache key_rstd=rsqrt(mean(key.float()**2)+eps) in original FP32 arithmetic; later execute original hidden_rstd*=key_rstd and original raw dot/order. Do not replace with normalized/preweighted key.',
        formal_or_native_validation=False,overall_complete=False))
    gates=[
      'The current target guard is active, rows[:n] and exact DP/TP/SP row/head mapping and lookup-ready generation are known. Preserve original table ownership/zeros, dtype/device/packing/native BF16 WKV numerical policy and all collective participation.',
      'WKV layer/weight/quantizer version and packed/native algorithm contract stay fixed. Current hidden values are not inputs to WKV, but gate and following HcPre coefficients/Y still depend on current updated hidden.',
      'Hoisted outputs have private immutable generation/row banks until last reader; old readers AND writers retire before reuse. No stale capture bank/address/ABA substitution or tensor-shape equality as value proof.',
      'Early scheduling preserves conditional errors/side effects and the original native numerical contract. It may extend GM liveness and compete with current HF32/INT8/BF16 Cube work, Vector, memory and communications; no free overlap or removed-work claim.'
    ]
    variants=[
      dict(id='F25-WKV-LOOKUP-READY',state='CONDITIONAL_OPERATOR_REORDERING_REVIEWED',
        graph='lookup/TP/SP rows ready -> WKV_BF16 ahead of layer hidden; current hidden -> original rotation/gate; join -> original BF16 gate result -> HcPre',
        permits='Projection for target layer14 can be scheduled earlier than its hidden input when the exact row table is ready. Not across a weight/table/row generation change.',
        work_deleted=False,all_original_hash_collective_sharing_is_baseline=True,conditions=gates,actual_gain=None),
      dict(id='F25-KEY-RSTD-EARLY',state='CONDITIONAL_SCALAR_HOIST_REVIEWED',
        graph='WKV_BF16 -> original key.float/square/mean/eps/rsqrt -> cached4FP32scalars pertoken; later multiply original hidden_rstd in original order',
        conditions=gates+['Ordinary common-gate branch only (captured rotated path). Unrotated native fused-gate branch does not expose a proven compatible scalar handoff.',
            'Preserve exact reduction/cast arithmetic and eps. Keeping BF16 KV plus only scalar may require another key read/cast later; retaining original FP32 key instead requires larger live backing.'],
        work_deleted=False,actual_gain=None),
      dict(id='F25-WKV-MEMO',state='CONDITIONAL_CROSS_ITERATION_PURE_OUTPUT_CACHE_REVIEWED',
        graph='Generation-keyed identical row content -> immutable original WKV_BF16 output; misses compute original dense WKV; every current hidden still gates freshly',
        key='LayerID + exact24head IDs/full equality and table/storage/dequant/ownership versions producing identical6144BF16 rows + WKVweight/packing/device/quant/native numeric algorithm generation. RMS cache additionally keys eps/reduction policy. If ID tuple cannot certify row content use exact content/generation proof.',
        does_not_require_same_hidden_value=True,does_not_reuse_hidden_gate_or_HC_coefficients=True,
        conditions=gates+['Capture-compatible output materialization, lookup/insert/eviction/lastreader protocol and miss packing remain costs. No actual duplicate/hit rate or available cache budget was recorded.',
            'Plan here keeps current lookup/DP/TP/SP production; a hit does not automatically delete those collectives/UVA reads. Crossrank result distribution can be51200B/row versus12288B original full feature input, so no automatic8x saving.',
            'Key equality must include every row-content dependency and use collision-safe equality. Cache output/shape-specific native tiling must preserve the original BF16 result; different M can change numerical algorithms.'],actual_gain=None),
      dict(id='F25-GATE-HC-COLLAPSE',state='CONDITIONAL_POST_BF16_VECTOR_HANDOFF_REVIEWED',
        graph='original gate BF16 E tile -> original external-premix collapse -> BF16Y -> norm; publish E for current coefficient/HcPost residual and other live readers',
        conditions=gates+['Do not cancel the gate BF16 round (5vs4 witness). Engram-enabled boundaries cannot run HcPre on pre-gate hidden. Gate full-H norms/dot and HC full-H norm/sync are real dependencies.',
            'Keep updated residual publication and original HcPre HF32 coefficient/A2 FLOAT input GM path. A3 Vector->Cube workspace cannot be declared gone from fusion name alone.'],actual_gain=None),
      dict(id='F25-DIRECT-HIDDEN-SLICE',state='CONDITIONAL_SOURCE_COMPATIBLE_OUTPUT_BANK_REWRITE',
        graph='Gate final BF16 stores directly into original hidden[:n] destination instead of private temp then source slice-copy',
        conditions=gates+['All original hidden rotation/gate/residual readers complete before overwriting affected slots. Crossprogram full-H reduction and split view/capture aliases constrain write ordering.',
            'Saved160MEMCPY_ASYNC records are structurally between gate Cast and HcPre but have no address/length; source slice-copy is not an exact native pointer-binding proof.'],
        same_copy_credit_as_previous_ENG_FINAL_STORE=True,actual_gain=None)]
    write(O/'Engram-WKV-hoist-and-memo-variants-v84.json',dict(variants=variants,common_conditions=gates,
        no_new_maincandidate=True,conditional_operator_analysis_deliverable=True,implementation=False,native_test=False,
        actual_hit_rate=None,lookup_ready_time_or_free_hoist_window=None,overall_complete=False))
    wkv_regimes=read(O/'Engram-HC-native-frontiers-v84.json')['WKV_observed_regimes']
    observed_M={r['M']for r in wkv_regimes}
    models=[];weight_bytes=6144*25600*2;input_per_row=6144*2;output_per_row=25600*2
    for count in [2,3,800,816,830,834]:
        models.append(dict(M=count,observed_WKV_M_in_saved_profile=count in observed_M,
            WKV_dense_BF16_ops=2*count*6144*25600,WKV_weight_payload_bytes=weight_bytes,
            BF16_WKV_result_bank_bytes=output_per_row*count,
            two_layer_full_hoisted_result_banks_bytes=2*output_per_row*count,
            raw_KV_plus_key_rstd_bytes=(output_per_row+16)*count,
            FP32_key_plus_value_view_retaining_full_KV_bytes=(81920+51200+16)*count,
            FP32_key_plus_independent_value_and_rstd_bytes=(81920+10240+16)*count,
            value_copy_to_retire_raw_KV_reference_read_write_bytes=20480*count,
            scalar_only_optional_extra_key_BF16_revisit_bytes=40960*count,
            scalar_only_optional_extra_key_widen_elements=20480*count,
            key_rstd_conventional_square_elements=20480*count,key_rstd_conventional_sum_additions=20476*count,
            key_rstd_conventional_mean_and_eps_ops=8*count,key_rstd_special_rsqrt_elements=4*count,
            current_gate_temp_to_hidden_copy_reference_bytes=4*count*4*5120,
            gate_E_to_HC_collapse_private_read_reference_budget=2*count*4*5120,
            HC_Y_write_read_private_interface_budget=4*count*5120,
            HC_coefficient_FLOAT_cast_GM_bridge_reference_bytes=8*count*4*5120,
            updated_hidden_BF16_publication_bytes=2*count*4*5120,
            actual_allocation_HBM_or_gain=None))
    memo=[]
    for count in [3,800,834]:
        for misses in sorted({0,1,count//2,count}):
            hits=count-misses
            base=weight_bytes+input_per_row*count+output_per_row*count
            proposed=(weight_bytes if misses else 0)+input_per_row*misses+2*output_per_row*misses+2*output_per_row*count+96*count
            memo.append(dict(M=count,distinct_misses=misses,hits=hits,
                baseline_conventional_ops=2*count*6144*25600,miss_conventional_ops=2*misses*6144*25600,
                full_uncompressed_weight_plane_required_in_miss_GEMM=bool(misses),
                baseline_cold_value_reference_bytes=base,declared_missbuf_assembly_cache_reference_bytes=proposed,
                reference_delta_bytes=proposed-base,
                weight_L1_B_only_warm_service_floor_bytes=max(0,weight_bytes-24*65536)if misses else 0,
                actual_hit_or_performance_observed=False))
    capacities=[dict(assumed_cache_budget_bytes=budget,minimum_explicit_entry_bytes=51200+96,
        maximum_full_key_raw_BF16_entries_before_extra_metadata=budget//(51200+96),
        maximum_entries_with_key_rstd_before_extra_metadata=budget//(51200+96+16),
        actual_available_memory_proved=False)for budget in [2**20,16*2**20,128*2**20]]
    write(O/'Engram-WKV-storage-and-work-models-v84.json',dict(models=models,memo_reference_models=memo,cache_capacity_reference_models=capacities,
        input_feature_bytes_per_row=input_per_row,result_bytes_per_row=output_per_row,full_ID_key_bytes_per_row=96,
        result_to_original_full_feature_input_payload_ratio=output_per_row/input_per_row,
        liveness_domain='Uncompressed complete raw WKV BF16 and/or FP32key arrays. Split-view model retains full combined KV while value view lives. Copying value into independent backing can release raw KV only after all original readers, with explicit copy cost. These are representation-specific live-byte models, not actual allocator/UB/HBM bounds.',
        memo_reference_domain='Currentlookup/collectives kept. EachcallhasMrows,Udifferentmisses,nointra-call duplicatecollapse. OnecompactmissGEMM writesUresults;assemblyreadseachmiss/hitonce,writesfreshMoutputandmisscacheinsertsfromsamepass. Key96M readcost included; hashing/metadata/pack/eviction/events extra. Alternate scatter/direct-bank/allmiss-bypass plans require separate costs.',
        memo_reference_formula='Base=W+a*M+b*M. Plan=1[U>0]*W+a*U+2*b*U+2*b*M+96*M, W=314572800,a=12288,b=51200. PartialhitsdoNOTproportionallyremovethefullweightplane. U0coldreference saving=W-(b-a+96)*M;actualcache/HBMlevelmatters.',
        local_service_domain='Conventionaluncompressed dense weightB throughL1→L0B,24AIC andatmostoneinitial24*64KiBgrant. IfU>0fullWmandatoryweightplane remains. 2201GM→L0/orientation/otheralgorithms areoutside this floor;noactualtimeorfmaxassertion.',
        shared_resource_bound='Usev83 validjointcapacitysupport T>=lambda.W/beta(lambda) pluslink/read/write/dependencycuts. HoistingalonekeepsW unchanged andextendsliveness; memo replacesdenseWKVwork onlyforcertifiedhits andaddslookup/assembly/cachework. Unknownphysicalbeta/B/fmaxremain symbolic.',
        original_native_time_not_removed_by_schedule_claim=True,new_actual_speedup=None,overall_complete=False))
    write(O/'Engram-HC-joint-frontier-models-v84.json',dict(
        original_dependency_graph=['lookup->TP/SP rows->WKV_BF16','current_hidden->originalFLOATrotation','WKV_BF16->key_rstd',
            '(originalFLOATrotation,rawkey,channel_weight,key_rstd,mask,value,current_hidden)->gate_E_BF16',
            'gate_E_BF16->copy_into_current_hidden_slice->HcPre',
            '(updated_hidden,old_external_pre_mix)->Y_BF16->norm_BF16_and_widened_rounded_FLOAT->router/shared/routed',
            'updated_hidden->HF32_AB_A2_new_coefficients','(expert_or_attention_output,updated_hidden,current_coefficients)->HcPost'],
        legal_hoist_frontier='WKV/key_rstd may be ready earlier once exact row/weight generation is ready; gate and both HcPre branches must wait for updated current hidden. Current native whole-HcPre completion/SyncAll needs new visibility contract for earlyY split.',
        conditional_unlimited_resource_gate_ready='max(T_hidden_ready+T_rotation_hidden_prep, T_lookup_rows_ready+T_WKV+T_key_rstd_if_prepared)+T_remaining_gate_join',
        conditional_unlimited_resource_post_ready='max(T_current_coefficients_after_updated_hidden, T_collapse_Y+T_norm+T_router_experts)+T_HcPost; max models are optimistic dependency models, NOT physical lower bounds or measured speedups unless scheduling/resource premises are proved.',
        lower_bound_policy='Use necessary workload/resource cuts and true dependency lower bounds. A feasible or unlimited-resource candidate schedule time must not be inverted into a universal throughput upper bound.',
        preserved_outputs=['updated hidden residual for currentHcPost','new pre/post/comb','any aux/fullHC snapshots','current lookup/metadata/collective publications unless separately closed'],
        copy_and_fusion_credit_rule='PreviousENG-FINAL-STORE budget andnew160source-compatiblememcpyfrontier refer tosamecopy;countonce. Gate->collapse readbudget andYinterface budget aredistinct. Mandatoryupdatedhidden publication andHF32FLOATCVworkspace remain in currentownership.',
        implementation=False,native_test=False,new_native_legality_or_gain=None,overall_complete=False))
    print(dict(variants=len(variants),models=len(models),memo_scenarios=len(memo),
        M3_raw_WKV_bank_bytes=153600,M800_raw_WKV_bank_bytes=40960000,
        BF16_gate_round_counterexample=[original,bypassed],preweight_key_counterexample=True,
        public_primary_API_pages=1,physical_cache_hits_or_speedup_proved=False))


if __name__=='__main__':main()
