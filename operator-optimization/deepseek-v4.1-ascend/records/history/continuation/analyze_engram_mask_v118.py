"""Backward operator demand through an inactive Engram gate, CPU/source analysis only."""
from pathlib import Path
from fractions import Fraction
import ast,hashlib,json,math,struct
from finalize_origin_coverage_v9 import N,O,W,read,write

def pin(p):
    p=Path(p);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())

def section(p,a,b):return dict(**pin(p),first_line=a,last_line=b,source=Path(p).read_text().splitlines()[a-1:b])

def fp32(x):
    try:return struct.unpack('<f',struct.pack('<f',x))[0]
    except OverflowError:return math.copysign(math.inf,x)

def fbits(x):return struct.unpack('<I',struct.pack('<f',x))[0]
def bf(bits):return struct.unpack('<f',struct.pack('<I',bits<<16))[0]
def inactive_residual(h,v):return fp32(h+fp32(0.0*v))

def main():
    assert not(O/'REVIEW-CHECKPOINT-v118.json').exists()
    native=W/'sampler-source-supplement/frozen-source-snapshot/vllm-ascend/vllm_ascend/ops/triton/engram_gate.py'
    root=W/'frozen-source-snapshot/vllm-ascend/vllm_ascend/models/deepseek_v41'
    common=root/'engram/common.py';layer=root/'engram/layer.py';model=root/'model.py'
    old=[O/'state-engram-regime-models-v62.json',O/'Engram-hash-UVA-and-gate-costs-v62.json',
        O/'Engram-HC-operator-source-contract-v84.json',O/'Engram-WKV-storage-and-work-models-v84.json',
        O/'Engram-WKV-hoist-and-memo-variants-v84.json',O/'sample-hash-source-contract-v114.json',
        O/'sample-hash-factor-and-lookahead-work-v114.json']
    # Resolve the existing v114 cost filename explicitly, without reading other projects' source.
    if not old[-1].exists():old.pop()
    write(O/'engram-mask-inputs-v118.json',dict(inputs=[pin(p)for p in [native,common,layer,model]+old],
        operator_questions=['What does a false gate mask actually make dead?',
            'Can key and value projection demands be separated without changing BF16 residual/special-value semantics?',
            'How do masked demand, early projection and content memoization share the same work budget?'],
        profile_first='Reuse five WKV shapes/208 executions from v62, and prior v62/v84/v114 answers. Only the local gate body and directly controlling caller are expanded.',
        no_remote_or_device_work=True,overall_complete=False))
    nt=native.read_text();ct=common.read_text();lt=layer.read_text()
    assert 'enable_fp_fusion=False'in nt and 'if tokens:'in nt
    assert 'gate = tl.where(active, gate, 0.0)'in nt and 'h + gate * value'in nt
    assert 'gate = gate.masked_fill(~token_mask.unsqueeze(-1), 0)'in ct
    assert 'kv = self.wkv(rows)'in lt and 'bias=False'in lt
    write(O/'engram-mask-source-contract-v118.json',dict(parts=[section(native,9,78),section(layer,17,57),
        section(common,46,72),section(model,1554,1597)],
        native_unrotated=dict(guard='rotation is None AND npu AND ENGRAM_UNROTATED_GATE capability',
            source_grid='tokens by HC branches; BLOCK=next_power_of_2(H),num_warps=4 is a source launch setting,not proof of physical NPU occupancy.',
            shape_checks='Hidden and KV inner stride must be1 or raise; zero-token output allocated but kernel launch skipped.',
            math='FP32 h/key/q/k; separate qw*kw then h*product then key; two full-H RMS,weighted dot,signbit via bitcast,sqrt(abs(dot).clamp_min1e-6),sigmoid; mask after all this; value shared across HC; separate residual multiply/add;BF16 output store.',
            source_5120_padding='BLOCK8192 holds only5120 valid channels. Padded lanes load0; no8192 necessary-work or physical bound inference.',
            actual_current_profile_hit=False),
        rotated_current='Common path rotates hidden inFP32 but residual uses original hidden; false mask overwrites gate with+0 after dot,then still multiplies shared value and adds hidden before BF16 cast.',
        source_body_expansion_not_new_native_binding=True,historical_native_loaded_ABI_precision_not_attested=True,overall_complete=False))

    regimes=[r for r in read(old[0])['regimes']if r.get('work_model',{}).get('source_role')=='WKV_COMBINED_KEY_AND_VALUE']
    assert len(regimes)==5 and sum(r['executions']for r in regimes)==208
    cohorts=[dict(regime_id=r['regime_id'],executions=r['executions'],T=r['work_model']['M'],D=r['work_model']['K'],N=r['work_model']['N'],
        signature=r['signature'],context=r['context'],measured_native_task_ns=r['measured_native_task_ns'],actual_mask_values_or_A=None)for r in regimes]
    write(O/'engram-mask-profile-scope-v118.json',dict(cohorts=cohorts,existing_executions=208,
        source_branch='Existing captured rotated ordinary bodies are prior evidence; unrotated fused kernel is a conditional alternative,not reassigned to these TASKs.',
        attribution='T is local captured descriptor geometry,not unique request/token count. Preserve32phase-unbound executions and TP replicas;no ordinal/mask-field identity join.',
        unresolved_BOOL24_not_assigned_to_Engram_mask=True,
        no_old_assumed_numeric_reference_floor_reused=True,no_actual_inactive_row_count_or_time_saving=True,new_native_promotions=0,overall_complete=False))

    # Exhaust every finite BF16 hidden value against eight finite value/sign classes.
    # This validates a scalar IEEE host reference, not the Ascend reduction/rsqrt ABI.
    vals=[0x0000,0x8000,0x0001,0x8001,0x3f80,0xbf80,0x7f7f,0xff7f]
    cases=0;diffs=0;digest=hashlib.sha256()
    for hb in range(65536):
        if (hb&0x7f80)==0x7f80:continue
        for vb in vals:
            got=fbits(inactive_residual(bf(hb),bf(vb)))
            expected=((hb<<16)if hb!=0x8000 or vb&0x8000 else 0)
            assert got==expected
            safe_bypass=(hb!=0x8000 or bool(vb&0x8000))
            assert (got==(hb<<16))==safe_bypass
            cases+=1;diffs+=not safe_bypass;digest.update(struct.pack('<HHI',hb,vb,got))
    assert cases==522240 and diffs==4
    witnesses=[]
    for hb,vb in [(0x8000,0x3f80),(0x8000,0xbf80),(0x3f80,0x7f80),(0xbf80,0xff80),(0x3f80,0x7fc1),(0x0000,0x8000)]:
        h=bf(hb);v=bf(vb);z=inactive_residual(h,v)
        witnesses.append(dict(hidden_BF16_hex=f'{hb:04x}',value_BF16_hex=f'{vb:04x}',
            result_FP32_hex=f'{fbits(z):08x}',result_is_NaN=math.isnan(z),
            raw_hidden_copy_equal_numeric_and_zero_sign=(z==h and (z!=0 or math.copysign(1,z)==math.copysign(1,h))),
            no_native_NaN_payload_or_error_equivalence_claim=True))
    write(O/'engram-mask-residual-reference-v118.json',dict(finite_BF16_hidden_by_eight_value_class_cases=cases,
        unsafe_raw_copy_cases=diffs,result_digest=digest.hexdigest(),witnesses=witnesses,
        exact_false_mask_expression='roundBF16(addFP32(float(hidden),mulFP32(+0,float(value))))',
        finite_scalar_bypass_predicate='Given finite BF16 h and finite BF16 v under separate IEEE RNE ops, raw BF16 h copy is exact iff h is not -0 OR signbit(v)=1.',
        row_certificate='Sufficient: every hidden channel is finite and not-0,and every projected value channel is finite. A weaker per-channel signed-zero predicate may also be certified. No such actual workload certificate recovered.',
        special_value_caution='0*Inf and0*NaN makeNaN even for inactive gate; hidden-0 plus+0 can become+0. Do not claim NaN bit payload or hardware FPstatus equivalence from host checks.',
        prior_v62_not_new='v62 already warns that deadmask gate0 does not authorize returning hidden. New work is exact residual predicate and backward key/value demand/cost composition.',
        native_precision_test=False,overall_complete=False))

    Tset=sorted({r['T']for r in cohorts});models=[]
    D,H,C=6144,5120,4
    for T in Tset:
        for A in sorted({0,1,T//2,T}):
            assert 0<=A<=T
            for B in sorted({0,T-A}):
                # B is inactive rows with a separately verified complete bypass certificate.
                R=T-B
                out=H*(C*A+R);base=H*(C+1)*T
                models.append(dict(T=T,active_gate_rows=A,certified_bypass_inactive_rows=B,
                    value_rows=R,key_rows=A,baseline_dense_MAC=D*base,demand_dense_MAC_reference=D*out,
                    selected_GEMM_MAC_ratio=str(Fraction(out,base)),
                    baseline_private_BF16_KV_elements=base,demand_private_BF16_KV_elements=out,
                    baseline_BF16_KV_bank_bytes=2*base,demand_BF16_KV_bank_bytes=2*out,
                    weight_unique_cold_payload_bytes=2*D*H*((C if A else 0)+(1 if R else 0)),
                    separate_value_then_active_key_input_interface_bytes=2*D*(R+A),
                    active_full_then_inactive_value_input_interface_bytes=2*D*R,
                    separate_value_then_active_key_weight_call_payload_bytes=2*D*H*((C if A else 0)+(1 if R else 0)),
                    active_full_then_inactive_value_weight_call_payload_bytes=2*D*H*((C+1 if A else 0)+(1 if R>A else 0)),
                    lookup_feature_rows_still_demanded=R,
                    output_BF16_hidden_publication_bytes=2*T*C*H,
                    raw_copy_bypass_not_bit_proved_without_certificate=True))
    write(O/'engram-ma<REDACTED_TOKEN>.json',dict(models=models,
        variables='Tinputrows,Atrue-maskrows, Bseparatelycertifiedinactivefull-bypassrows (0<=B<=T-A),R=T-Bvalue-demandrows,D6144,H5120,C4.',
        conventional_dense_formula='MAC=D*H*(C*A+T-B);BF16privateKVvalues=H*(C*A+T-B). Baseline=D*H*(C+1)*T. With no bypass certificate B=0,ratio=(T+4A)/(5T).',
        all_inactive_not_all_dead='A=0,B=0 retains value projection/lookup and exact masked residual; it may omit key columns/rotation/dot/norm only under the declared no-other-observer contract.',
        cold_weight_discontinuity='Any A>0 still demands all four key weight blocks plus the value block ifR>0:314572800B dense weight payload. Only A=0 removes the four key blocks:62914560B forvalue. These are unique cold logical contents,not measured HBM.',
        split_schedule_tradeoff='Value(Rrows,Hcols)+key(Arows,4Hcols) has one logical weight request perblock but R+A input row references. Full(Arows,5Hcols)+value(R-Arows,Hcols) references each input row once but requests the value weight block twice when0<A<R. A fused conditional row kernel may avoid these interface duplicates,with new tiling/resource/precision costs.',
        profile_T3_A1_reference='ForT3,A1,B0,the dense MAC ratio is7/15. Full-active+inactive-value splits can request377487360B weight content percall versus314572800B baseline (1.2x),while separate-value+active-key reads4 ratherthan3 logical feature rows. Neither is actual HBM or measuredtime.',
        dense_domain='Conventional row-independent dense dot evaluation of exactly demanded key/value outputs;no sparsity,lowrank,alternativeclassificationalgorithm or caches assumed. This is not a universal algorithmic lowerbound.',
        classification_alternative='Without full-value observers,inactive output needs special-value/sign behavior rather than an independently materialized full value. Cheaper certified summary/classification representations are allowed;no full GEMM necessity theorem asserted.',
        output_publication='All T rows still reach downstreamHC,residual,aux readers even when key computation is removed. A fully certified in-place no-op may change publication only after alias/event/reader checks.',
        physical_time_lower_bound='For any independently proved necessary workQ_e and typed capacity upper boundU_e use T>=max_e(Q_e/U_e). The dense reference above does not supply Q_e for all allowed algorithms. ActualA/B/capacities unavailable;no numeric new floor or MFU.',
        actual_A_B_traffic_or_speedup_proved=False,overall_complete=False))

    write(O/'engram-ma<REDACTED_TOKEN>.json',dict(
        demands=[dict(mask=False,key_columns=False,key_norm_dot_rotation=False,value='Retain exact residual,or separatelycertified no-op/summary.',lookup='Retain ifvalue needed.',history='Retain allcache/dead/pad/requestepoch updates demanded by later tokens.'),
            dict(mask=True,key_columns=True,key_norm_dot_rotation=True,value='Retain.',lookup='Retain.',history='Retain.')],
        v84_hoist='If full WKV was already hoisted,late mask discovery cannot retrospectively save its work. Choose early full compute vs demand-sensitive compute after mask readiness; possible critical-path overlap and extra work are not additive credits.',
        v84_memo='A value-only cache entry is not a full KV hit. Tag exact content,table/weight/quantizer/layout/precision generation and which key columns/rstd are valid;later true-mask reuse must compute missing keys. Fixed retained full KV cannot also claim compact bank retirement.',
        v114_hash='Inactive token can block hash and yield pad IDs, but commit/rollback/DEAD cache state and all other layers/readers remain. Removing a gate lookup row never automatically removes history writes or global readiness/collective participation.',
        F24_F25_partition='Value-only still reads the same6144feature row in the dense domain. Nolookup/UVArow deletion credit unlessvalueoutputorcertification andallotherreadersalsoallowit. Row/head alternatives need one selected owner/communication plan.',
        admission_guards=['Shape/stride/device/capability and preexisting errors unchanged;zero-token branch retained.',
            'No other key/KV/rstd/rotation/RNG/FPstatus/telemetry/hook/error reader. Skipped arithmetic or memory faults must not be observable.',
            'Keep baseline BF16 WKV and residual rounds,normalizer order,mask exact+0,signbit/NaN behavior,valid output coverage and all capture/alias epochs.',
            'Changing active-M or selecting columns must preserve value bits or pass an explicitly accepted numerical contract;native GEMM tiling/FP32-HF32/quantizer output-wide reductions can couple shape/channels.',
            'Mask readiness/packing/scatter/collective count/zero tails/receiver ordering and branch-capability/source binding must be concrete before claiming runtimegain.',
            'Finite bypass certificates must cover actual projected values and hidden,not just input finiteness;finite inputs can overflow projection. No such certificate currently exists.'],
        certificate_cost='Certificate construction,hidden/special-value scans,version checks,storage and readiness must be charged. If projected value was computed to certify it,the projection work was already spent;Bcannot retroactively save it. A proven static domain certificate may differ from runtime screening.',
        invalid_rewrites=['falsemask=>returnhiddenforallBF16','falsemask=>deleteallWKV/hash/cachehistory','A/TMACratio=>HBMorcriticalpathspeedup','Apositive=>coldweightsreduceproportionallywithA',
            'memoizedvalue=>allKVhit','fullWKVhoist+samedemandremovedwork counted twice','num_warps4/BLOCK8192=>physicalNPUoccupancy/necessary8192work'],
        new_pairs=0,new_groups=0,new_native_promotions=0,whole_module_closed=False,overall_complete=False))
    print(json.dumps(dict(status='ANALYZED',reused_WKV_executions=208,finite_scalar_references=cases,
        demand_models=len(models),native_unrotated_body_expanded=True,actual_mask_counts_bound=False,new_device_work=False)))

if __name__=='__main__':main()
