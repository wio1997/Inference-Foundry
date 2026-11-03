"""Shared-expert exact operator paths, traffic formulas and fusion interaction constraints."""
from pathlib import Path
from datetime import datetime,timezone
from collections import defaultdict,Counter
import ast,hashlib,json,math,statistics,sqlite3
from finalize_origin_coverage_v9 import N,O,read,write


def stat(a):
    a=sorted(a);return dict(n=len(a),minimum=a[0],median=statistics.median(a),p90=a[int(.9*(len(a)-1))],maximum=a[-1])


def shape(s):return [[int(x)for x in v.split(',')if x]for v in s.strip('"').split(';')]


def main():
    assert not(O/'shared-expert-shape-ceiling-v57.json').exists()
    stamp=datetime.now(timezone.utc).isoformat();inputs=read(O/'shared-operator-inputs-v57.json')
    caller=next(r for r in inputs['files']if r['status']=='READ'and r['path'].endswith('shared_experts.py'))
    p=Path(caller['local_path']);assert hashlib.sha256(p.read_bytes()).hexdigest()==caller['sha256']
    text=p.read_text();tree=ast.parse(text);ls=text.splitlines();parts=[]
    for f in ast.walk(tree):
        if isinstance(f,ast.FunctionDef)and f.name in ['enqueue_gate_up_after_router','_run_a8_int_mlp']:
            parts.append(dict(name=f.name,first_line=f.lineno,last_line=f.end_lineno,source=ls[f.lineno-1:f.end_lineno]))
    assert len(parts)==2
    headers=[r for r in inputs['files']if r['status']=='READ'and r['path'].endswith('.h')]
    h={Path(r['path']).name:Path(r['local_path']).read_text()for r in headers}
    gmm=h['aclnn_grouped_matmul_swiglu_quant_weight_nz_v2.h'];dsq=h['aclnn_dequant_swiglu_quant_v2.h']
    assert 'clampLimit'not in gmm and 'swigluMode'not in gmm and 'clampLimit'in dsq and 'swigluMode'in dsq
    config_index=read(O/'sampling-guard-inputs-v55.json');cfgrow=next(r for r in config_index['files']if r['path'].endswith('/config.json'))
    cfg=read(Path(cfgrow['local_path']))['text_config'];assert cfg['hidden_size']==5120 and cfg['moe_intermediate_size']==2304 and cfg['swiglu_limit']==10.0
    # Do not convert opaque profiler attr_N positions into semantic names using API argument order.
    write(O/'shared-operator-numerical-and-library-contract-v57.json',dict(generated_utc=stamp,
        caller=dict(path=str(p),sha256=caller['sha256'],parts=parts),SDK_headers=headers,
        model_text_config=dict(path=cfgrow['local_path'],sha256=cfgrow['sha256'],H=5120,I=2304,swiglu_limit=10.0),
        profile_typed_path='BF16→DynamicQuant(INT8,FLOATscale)→GEMM1 INT32→DequantSwigluQuant(INT8,FLOATscale)→GEMM2 BF16→BF16merge.',
        numerical_constraints=['SharedGEMM1 INT32 boundary differs fromroutedMC2 FP16 intermediate;do not reuseoneprecisionmodel blindly.',
          'Caller preservesINT32gateup,FP32weight/pertokenscales,activate_leftTrue,quant_mode1,swiglu_mode1,clamp_limit andoptionalglu tuningargs.',
          'Per-token dynamic quantization has oneoutputscale perrow;fullrow scale/reduction precedes finalINT8values. Per-tileindependent scales change the operator.',
          'FusingDown withmerge mustpreserve the existingBF16Down rounding beforeBF16addition;it can be on-chip if allreaders/ownership/readiness permit.',
          'GEMM1 scale operand is stilla descriptor;this analysis doesnot assume its compulsoryreads disappear merelybecauseoutput isINT32. Token-scale application is deferred in thecaller.'],
        installed_fused_API=dict(name='aclnnGroupedMatmulSwigluQuantWeightNzV2',header_present=True,
          matching_domain='INT8 x/weightNZ,FP32row/weightscales,INT8out+rowScale;singleexpert requiresvalidonegroupmetadata/layout/typeadaptation.',
          exposed_clamp_or_swiglu_mode=False,drop_in_equivalence_proved=False,
          conclusion='PublishedstandardSwiGLUformula andinstalledV2signature do not supply currentclamp10/mode1 contract. A direct replacement isnot established;clippedkernel/customvariant mustpreserve these semantics.',
          no_claim_all_library_variants_absent=True,active_kernel_or_python_binding_not_executed=True),
        clamp_counterexample=dict(kind='Independentreal-valuedsemanticwitness,notcurrenttensor/native test.',gate=20,linear=1,limit=10,
          clipped=10/(1+math.exp(-10)),unclipped=20/(1+math.exp(-20)),
          observation='Evenifone-elementINT8q is127bothways,returnedquantScale differs andchangesDown dequantization.'),
        tile_scale_counterexample=dict(activation=[1,100],fullrow_scale=100/127,fullrow_quant_RINT=[1,127],per_tile_quant=[127,127],
          observation='A single rowScale cannot in general reconstructthefullrow result fromindependentlyquantizedtiles.'),
        safe_INT32_dot_bounds=dict(max_INT8_product_abs=16384,K_up=5120,absolute_partial_sum_bound_up=83886080,
          K_down=2304,absolute_partial_sum_bound_down=37748736,INT32_max=2147483647,
          scope='Conventionalintegerdot withoutbias;allpartial sums fitINT32. SubsequentFP32dequant/activation/roundorder stillmust match.'),
        public_primary_references=['https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/aolapi/context/ops-transformer/aclnnGroupedMatmulSwigluQuantWeightNzV2.md','https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/aolapi/context/ops-nn/aclnnDequantSwigluQuantV2.md'],
        website_limitation='Openreturnedclientnavigation;indexedofficialtextprovidedstandardformula/operatorpurpose. Installedheadersandpinnedcaller usedforcurrentsignature;noBrowserused.',
        current_loaded_object_or_binary_identity_proved=False,implementation=False,overall_complete=False))
    led=read(O/'operator-shape-ceiling-ledger-v55.json');regs=[r for r in led['observed_regimes']if r['module_candidates']==['M13']]
    assert len(regs)==42 and sum(r['executions']for r in regs)==22360
    kinds=Counter();rows=[];Ms=set();H=5120;I=2304;P=752e12;B=1.6e12
    for r in regs:
        sig=r['signature'];ins=shape(sig['inputShapes_text']);outs=shape(sig['outputShapes_text']);m=ins[0][0];Ms.add(m);op=r['operator'];flops=None;q=None
        if op=='DynamicQuant':
            assert ins[0]==[m,H]and outs==[[m,H],[m]]
            kind='INPUT_QUANT';q=3*m*H+4*m
        elif op=='DequantSwigluQuant':
            assert ins[0]==[m,2*I]and outs==[[m,I],[m]]
            kind='ACTIVATION_QUANT';q=9*m*I+8*I+8*m
        elif op=='ForeachAddList':
            assert ins[:2]==[[m,H],[m,H]]and outs==[[m,H]]
            kind='MERGE';q=6*m*H
        elif op=='QuantBatchMatmulV3':
            assert math.prod(ins[1])==ins[0][1]*outs[0][1]
            if sig['outputDataTypes_text']=='INT32':
                assert ins[0]==[m,H]and outs[0]==[m,2*I]
                kind='GATE_UP';flops=4*m*H*I;q=m*H+2*H*I+8*m*I
            else:
                assert sig['outputDataTypes_text']=='DT_BF16'and ins[0]==[m,I]and outs[0]==[m,H]
                kind='DOWN';flops=2*m*H*I;q=m*I+H*I+2*H+4*m+2*m*H
        else:raise AssertionError(op)
        kinds[kind]+=r['executions'];t=r['measured_native_task_duration']['median']/1000
        rows.append(dict(regime_id=r['regime_id'],stage=kind,M=m,origin=r['origin'],context=r['context'],executions=r['executions'],
            conventional_dense_INT8_ops_2perMAC=flops,one_read_write_logical_model_bytes=q,
            conditional_cold_reference_floor_us=max(q/B*1e6,(flops/P*1e6)if flops else 0),
            vector_compute_floor_unbound=flops is None,profile_median_us=t,
            qualification='LogicalIO/densealgorithm atreference752TOPS/1.6TB/s,coldHBMassumed;notactualtraffic/capacity/efficiency. GateUp possibleextraScale reads are unpriced;extraoperatorworkspace/tiling excluded.'))
    assert kinds=={'INPUT_QUANT':4472,'GATE_UP':4472,'ACTIVATION_QUANT':4472,'DOWN':4472,'MERGE':4472}
    models=[]
    for m in sorted(Ms):
        interfaces=dict(input_quant_to_G1=2*m*(H+4),G1_INT32_to_DSQ=16*m*I,DSQ_to_G2=2*m*(I+4),G2_BF16_to_merge=4*m*H)
        weights=3*H*I;scales=8*I+2*H
        base=weights+scales+12*m*H+18*m*I+16*m
        fused=weights+scales+6*m*H
        assert base-fused==sum(interfaces.values())
        models.append(dict(M=m,H=H,I=I,INT8_GEMM_ops=6*m*H*I,weights_bytes=weights,semantic_weight_scales_bytes=scales,
            materialized_chain_logical_bytes=base,ideal_interface_eliminated_bytes=fused,removable_interfaces=interfaces,
            total_removed_interface_bytes=base-fused,
            materialized_cold_reference_io_us=base/B*1e6,ideal_fused_cold_reference_io_us=fused/B*1e6,
            reference_compute_us=6*m*H*I/P*1e6,removed_interface_reference_transfer_us=(base-fused)/B*1e6,
            scale_descriptor_addendum='GateUp BF16scale input has4Ibytes extent;notassumedsemanticallydead orreadfree. It isnot included inmandatorylowerIOmodel.',
            qualification='Idealallinterfaceelimination withonchipcapacity/nativeprecision/allconsumerconditions,not anavailableorbenchmarkedkernel.'))
    prev=read(O/'MC2-operator-profile-v56.json');stage_times=defaultdict(list);chains=[]
    for row in prev['root_bound_graph_layers']:
        stages=row['shared_stages'];names=['INPUT_QUANT','GATE_UP','ACTIVATION_QUANT','DOWN','MERGE']
        for name,s in zip(names,stages):stage_times[name].append(s['duration_ns'])
        prep=stages[2]['end']-stages[0]['start'];route=row['route_duration_ns']
        chains.append(dict(database_index=row['database_index'],graph_ordinal=row['graph_ordinal'],block_ordinal=row['block_ordinal'],
            pre_down_span_ns=prep,full_shared_premerge_span_ns=row['shared_premerge_span_ns'],
            full_branch_unlimited_overlap_cap_ns=min(route,row['shared_premerge_span_ns']),
            delayed_down_for_fused_merge_overlap_cap_ns=min(route,prep)))
    write(O/'shared-expert-shape-ceiling-v57.json',dict(generated_utc=stamp,all_captured_M13_regimes=rows,
        regime_count=42,captured_stage_executions=dict(kinds),logical_shape_models=models,
        root_bound_graph_stage_times_ns={k:stat(v)for k,v in stage_times.items()},root_bound_graph_chains=chains,
        pre_down_span_ns=stat([r['pre_down_span_ns']for r in chains]),
        fusion_and_overlap_tradeoff='FusedDown+Merge needsroutedoutputreadyatDown epilogue. FullsharedoverlapcanfinishDownbeforethat;thenretainbuffer/keepoccupancy or delayDown. Do not add fullbranch57usoverlapcap toBF16Down-bufferelimination asindependentgains.',
        formulas=['2GEMM integerops=6MHI;weights=3HIbytes.',
          'MaterializedsemanticchainQ=3HI+(8I+2H)+12MH+18MI+16M,excludingG1possibleextraBF16scale reads.',
          'IdealallinterfaceseliminatedQ=3HI+(8I+2H)+6MH;delta=6MH+18MI+16M.',
          'Tref>=max(INT8ops/Pint8,Q_HBM/B_HBM,Q_L2/B_L2,quant/activation/reductiondependencybounds) onlyunderdeclarednecessarywork/cache/capacity assumptions.'],
        performance_gain=None,implementation=False,native_test=False,whole_module_closed=False,overall_complete=False))
    variants=[
        dict(id='F22-IN',proposal='Fuse inputdynamicquant withGateUp',condition='OnefullHrow scale beforeINT8GEMM;reusequantrow/scale acrossNtiles,avoid unpriced duplicatequantization andpreservecurrentquantizer.'),
        dict(id='F22-UP',proposal='Fuse INT32GateUp,dequant/SwiGLU androwquant',condition='PreserveINT32→FP32 andclamp10/mode1/alltuningargs. FullIrowamax barrier/crosscolumncoordination required;3smallMrows canloseCubeparallelism ifonecoreperrow.'),
        dict(id='F22-SPLIT',proposal='GEMMepilogue producesFP32SwiGLUtiles+amax;finishrowquant separately',condition='Conditionalfinite/NaN/zero-aware maxcomposition;FP32activationwrite/read8MI versusINT32gateupwrite/read16MI,sohalve thisinterface butretain a rowbarrier andpriceextra storage/reduction.'),
        dict(id='F22-LIB',proposal='Reuse installedGroupedMatmulSwigluQuantWeightNzV2',condition='Header/dtype/shape candidateonly. Currentclamp10/mode1 not exposed;unclampedpublishedformula failswitness. NeedcompatibleclampedAPI/kernel or anexplicitdifferent/unclampeddomain;not currentdrop-in.'),
        dict(id='F23-DOWNADD',proposal='Downprojection epilogue+shared/routedadd',condition='RoundDown toBF16onchip beforeadd/finalBF16round;allotherreaders/hooks/row/transforms androutedtilereadiness retained.'),
        dict(id='F21-F22-QCOMMON',proposal='Commonquantizedinput forshared androutedbranches',condition='Firstprove sameinputgeneration/rowmap/mask andsamequantizer/scale/round/zero semantics;MC2currentBF16input ABI needsnewprequantizedroutingcontract. SameINT8dtypeisnotsemanticidentity.')]
    write(O/'shared-expert-fusion-variants-and-interactions-v57.json',dict(generated_utc=stamp,variants=variants,
        interactions=[dict(a='F22-UP',b='F22-LIB',relation='ALTERNATIVES_WITH_CLAMP_COMPATIBILITY_GATE'),
            dict(a='F23-DOWNADD',b='F22-OVERLAP-v56',relation='READINESS_AND_BUFFER_TRADEOFF;NOT_FREE_ADDITIVE'),
            dict(a='F22-IN',b='F21-F22-QCOMMON',relation='SHARED_QUANT_PRODUCER_OR_DUPLICATED_WORK;PRICETOGETHER'),
            dict(a='F22-UP',b='F22-SPLIT',relation='ALTERNATIVES_FOR_ROW_REDUCTION_AND_STORAGE'),
            dict(a='F23-DOWNADD',b='F23-MERGE-HC',relation='PRESERVE_BF16_ROUNDING_AND_HC_DEPENDENCIES_FROMv22')],
        no_new_maincandidate=True,measured_speedup=None,implementation=False,whole_module_closed=False,overall_complete=False))
    print(json.dumps(dict(regimes=42,each_stage_executions=4472,root_bound_chains=len(chains),M_domains=sorted(Ms),
        root_stage_ns={k:stat(v)for k,v in stage_times.items()},pre_down_ns=stat([r['pre_down_span_ns']for r in chains]),variants=len(variants),overall_complete=False)))


if __name__=='__main__':main()
