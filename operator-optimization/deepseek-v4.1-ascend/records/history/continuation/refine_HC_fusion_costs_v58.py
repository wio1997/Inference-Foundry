"""Tie captured regimes to conditional costs and the delayed-collapse dependency DAG."""
from finalize_origin_coverage_v9 import N,O,read,write

def main():
    assert not (O/'operator-regime-refinement-v58.json').exists()
    p=read(O/'HC-work-traffic-and-reference-bounds-v58.json')
    for r in p['models']:
      m,h,H=r['M'],r['h'],r['H']
      r['split_Y_norm_value_bytes']=dict(attention=2*m*h*H+4*m*h+2*H+2*m*H,
        FFN_with_live_BF16_and_FLOAT_planes=2*m*h*H+4*m*h+2*H+6*m*H,
        qualification='Value IO of isolated collapse+norm branch, BF16gamma. Does not price coefficientbranch, normrstd extraoutput/otherreaders, cache tiers, nativeworkspaces or synchronization.')
    p['split_dependency_reference']=dict(
      current='WholeHcPre completion gates RMSNorm/attention-or-MLP evenwhen Y onlyusesexternalpre_mix.',
      conditional_split_DAG='R,oldPre→Y_BF16→Norm→A; R→currentCoeff; (A,R,currentPost,currentComb)→HCPost. newPre remains live fornextHC.',
      unlimited_resource_critical_path='max(T_currentCoeff,T_Y+T_norm+T_attention_or_MLP)+T_HCPost, beforelaterdependencies; no numericgain because internalC/Yservice times andsharedresource costs areunresolved.',
      input_traffic='Currentoriginalkernelalready readsR separatelyforcoefficientinput andcollapse. Splittingalone neednotadd a thirdRread;physicalcache/locality andlaunch/barriercostscan stillchange.',
      no_claim_coefficient_work_deleted=True)
    write(O/'HC-work-traffic-and-reference-bounds-v58.json',p)
    prof=read(O/'HC-operator-profile-and-edges-v58.json')
    ms={r['M']:r for r in p['models']};rows=[]
    for r in prof['regimes']:
      assert r['M'] in ms
      rows.append(dict(regime_id=r['regime_id'],operator=r['operator'],M=r['M'],executions=r['executions'],
        shape_model='HC-work-traffic-and-reference-bounds-v58.json#M='+str(r['M']),
        fusion_contract='HC-operator-numerical-and-dependency-contract-v58.json',
        conditional_variants='HC-fusion-split-reorganization-variants-v58.json',
        conditional_analysis_status='CAPTURED_OPERATOR_FORMULA_WORK_LOGICAL_IO_AND_CONDITIONAL_FUSION_REVIEWED',
        full_numeric_hardware_ceiling_bound=False,native_fusion_proved=False,
        remaining='HF32/A2/Vectorphysicalrates/cachetiers/internalphasecost andnewkernelallconsumer/precision/readiness requirements explicitlyunbound. Nativeimplementationabsence alone isnotanunanalysedconditionalproposal.'))
    assert len(rows)==16 and sum(r['executions'] for r in rows)==17888
    write(O/'operator-regime-refinement-v58.json',dict(parent='operator-shape-ceiling-ledger-v55.json',
      observed_total_regimes=2082,refined_regimes=rows,refined_regime_count=16,refined_execution_count=17888,
      meaning='Incremental operator-specific evidence links; 16reviewedregimes isnot16closedmodules orcompletedtypedphysicalceilings. Otherregimesretainexistingreviews andremaininggaps.',
      framework_semantic_inventory_changed=False,whole_modules_closed=0,overall_complete=False))
    write(O/'HC-hardware-reference-scope-v58.json',dict(
      sources=[dict(url='https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/ascendcopapi/docs/en/api/SIMD-API/C-API/cube_compute/asc_enable_hf32.md',
        claim='Official API indexedtext: HF32mode roundsL0A/L0B FLOAT32 operands toHF32 beforeMmad onA2/A3. Supports precisiondistinction, not anumericpeak.'),
        dict(url='https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/910/devaids/atctool/atlasatcparam_16_0077.html',
        claim='Official indexedtext distinguishes HF32internalmantissaprecision fromFP32. It doesnotbindcurrentcustomkernelpeak.')],
      limited_primary_peak_lookup='Queries910C HF32FLOPS/峰值算力 and910B/C HF32Vectorpeak foundnoprimary sourcebindingper-die numericHF32,A2Gram andVectorrates tothecurrentvariant. Secondary tables notused.',
      no_assumed_FP32_to_BF16_peak_ratio=True,compute_capacity_remains_unbound=True,
      next_action='Carrytypedrateparametersforward;do notrepeatbroadhardwaresweeps orsubstitutewrongprecision. Additionalcapacityevidence onlyifit changesnamedoperatorchoice.',overall_complete=False))
    print('16 HC regimes / 17888 executions linked to source-conditioned models; DAG/precision/hardware limits recorded.')

if __name__=='__main__':main()
