"""Separate A3 CV,CC,VV fusion domains and reprice shared-MLP interfaces."""
from collections import Counter
from copy import deepcopy
from finalize_origin_coverage_v9 import O,read,write


def main():
    assert not(O/'A3-fusion-interface-frontiers-v80.json').exists()
    # Navigationcoverage forall35candidates;notbinary/codepath attestation.
    flags={
      'F01':['CV','VV'],'F02':['VC','VV'],'F03':['VV','VC'],'F04':['VV'],
      'F05':['VV','VC'],'F06':['CC_SIBLING_INPUT'],'F07':['CV'],'F08':['CV','VV'],
      'F09':['CV','VV'],'F10':['CV'],'F11':['VV'],'F12':['VV','INTERNAL_CV'],
      'F13':['CV','VC'],'F14':['CV','VV','COMM_C'],'F15':['COMM_C','VC'],
      'F16':['CC','COMM_V'],'F17':['VC','CC_SIBLING_INPUT','CV'],'F18':['VV'],
      'F19':['VV','FANOUT'],'F20':['CONDITIONAL_CV','VV','COMM_V'],
      'F21':['CV','VC','COMM_V'],'F22':['VC','CV','VC','CV'],
      'F23':['CONDITIONAL_CV','VV'],'F24':['HOST_UVA','VV'],
      'F25':['CV','VV'],'F26':['VV','VC'],'F27':['CV','COMM_V'],
      'F28':['VV','COMM_V'],'F29':['CV','VV'],'F30':['VC','CV','RECURRENCE'],
      'F31':['METADATA_STATE'],'F32':['COMM_C','CC_SIBLING_INPUT'],
      'F33':['CC_SIBLING_INPUT','CV'],'F34':['STATE','VV'],'F35':['CV','VV']}
    candidates=read(O/'candidate-profile-ledger-v79.json')['candidates'];assert set(flags)=={r['id']for r in candidates}
    rows=[dict(candidate_id=r['id'],name=r['name'],frontier_kinds=flags[r['id']],
        state='OPERATOR_FRONTIER_APPLICABILITY_INDEX;NOT_ALL_VARIANT_OR_RUNTIME_PATH_PROOF',
        rule='CV/VC:underCube+Vectorownershiponarch2201retainGMbridgeunlessdifferentlegalproducer/consumerAPI/algorithmproved. CC:localL0C→L1possiblewithsameAIC/tile/dtype/readiness. VV:sameAIVUBpossiblewithcapacity/fanout/roundingcontracts. Comm/UVA/stateboundaries retainownobligations.',
        whole_candidate_closed=False,actual_native_or_new_fused_path_validated=False)for r in candidates]
    write(O/'A3-fusion-interface-frontiers-v80.json',dict(candidates=rows,candidate_count=35,
        frontier_definition=dict(CV='Cube-ownedproducedvalue→Vectorconsumer',VC='Vector-ownedproducedvalue→Cubeconsumer',CC='Cube→CubewithinthedeclaredAICtile',VV='Vector→VectorwithinthesameAIVtile'),
        current_CANN_9p2_native_all_paths_attested=False,
        proof_scope='Physicalarch2201CVpathconstraint andconditionalimplementationdomains. Candidateflagslocatetherelevantoperandfrontierusingexistingoperatorcontracts;notfullsemanticproofbynameorindex.',
        excluded_generalizations=['OnekerneldoesnotimplyzeroGMorzeroHBM;GMandHBMaredifferenthierarchylevels.',
            'GetTensorCUBoutputdoesnotbyitselfproveadirectL0C→AIVUBwire. New3510UB/L1orL0C/UBandSSBufferpathsnotborrowedforA3.',
            'Noallcandidatefusionexclusion:changeownership,Fixpipe/atomicepilogue,Vector/scalarownedalgorithm,orotherrepresentations requiretheirownsemantics/capacitybounds.',
            'UB/L1payloadchecksapplytoaspecificsimultaneous-residencyplan;tiling/streaming/replicationareotherdomains,notexcludedbywhole-tensoroversize.'],
        overall_complete=False))
    H,I=5120,2304;models=[]
    oldmodels=read(O/'shared-expert-shape-ceiling-v57.json')['logical_shape_models']
    for m in [2,3,800,816,834]:
        old=next(r for r in oldmodels if r['M']==m)
        cuts=dict(input_quant_INT8_VtoC=2*m*H,G1_INT32_CtoV=16*m*I,activation_quant_INT8_VtoC=2*m*I,Down_BF16_CtoVector_merge=4*m*H)
        total=sum(cuts.values());assert total==6*m*H+18*m*I
        assert old['total_removed_interface_bytes']==total+16*m
        models.append(dict(M=m,H=H,I=I,prior_v57_ideal_all_interface_value_budget_bytes=old['total_removed_interface_bytes'],
            full_value_store_plus_read_GM_bridges_in_CV_ownership_domain=cuts,GM_bridge_reference_sum=total,
            prior_scale_interface_addendum_bytes=16*m,
            split_activation_interface=dict(prior_external_FP32_activation_write_read_bytes=8*m*I,retained_internal_G1_INT32_CV_bridge_bytes=16*m*I,
                combined_internal_plus_external_reference_bytes=24*m*I,baseline_G1_to_DSQ_reference_bytes=16*m*I,
                added_reference_bytes_if_both_materialized=8*m*I),
            HBM_traffic_necessary_from_GM_bridges_proved=False,GM_bandwidth_numeric_upper_known=False,actual_latency_gain=None))
    write(O/'shared-CV-interface-revaluation-v80.json',dict(models=models,
        declared_domain='A3arch2201,conventionalCubeGEMM1/2,VectorDQ/DSQ/merge,allfreshuncompresseddefinedoperandvalues crossspecifiedowners,allcurrentINT32/FP32/clamp/BF16roundsretained. SameAIVrow-scaleretention/extraCubeepiloguemetadata notassumed.',
        refinement='v57externalinterfacearithmeticremainsasarelaxedall-onchipbudget;itsfullGMeliminationisnotestablishedforA3CVownership. RequiredinternalGMbridgesmustbeincludedbeforecallingSPLITUP/DownAddanactualGMtrafficreduction. RetaininginL2canavoidsomeHBMwithouteliminatingGM.',
        named_routes=[dict(id='F22-UP-CV2201',state='KEEP_CV_GM_BRIDGE',conclusion='G1int32valuescrossGMtoVectoractivation;rowwiseUBfusioncanremoveextramaterializationafterthebridgebutnotassumefreeCube→UB.'),
            dict(id='F22-SPLIT-CV2201',state='PRICE_INTERNAL_BRIDGE_AND_EXTERNAL_SPLIT',conclusion='AnexternalFP32SwiGLUsurfaceof8MIbytesdoesnotreplaceunpricedinternal16MIrawCube→Vectorbridge;sum24MIifbotharematerialized.'),
            dict(id='F23-DOWNADD-CV2201',state='NATIVE_EPILOGUE_DOMAIN_OPEN',conclusion='IfmergeisVector-owned,DownBF16needsGMbridge. AlternativeFixpipe/atomic/scalarepiloguesmayavoidanindependentDownsurfaceonlyaftertwo-round/alias/atomicsemanticsproof;notexcludedorcreditedhere.'),
            dict(id='VV-UB-FUSION',state='STILL_CONDITIONAL_LOCAL',conclusion='RMS→DQ,RoPE→quant,activation→rowquant,merge→HCcanreuseUBwithcorrectowner/fullrow/rounding/fanout/footprint. NoCVdirectwireclaimed.'),
            dict(id='CC-L1-FUSION',state='STILL_CONDITIONAL_LOCAL',conclusion='TwooutputprojectionscanhandoffroundedBF16viaAICL1 ifdtype/layout/fullrankpanels/order/ownershipfit;VectorcannotconsumeAICL1byborrowing3510paths.')],
        not_new_native_measurement=True,whole_candidate_closed=False,overall_complete=False))
    router_models=[]
    for m,e,role in [(3,384,'targetgraphlocaldeclaredno-extra-pad'),(2,128,'draftlocaldeclaredno-extra-pad'),(3,128,'draftlocaldeclaredno-extra-pad')]:
        router_models.append(dict(M=m,E=e,H=5120,domain=role,
            norm_FLOAT_Vector_to_Cube_value_write_read_bytes=8*m*5120,
            router_FLOAT_logits_Cube_to_Vector_value_write_read_bytes=8*m*e,
            same_width_two_GM_bridges_reference_bytes=8*m*(5120+e),
            actual_same_call_tensor_pointer_equivalence_attested=False))
    write(O/'router-CV-interface-domains-v80.json',dict(models=router_models,
        domain='Only when normremainsVector-ownedandrouterremainsCube-ownedconsumingFLOATandproducingFLOATforVectorTopK. Noassumption everylogicalFLOATAPlMatmullowerslikethis.',
        revised_question='RemovingapublishedFLOATnorm/logitarray isnotproofitsprivateGMbridge disappears. A BF16-input widening,ownership/nativeepilogue changeorallVectoralgorithmneedsitsownpath/precision/representationcapacityproof.',
        live_BF16_expert_shared_plane_preserved=True,FP32_HF32_compute_numeric_capacity_not_inferred=True,
        positive_padding_scope='Do notpairN800norm/routerwithP816TopKbyshape/count;v72/v77explicitpadding/fulloutputcostsremainseparate.',
        actual_GM_or_HBM_savings_or_speedup=None,overall_complete=False))
    # Specificcapacitychecks;notminimumstorageofeverypossiblelegaltiling.
    qb=3*32768*2;cs=2*3*64*4
    assert qb==192*1024 and qb+cs>192*1024
    write(O/'A3-local-capacity-and-ceiling-scope-v80.json',dict(
        documented_physical_bytes=dict(UB_per_AIV=192*1024,L1_per_AIC=512*1024,L0A_per_AIC=64*1024,L0B_per_AIC=64*1024,L0C_per_AIC=128*1024),
        specific_plan_check=dict(plan='SingleAIVsimultaneouslyretainsentireM3QB BF16queryandFLOATC/Srows beforepartialRoPE.',QB_result_bytes=qb,C_and_S_bytes=cs,total=qb+cs,physical_UB=192*1024,fits=False,
            consequence='Thiswhole-tensorplanfailscapacity;head/rowtiling,streamedC/S,differentownershipandGMworkspace remaineligible. Nofusionimpossibilityclaim.'),
        resource_bound_rule='T>=max_r(Qnecessary_r/Cupper_r)forone declaredlegalgraph. L1-onlycycleboundsneedthatpathanduncompressedoperanddomain;GMbridgessupplylogicalinterfacework,notHBMvolume. NewgraphcanchangeQandmustberepriced.',
        actual_Vector_FP32_HF32_Sort_nonlinear_integer_capacity_upper=None,actual_peak_frequency_upper=None,actual_HBM_L2_GM_bandwidth_upper=None,
        whole_model_lowest_MFU_module_established=False,
        module_ranking_gate='97BF16/INT8matrixregimesareaconditionaltypedpathsubset,notcompletehierarchicalmodulecost. PMUbusyfractions,cycles,peakreferencefractionsandnewingressfloorsare notinterchangeableMFU. Zero-primarymodulesnotrankedzero.',
        no_clock_scaling_of_published_BF16_or_INT8_peak=True,all_actual_physical_numeric_time_ceilings_complete=False,overall_complete=False))
    print(dict(candidate_frontier_index=35,shared_models=5,M3_previous_interface_budget=216624,M3_conditional_GM_bridges=216576,
        M800_previous_budget=57766400,M800_GM_bridges=57753600,whole_M3_QB_plus_CS_fits_single_UB=False,actual_MFU_rank=None))


if __name__=='__main__':main()
