"""Persist typedcyclebounds and architecture-specific fusion-frontier corrections."""
from copy import deepcopy
from pathlib import Path
from finalize_origin_coverage_v9 import N,O,read,write
from persist_profile_refinement_v5 import persist_refinement


def main():
    assert not(O/'REVIEW-CHECKPOINT-v80.json').exists()
    names=['A3-datapath-primary-evidence-v80.json','A3-typed-ingress-regime-models-v80.json','Markov-warm-ingress-frontier-v80.json',
        'A3-joint-ingress-cut-proof-v80.json','A3-fusion-interface-frontiers-v80.json','shared-CV-interface-revaluation-v80.json',
        'A3-local-capacity-and-ceiling-scope-v80.json','router-CV-interface-domains-v80.json',
        'candidate-relation-clauses-v80.json','candidate-relation-coverage-v80.json','operator-analysis-coverage-ledger-v80.json',
        'A3-datapath-branch-inventory-v80.json','next-operator-proof-questions-v80.json']
    for name in names[:8]:assert(O/name).exists()
    dense=read(O/names[1]);assert(dense['regime_count'],dense['native_executions'])==(97,30030)
    rel=deepcopy(read(O/'candidate-relation-clauses-v79.json'));assert len(rel['clauses'])==46 and len(rel['higher_order_groups'])==10
    members=['F04','F05','F07','F09','F10','F13','F16','F20','F21','F22','F23','F30','F35']
    rel['higher_order_groups'].append(dict(id='HG80-A3-LOCAL-PATH',members=members,
        contract='Chooseonelegalalgorithm/precision/operandownershipanddatapathdomainbeforepricinginterfaces. A3CVusesGMbridge;CCmayuseAICL1,VVmayreuseAIVUB. ForL1-onlydenseweights,accumulatenecessarysharedingressworkoncewithboundedinitialL0residency;maxwithotherresources. NoGM=HBM,no capturedclock=max,andnoall-onchipCube/Vectorassumption.',
        evidence=names[:8],scope_state='CONDITIONAL_ARCHITECTURE_AND_TYPED_RESOURCE_CUT_REVIEWED;ALL_TRANSFORMED_GRAPHS_OPEN',all_subsets_analyzed=False,legal_joint_fusion_proved=False))
    write(O/names[8],rel)
    cg=deepcopy(read(O/'candidate-relation-coverage-v79.json'));cg.update(parent_coverage='candidate-relation-coverage-v79.json',parent_relation_index='candidate-relation-clauses-v80.json',
        higher_order_groups=11,A3_local_datapath_refinement_v80='HG80-A3-LOCAL-PATH',overall_complete=False);write(O/names[9],cg)
    cov=deepcopy(read(O/'operator-analysis-coverage-ledger-v79.json'));cov.update(parent_coverage_ledger='operator-analysis-coverage-ledger-v79.json',
        new_pending_regimes_reconciled=0,new_pending_executions_reconciled=0,
        A3_typed_cycle_review_v80=dict(regimes=97,native_executions=30030,native_PMU_records=30030,new_regime_promotions=0,
            scope='ConditionalL1-onlyuncompressedweightcyclebounds/GMfrontierdomains;notallactualphysicaltimeceilings orMFU.'),overall_complete=False);write(O/names[10],cov)
    old={m['id']:m for m in read(O/'module-phase-branch-matrix-v79.json')['modules']}
    mids=['M07','M09','M10','M11','M12','M13','M16','M20','M22'];branches={mid:deepcopy(old[mid]['branches'])for mid in mids}
    details={
        'M07-OP-02':'RMS→DQ isVV andcanconditionalUBfuse;INT8inputVector→CubeQB onarch2201stillneedsGMbridge ifownershipkept. DQcommonproducercanremoveoneprivateplane withoutclaimingallCubeinputloadsdisappear. 97typedmatrixsubsethasconditionalL1ingresscyclemodels,notcurrenthardwareMFU.',
        'M07-OP-03':'FrozenV41QB BF16→partial64RoPE isCV. A3doesnothave3510L0C→UBdirectpath;fusedAPI/onekernel doesnotzeroGM. WholeM3QB+FLOATC/SsingleUBplanexceedsphysical192KiB;head/rowtilingorotherownership remainlegal. NolegacyQBheadRMSreintroduced.',
        'M09-OP-04':'IndexQprojection→RoPE/quant isCVthenVV:retainfirstGMbridge in2201Cube+Vectorvariant;Rope→quantUBreuse andnativeFP16scale/RINT/finitegates remain. Matrixpayloadcyclebounddoesnotbindactualquantizedkernelinstructionpath.',
        'M09-OP-05':'WeightprojectionCube→VectorFP32/FP16scaling requires2201GMhandoffunderthoseowners. NoF32VectorpeakborrowedfromBF16Cube orlogic-dtypehardwareMACassumption.',
        'M10-OP-06':'AlreadyfusedSMLA QKCube→VectoronlineSoftmax→PVCube hasGMscore/probabilitystages inthesourcecontract. Architecture2201CVVCbridgesmustbepricedinanyclaimedGM-eliminatingvariant;streamingGMtilereusemayavoidHBM/fullworkspacewithoutzeroGM.',
        'M11-OP-04':'WA→BF16rank1024→WB isCube→Cube,soA3L0C→L1APIcanconditionalkeepintermediateonAICwithdtype/layout/owner/fullrankpanel/roundingordercontracts. Do notapplyCVGMrequirementtoeveryCCfusionorassumeallL1loadtrafficvanishes.',
        'M11-OP-06':'WOB→RS→HC keepsroundedperrankpartial/reducedBF16 andcomm/readiness contracts. Jointingressboundsharenecessarytypedweightservice once;notaddperkernel/TASKfloorsasrequestlatency.',
        'M12-OP-02':'LogicalFLOATroutermatmuldoesnotattest itsphysicalinputformat/engine/nativeFP32/HF32capacity. IfCube-ownedFLOATproducerfeedsVectorTopK,retainGMscorebridge;allVector/otherproducerownership/newprecision-safeAPIrequiresseparateworkandcapacitymodel.',
        'M12-OP-05':'Cube-ownedrouter→VectorTopK cannotborrow3510directL0C→UB orSSBufferforA3. RowTopK/nonlinear/sort work/weights/tiesremain;onekerneldoesnotprove8MEGMtrafficelimination.',
        'M12-OP-06':'RemovingpublishedFLOATnormplane isnotproofprivateFLOATVector→Cubebridge disappears. BF16-inputwidening/insideGEMM requiresrealformat/conversion/precisionpath;currentBF16expert/sharedplane remains. DeclaredM3targettwoFLOATbridges132096Breference,notHBM oractualsavings.',
        'M12-OP-13':'MC2alreadyfusedCube/Vector/communication ownershipmustcountinternalGMinterfaces;samekernelsymboldoesnotmeanallonchip. Receiverrows/nativepadding/sharedquant andnetworkbytecontracts retained,actualGM/HBMupperratesunknown.',
        'M13-OP-06':'VectorDQ→CubeGateUp retainsINT8GMinputhandoffunderA3CVowners. RowScaleVVretentionmaybeconditionalbutdoesnotestablishfreeINT8input. Quantbeforegather andshared/routedcommonABIstillhaveoriginalprecision/row/generationconditions.',
        'M13-OP-07':'v57F22UP/SPLIT externalarithmeticretainedasrelaxedbudget. Under2201CV,rawG1INT32internalbridge16MI remains;externalFP32activation8MI produces24MIcombinedifbothmaterialized,notawholeGMhalving. Fullrowamax/UBcapacity/nativeDSQsemanticsretained.',
        'M13-OP-08':'CubeDownBF16→Vectormerge retains4MHGMstore/readbridge inthatownershipdomain. AlternativeFixpipe/atomic/scalarepilogues needcast-beforeadd/twoBF16rounds/alias/atomicorderingproof;notexcludedorgivengainhere. TPrawINT32RSdomainfromv78remainsspecificnewABI.',
        'M16-OP-01':'97BF16/INT8matrixregimesincludeheadweightpayloadbutL1-onlypath/orientationandfmaxnotattested. ConditionalcyclesnotactualnumericMFU;oldcoldHBMreferencecannotbeforcedonto cache-resident/currentnativecalls.',
        'M16-OP-02':'520W2MarkovGEMMs/3regimes havewarmL1-ingresscounterfrontier. Conventional5fullrecompute/Bweight/L1domainrequires323092480Bloads afterperstepL0capacitygrant,105173.333elapsedcorecycles;at1800MHz58.4296usonlyifadditionalfmaxassumption. Memo/prune/precompute/differentpathsoutsidebound.',
        'M20-OP-05':'FiveserialglobalIDdependencies plusfullW2recomputecanretainL2warmweights yetstillpayL1→L0service. No5xscanfreefromfusedlooporweightcachealone;sharding/memo/certifiedprunechangesQ/path andneedsnewbound. SavedMTE1budgetcompatibilitydoesnotproveactualpath orsaturation.',
        'M22-S14':'Documented2201localcapacities/rates added;97typedregimes30030PMUrecordsconditionalloadcyclemodels allbudgetcompatible. 1800MHzcapture!=fmax,MTE1busy!=bytes/MFU,GM!=HBM. ActualVector/FP32/HF32/Sort/nonlinear/memory/networkcapacities andwholemodulelowestMFUremainunknown.'}
    for bs in branches.values():
        for b in bs:
            if b['id']in details:b['evidence']+=names[:10];b['A3_datapath_review_v80']=details[b['id']]
    write(O/names[11],dict(modules=branches,extended_existing_families=sorted(details),unchanged_family_IDs_and_counts=True,protected_other_modules=17,
        global_records=342,semantic_modules=24,whole_modules_closed=0,overall_complete=False))
    write(O/names[12],dict(completed_bounded_questions=[dict(id='A3-LOCAL-DATAPATH-CAPACITY-AND-CV-INTERFACES',state='CONDITIONAL_OPERATOR_AND_CYCLE_BOUND_ANALYSIS_REVIEWED',evidence=names[:8])],
        questions=[dict(id='F23-FIXPIPE-ATOMIC-BF16-ORDER',state='NEXT_ANALYSIS_NOT_COMPLETED',question='ForA3Fixpipe/nativeatomicoutput,cananepilogueperformDownBF16round BEFORE BF16addand finalroundwhilepreservingroutedinputaliases/onewriter? UseprimaryAPI/SDKclauses first;do notcreditpotentialzeroDownsurfacefromfunctionnamealone.'),
            dict(id='BATCH-EXISTING-FUSION-GRAPH-COMPOSITION',state='OPEN',question='Reconcileoriginalfusion-scan/sourcegraphandcandidate-footprints/olderclausesinbatches,nowincludingtypedsharedresourcecuts/initialresidency andCVGMfrontiers.549centralpairslotsareunreconciledhere,notproofunexamined.'),
            dict(id='DECISIVE-TYPED-CAPACITY',state='OPEN',question='VectorFP32/Exp/Ln/Sqrt/Sort/INT64,AICPUandactualGM/L2/HBM/network/fmaxcapacityremainunbound. Targetprimaryevidenceonlyifitchangesnamedoperatorchoice;do not repeatbroadHF32peak/SoCguesses orwrongSKUtables.')],
        whole_model_lowest_MFU_module_established=False,unchanged_pending='9regimes1352exec/849missingPMU16phaseunbound3136ambiguous',overall_complete=False))
    write(O/'PROFILE-CODE-QUESTION-GATE-v80.json',dict(status='A3_ARCHITECTURE_LOCAL_CAPACITY_CYCLES_AND_FUSION_INTERFACE_DOMAINS',
        previous_turn_classification='PROGRESS:v79statephysicalstorageandlifetimeproof',
        questions=['WhichCV/CC/VVpathsactuallybelongtoA3ratherthan3510?','CanapiUBoutput/onefusedkernelproveGMinterfaceelimination?','WhatcyclefloorfollowsfromL1-onlyuncompressedoperandserviceandfiniteinitialL0capacity?','Do30030savedPMUbudgetscontradictthestatedmodels,anddoescompatibilityproveMFU?'],
        allowed_source='Sixpublicprimaryarchitecture/APIpages;existingtyped97regimes/30030TASK+PMUrecords,520Markovrecords andpinnedoperatorcontracts. No newtargetcode/server/model/Torch/NPU/capture/service/process work.',
        new_remote_source_read=False,new_target_source_files_obtained=0,new_public_document_snapshots=6,overall_complete=False))
    gap='Newlocalpathcycleboundsareconditionalon2201documentedrateasupper/uncompressednecessaryweights/L1-onlyingress/orientationandboundedinitialresidency. 2201GM→L0alternativeandotheralgorithmsrequireotherbounds. Captured1800MHznotfmax,PMUcyclebudgetcompatibilitynotbandwidth/MFU. A3CVfusionmustpriceprivateGMbridges;CC/VV/epiloguealternativesnotexcluded. Allactualtypedtimeceilings/moduleminimum/fullmodelbranchcomposition and9/1352operandsremainopen.'
    updates={cid:dict(review_v80=dict(primary_architecture=str(O/names[0]),typed_cycle_models=str(O/names[1]),Markov_frontier=str(O/names[2]),joint_resource_cut=str(O/names[3]),
        candidate_frontiers=str(O/names[4]),shared_interfaces=str(O/names[5]),capacity_ranking_scope=str(O/names[6]),router_interfaces=str(O/names[7]),
        conditional_operator_analysis_deliverable=True,implementation=False,native_test=False,new_performance_result=False))for cid in members}
    for cid in ['F04','F05','F07','F09','F10','F13','F16','F20','F21','F22','F23','F30','F35']:updates[cid]['conclusion_changing_gap']=gap
    for cid in ['F04','F20','F22']:updates[cid]['profile_review_state']='A3_CV_GM_INTERFACE_OWNERSHIP_AND_TYPED_RESOURCE_DOMAIN_RECONCILED_CONDITIONAL'
    updates['F22']['next_profile_or_contract_action']='Retainv57relaxedinterfacebudgetsbutrepriceA3Cube/VectorinternalGMbridges;F22SPLITexternal8MIplusinternal16MInotwholeGMhalving. Keepv78TPshardamax/rawINT32RSdomains,actualcapacity/workspace/precisiongates;noimplementationneededforconditionalanalysis.'
    updates['F23']['next_profile_or_contract_action']='NextverifyA3Fixpipe/atomicBF16cast-addorderandallaliases/readersbeforeclaimingindependentDownBF16surfacedeletion. CurrentVector-ownedmergeusesGMbridge;VVmergeHCfusionsremainconditional.'
    updates['F30']['next_profile_or_contract_action']='Usefive-stepdenseL1-onlywarmcycleboundasconditionalcapacitycut,notactual58uslowerboundfromcapturedclock. Memo/prune/precomputedtransition/shardedrecurrencechangesdomain;preserveglobalID/RNG/fullscorecontracts.'
    updates['F35']['next_profile_or_contract_action']='PureID/readers/rowselectionroutesstillconditional;Cube→VectorargmaxcanuseGMtileswithoutfullpublishedscorebankbutdoesnotproveallGMscorevolumezero. Alternativeproducer/epiloguealgorithmneedsnewprecision/capacityproof.'
    mods={mid:dict(finding='A3arch2201CVusesGM,CCmayuseL1,VVmayuseUBunderstatedownership.6primarydocs+97typedregimes30030PMUrecords giveconditionalL1cycleconstraints,notactualfmax/MFU. RelaxedGMinterfacebudgetsrepricedforrouter/sharedMLP;Markovwarmcycle/initialresidencyresourcecutadded.',
        evidence=names,candidates=[cid for cid in members if cid in old[mid]['candidate_links']],gap=gap,
        profile_scope='Onlycomputationalownership,localmemory/dataflowcapacity,typedmatrixweightingress andspecificfusion/cyclebounddomains.')for mid in mids}
    remaining=list(read(O/'module-coverage-gaps-v79.json')['remaining'])
    remaining += ['v80primaryarch2201A3/A2CVsplitandlocalbuffersdocumented:UB192KiB,L1 512KiB,L0A/B64KiB,L0C128KiB;L1→L0A256B/cycle/L0B128B/cycle. Sameguide2201GM→L0directpathsmeanL1-onlyisnotuniversal. Captured1800MHznotfmax;Vector/FP32/HF32/Sort/nonlinear/integer/actualHBM-L2-GM-networkcapacitiesremainunbound;wholelowestmoduleMFUnotestablished.',
        'v80all35candidateCV/CC/VVfrontiersindexednotallvariantsproved;sharedM3old216624Brelaxedinterfaces includes216576BconditionalCVGMbridges,currentownership. F22SPLIT8MIexternal+16MIinternal=24MIifbothmaterialized,nothalfwholeGM;DownAddFixpipe/atomicroundorderstillopen. RouterM3FLOATsamewidthtwoCVbridges132096Bconditional,notactualHBMsaving. CC-L1andVV-UB remaineligible withcontracts;noallfusionimpossibility.',
        'v8097BF16/INT8matrixregimes30030native+PMUrecords30030,noPMUmissinginthesubset;nophysicalassignmentchanges.520Markov/3regimescompatiblewithL1ingresscounterbudget,butcompatibilitynotpath/saturationproof. FivefullW2recomputeL1/Bdomain105173.333cycles,58.4296usonlyextra1800MHzupperassumption. DistinctQB/sharedG1/G2jointweightcut77332480BgrantsinitialL0capacityonce,not3times.46pairclauses11groups/549centralpending;342records24semanticmodulesandglobal9/1352+849/16/3136/0wholeclosurespreserved.']
    action=('继续原Goal,v80 PROGRESS(前v79有效)。只围绕算子融合/理论上限，profile优先，代码定点补语义。先读A3-datapath-primary-evidence/typed-ingress-regime-models-v80、Markov-warm-ingress-frontier/A3-joint-ingress-cut-proof/A3-fusion-interface-frontiers/shared-CV-interface-revaluation/A3-local-capacity-and-ceiling-scope/router-CV-interface-domains-v80和next-operator-proof-questions-v80。'
        '6publicprimary页快照SHA，previewmaster不是installedCANN9.2证明。Arch2201 A3/A2 separateAIC/AIV1:2，CV/VC经GM；3510新增UB→L1/L0C→UB/SSBuffer不能给A3。GetTensorCUB逻辑返回不证明直连，异步SetWorkspace明确GM缓存。CC可L0C→L1保持BF16，VV可UB，不能一概禁fusion。'
        '物理buffers UB192KiB,AICL1 512KiB,L0A/B64KiB,L0C128KiB；L1→L0A256B/cycle、B128B/cycle是声明架构容量模型。2201另有GM→L0直达，所以L1-only lowerbound不是所有实现。1800MHz只是捕获频率不是fmax，别clockscale376/752publishedpeaks。'
        '97BF16/INT8matrixregimes30030exec，30030uniquePMU无missing，simple2DdenseweightpayloadK*N verified。固定B通路warmR24*64KiB/cap24*128，允许A/B通路warmR双倍/cap24*(256+128)弱界，指令路径/压缩/算法/原始数据必要性条件显式。原nativeMTE1counterbudget全兼容(0exceed)，不证明实际路径/饱和率/MFU。'
        'Markov520/3regimes，fullW2=66191360B；5serialIDdependentdensefullrecompute且L1/B域eachstepatmost1.5MiBL0Bwarm，necessaryload323092480B、elapsedcycle105173.333。58.4296us仅额外fmax<=1.8GHz条件，当前实际时间下界未定。Memo/prune/precompute/directGM/otherorientation另域。QB41.943MB+sharedG1/Down35.389MBdistinctplanes联合cut77.33248MB只赠初始L0warm一次，24661.333cycles；资源work先积累、不同资源取max，别加每kernel下界当request。'
        '关键scope修正：v57idealallinterface字节算术保留为relaxed域；保留A3CV分工时sharedM3最少4GMbridges216576B接近旧216624B，M80057753600B对旧57766400B，GM不等于HBM(可能L2)不可除1.6TB/s造actualtime。F22SPLIT外部8MI FP32activation面之外仍有内部16MIINT32CVbridge，总24MI若均写读，不能说整体GM减半。F23DownAdd若mergeVector就保4MHGMbridge，Fixpipe/atomic/scalar二舍入/alias独立新域下一问。'
        'RMS/DQ VV、RoPE/quant VV、CC两级投影L1融合依旧条件成立；M3QB全矩阵196608B再放1536B C/S不能单UB全驻留，但按head/rowtile/流水等域不排除。F04/F20新routerBF16-inputwiden或producerownership必须证真实cast路径；删publishedFLOAT数组不保证私有GM桥消失，FLOATrouter物理指令/FP32peak仍未知。M3same-rowtarget双FLOAT桥132096B；N800/P816padding不得混配。'
        '全35frontier导航而非全部新语义证明。13candidate主记录富化，9modules保原ID/342records24semantic，17其他modulematrix完全不变；46pairs不增边，新HG80A3LOCALPATH使11groups/549中央未汇入。优先下一问F23 A3Fixpipe/nativeatomic BF16roundbeforeadd是否能保二舍入及routedinputalias/onewriter；只读primaryAPI/SDK定点条件不实施，不因函数名credit潜在GM省。随后批量originalfusion-scan/graph/candidatefootprints关系汇入及namedtyped容量，不重搜泛HF32peak/猜SoC。'
        'actual全模块最低MFU未成立，缺Vector/FP32/HF32/Sort/ExpLnSqrt/INT64/AICPU/actualGM-L2-HBM/network/fmax上界必须保留；条件分析本身不需实现。2073/285598review9/1352pending、849PMU16phase3136ambiguity35model7legacyframework0wholeclosure保护。3PASS加1resolvedprewriteERROR：verifier条件表达式else2漏空格，py_compile在checkpoint前拒绝，旧脚本hash保留后已修正。所有localexec已完成，无server/model/Torch/NPU/newcapture/服务/进程/实现。verify_A3_datapath_ceiling_v80应PASS。Goalactive，未全覆盖不得因报告成稿complete。')
    report='''本轮补上A3局部数据通路的容量模型，并据此修正融合接口预算的适用域。未执行模型、设备测试或优化实现。

当前官方[2201架构资料](https://asc.gitcode.com/guide/programming_guide/advanced_programming/hardware_implementation/architecture_spec/npu_arch_2201.html)描述了分离的Cube/Vector核、192KiB UB、512KiB L1，以及L1到L0A/L0B的256/128字节每周期通路。[3510迁移资料](https://asc.gitcode.com/guide/cross_gen_migration_guide/3510_arch_migration/2201_to_3510_arch_changes.html)明确把UB→L1、L0C→UB直达列为新通路，因此不能将这些能力套用到A3。资料为公开master预览，未绑定本机CANN9.2二进制。

这会改变“融合后中间结果全留片上”的预算。在保留Cube矩阵计算、Vector量化/激活/合并的A3实现域内，内部GM中转仍需计价。共享专家M=3的四处数据桥合计216,576B，接近此前放松模型的216,624B全部接口预算。GM访问可以命中L2，不能直接换算为HBM流量。

F22-SPLIT外部FP32激活面8MI字节，也不能直接替代内部INT32 Cube→Vector桥的16MI；若两者都物化，应计24MI。融合仍可减少启动、缩小workspace、形成分块流水，VV可在同一UB复用，CC在符合条件时可通过L1传递。候选需要按具体计算单元和数据通路重新计价。

[GetTensorC](https://asc.gitcode.com/api/SIMD-API/adv_api/cube_compute/Matmul_Kernel/GetTensorC.html)可返回UB结果，但异步模式会使用GM workspace；API名称或返回类型不足以证明硬件直连。类似地，F23若仍由Vector做共享/路由合并，就需保留Down到合并的GM桥。能否借Fixpipe/原子输出保留“Down先BF16舍入，再相加并舍入”的合同，已列为下一项具体问题。

本轮对97类、30,030次BF16/INT8矩阵执行建立了条件周期模型，并关联全部30,030份保存的PMU记录。没有发现模型超过已记录MTE1周期预算，但这不证明实际采用该通路，也不证明带宽饱和或MFU。模型明确限定为L1进入L0、必须读取未压缩权重的稠密算法；2201还存在GM直达L0通路，其他实现要重建下界。

| 条件模型示例 | 周期下界 |
|---|---:|
| Q_B权重经L1→L0B，允许初始L0B全暖 | 13,141.33 |
| 五步Markov完整W2重算，同一通路 | 105,173.33 |
| Q_B与共享Gate-Up/Down的三份独立权重联合通路 | 24,661.33 |

联合模型只允许一次初始L0容量赠予，并先累加同一资源上的必要工作，再与其他资源下界取最大值。捕获频率1.8GHz不是硬件最高频率证明，因此对应微秒值仍是带额外频率前提的参考，尚不能标为本机实际时间下界。五步Markov即使权重在L2中保持温热，在该L1重算域内也仍有局部搬运工作；memo、预计算和剪枝属于其他算法域。

已给35项模型候选建立CV/CC/VV接口索引，更新13项候选和9个模块的相关条件。现有342条分支记录、46条候选对关系保持，新增一组共享局部资源约束后共11组。实际Vector、FP32/HF32、排序、非线性、整数、内存/网络及频率上界仍有缺口；全模块最低MFU尚未定案。2,073类、285,598次条件分析与9类、1,352次关键操作数缺口保持，Goal继续进行中。
'''
    docs=read(O/names[0])['primary_sources']
    jobs=[dict(kind='review',status='error',description='Resolvedprewriteverifierpy_compilesyntaxfailure:else2missingwhitespace. Oldverifiersnapshotandhashretained;correctedbeforecheckpoint/sharedstatewrites;analysisdataunchanged.',artifacts=[O/'A3-datapath-validation-history-v80.json',N/'verify_A3_datapath_ceiling_v80.before-else-space-fix.txt']),
        dict(kind='profile',status='pass',description='97typedmatrixregimes30030nativeTASKs/30030exactPMU,no missinginthesubset.520Markov/3regimes;B/eitherL1counterbudget0exceedances. Noactualpath/bytes/saturation/MFUorassignmentpromotionclaimed.',artifacts=[O/names[i]for i in[1,2,3]]+[N/'analyze_A3_ingress_cycles_v80.py']),
        dict(kind='design-check',status='pass',description='6primaryarchitecture/APIHTMLsnapshots;2201CVGM/CCL1/VVUBdistinction,finitebuffers/256-128Bcyclemodels,explicitGMdirectalternative/fmaxunknown. Shared/routerCVbudgetrevaluation andwholeM3UBcounterexample;35candidatefrontiersindexnotallvariantproof.',artifacts=[O/names[i]for i in[0,4,5,6,7]]+[Path(r['snapshot_path'])for r in docs if r['status']=='READ']),
        dict(kind='review',status='pass',description='13candidatemasters9modulefamiliesrefinedwithoutID/countdrops;17othermodulesidentical.46clauses11groups/549centralpending;2073/285598review9/1352pending/global84916/3136/0closuresprotected.',artifacts=[O/names[i]for i in[8,9,10,11,12]])]
    persist_refinement(version=80,previous=79,title='A3 local datapath cycle bounds and architecture-specific fusion interface domains',
        summary=dict(filename='A3-datapath-ceiling-summary-v80.json',previous_turn_classification='PROGRESS:v79statephysicalstorageandlifetimeproof',
            new_public_document_snapshots=6,new_target_source_files_obtained=0,new_remote_source_read=False,typed_matrix_regimes=97,typed_matrix_executions=30030,typed_matrix_PMU_records=30030,
            conditional_B_or_either_ingress_budget_exceedances=0,Markov_executions=520,Markov_regimes=3,Markov_five_step_conditional_cycle_floor=105173.33333333333,
            candidate_frontier_index=35,candidate_masters_refined=13,module_surfaces_refined=9,protected_other_modules=17,retained_global_records=342,retained_semantic_modules=24,
            relation_clauses=46,higher_order_groups=11,pending_central_pair_reconciliation=549,scoped_reviewed_regimes=2073,scoped_reviewed_executions=285598,pending_regime_reconciliation=9,pending_execution_reconciliation=1352,
            captured_frequency_is_maximum=False,actual_lowest_MFU_module_established=False,all_actual_physical_numeric_time_ceilings_proved=False,
            implementation=False,native_test=False,new_performance_result=False,remote_job_status='No remote/server/model/Torch/NPU/capture/service/process work;alllocal savedprofile/counteranalysis complete;6publicHTTPprimarypages only.'),
        changes=['SixprimaryA3architecture/APIpages giveexplicitCVGM/CCL1/VVUBdomains andlocalcapacity/cyclerates;no3510directpathsor1800MHzmaxassumed.','97typedregimes30030exactTASK/PMU/520MarkovconditionalL1ingressandjointinitial-residencycutmodels;zeroexceedancesnotactualpath/MFUproof.','Repriceoldshared/routerGMinterfaceeliminationwithinA3CVdomain;35candidatefrontiers/13masters9modulesrefined,46pairs11groupsandallglobalopenconditionspreserved.'],
        candidate_updates=updates,framework_updates={},module_updates=mods,semantic_branches=branches,remaining=remaining,action=action,jobs=jobs,report=report,
        read_first=names,publication_rows={},gate_updates=dict(overall_complete=False),scripts=[N/'capture_A3_datapath_evidence_v80.py',N/'analyze_A3_ingress_cycles_v80.py',N/'model_A3_fusion_frontiers_v80.py',N/'verify_A3_datapath_ceiling_v80.py'],
        executor_prefix='A3-datapath-ceiling',main_script=Path(__file__),source_read=False)


if __name__=='__main__':main()
