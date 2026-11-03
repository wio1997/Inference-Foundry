"""Persist shared-FFN operator ceilings, precise fusion gates and singleton merge specialization."""
from copy import deepcopy
from pathlib import Path
from finalize_origin_coverage_v9 import N,O,read,write
from persist_profile_refinement_v5 import persist_refinement


def main():
    assert not(O/'REVIEW-CHECKPOINT-v57.json').exists()
    d=read(O/'shared-expert-shape-ceiling-v57.json');v=read(O/'shared-expert-fusion-variants-and-interactions-v57.json')
    c=read(O/'shared-operator-numerical-and-library-contract-v57.json');s=read(O/'F23-singleton-merge-specialization-v57.json');inputs=read(O/'shared-operator-inputs-v57.json')
    assert d['regime_count']==42 and all(x==4472 for x in d['captured_stage_executions'].values())
    assert len(v['variants'])==7 and len(s['exact_root_graph_exposure'])==64
    m3=next(x for x in d['logical_shape_models']if x['M']==3);m800=next(x for x in d['logical_shape_models']if x['M']==800)
    assert m3['total_removed_interface_bytes']==216624 and m800['total_removed_interface_bytes']==57766400
    assert m800['aggregate_materialized_reference_floor_us']==m800['aggregate_ideal_fused_reference_floor_us']
    assert not c['installed_fused_API']['drop_in_equivalence_proved'] and c['model_text_config']['swiglu_limit']==10
    e=['shared-expert-shape-ceiling-v57.json','shared-operator-numerical-and-library-contract-v57.json',
       'shared-expert-fusion-variants-and-interactions-v57.json','F23-singleton-merge-specialization-v57.json','shared-operator-inputs-v57.json']
    write(O/'PROFILE-CODE-QUESTION-GATE-v57.json',dict(status='SHARED_OPERATOR_FUSION_NUMERICS_ROW_QUANT_AND_INSTALLED_FUSED_API',
      previous_turn_classification='PROGRESS:v56MC2operator_receiver_and_communication_models',
      profile_evidence=['MC2-operator-profile-v56.json','operator-shape-ceiling-ledger-v55.json',e[0]],
      questions=['Whichinteger/float/quant/round barriers constrainsharedGateUp-activation-Downfusion?',
        'Does the installedfusedGMM/SwiGLU/Quantinterfaceexpresscurrentclamp10?',
        'Can singletonForeachAddList be specialized andhowdoesDown-mergefusioninteractwith routed/shared overlap?'],
      allowed_source='Exactsharedoperatorcaller(2selectedmethods),3installedoperatorSDKheaders,existingmodelconfig/caller andpriorv22mergecontract. Noframework/APIresearch.',
      opaque_profile_attr_positions_not_used_as_semantic_schema=True,
      no_model_Torch_NPU_newcapture_service_health_process_work=True,whole_module_closed=False,overall_complete=False))
    updates={}
    for cid in ['F22','F23']:
      updates[cid]=dict(review_v57=dict(shape_ceiling=str(O/e[0]),numerical_library_contract=str(O/e[1]),variants=str(O/e[2]),singleton_merge=str(O/e[3]),
        relation='All42capturedM13regimes/5x4472stages modeled. INT32GateUp differsfromMC2FP16;rowquant/clamp10 constrainfusion. InstalledGMM-SwiGLUQuantV2header isnotclamp-equivalence proof. SingletonForeach specialization andDownBF16mergefusion arealternatives;overlapreadiness andaggregatecompute ceilings preventaddingallbyte/time savings.',
        implementation=False,native_test=False,new_performance_result=False,conditional_operator_analysis_deliverable=True),
        profile_review_state='SHARED_OPERATOR_REGIMES_WORK_TRAFFIC_FUSION_AND_INTERACTIONS_ANALYZED',
        conclusion_changing_gap='Captured42regimes and7conditionalproposaldecisions reviewed;actualphysicalcache/Vector/INT8capacity/internalkerneltiling/allconsumer/nativeprecision fornewimplementation notattested. Do not repeat currentclamp,INT32boundary,libraryheaderavailability orsingletonalpha1 as unknown. Nativebenchmarkabsence alone isnot unanalysedconditionalproposal.',
        next_profile_or_contract_action='AdvanceHcPre/HcPost→norm/quant andattention/indexer operatorceilings. KeepF23-ONEADD aspriorityoperator specialization;do notproclaim17usgainfromcrossshapeAddreferences ordropclampusinggenericfusedAPI.')
    mods={'M13':dict(finding='All42capturedsharedregimes modeled(4472eachof5stages). M3stage medians3.82/25.681/4.66/21.52/20.561us. CurrentINT32→FP32DSQ/clamp10/rowquant differsMC2FP16. 7conditionaloperatorvariants andreadiness/rooflineinteractions reviewed;singletonforeachAddspecialization prioritized.',evidence=e,candidates=['F22','F23'],gap='FullphysicalVector/cache/capacity/otherunobservedmodeloperatorbranch andnewkernelprecision/ownership proof notclosed;observedregime analysis isnotwholemodule/fullscopecompletion.'),
      'M12':dict(finding='Commonquant producer acrossrouted/shared isconditionalonrow/mask/generation andquantizer equality;existingMC2BF16inputABIneedsnewprequantizedcontract. SameINT8dtype isnot proof;newvariant relatesF21/F22,notframeworkwork.',evidence=e[:3],candidates=['F21','F22'],gap='Actualquantizer equivalence,allproducers/consumers/newnativebufferABI andphysicalcost stillconditional.'),
      'M06':dict(finding='Down→mergefusion maycomposewithHcPost onlywhilepreservingDownBF16round andmergeBF16round/allresidualcoefficients. Existingv22HCdependencyconditions retained;not anewunroundedepilogue rewrite.',evidence=[e[1],e[2],'merge-rounding-and-dependency-review-v22.json'],candidates=['F23'],gap='HcPre/HcPostactualoperatorwork/IO/vectorcapacity/tiling andotherconsumer bounds remainpriority nextwork.')}
    remaining=deepcopy(read(O/'module-coverage-gaps-v56.json')['remaining'])
    remaining.append('M13v57capturedoperatorregimes reconciled:42regimes,4472eachDQ/G1/DSQ/G2/merge;H5120/I2304all5Mdomains. SharedG1INT32→FP32DSQ,one-rowquantScale/clamp10/mode1 known;sourcecurrentV2GMMfusedheaders lackclamp/modeargs,so genericstandardformula notcurrentdrop-in. 7conditionaloperatorproposals analyzed including singletonForeach→Add;no same-shapeAddbenchmark or17usgain claim. M3allinterfacebytes216624versus35.39MBweights;M800delta57.7664MBbutaggregateINT8roofremains75.2967usunderpublishedreference. Vector/cache/physicalmode/allnewconsumer/othermodeloperator gaps remain;do notcontinueframework/lifecycleorinstalledAPIcatalog audits.')
    action=(
      '继续原operator Goal。v57完成共享专家捕获域42regime/5x4472stages的计算与逻辑IO模型，7个条件融合/替换方案及相互约束；不是框架研究。先读shared-expert-shape-ceiling-v57、shared-operator-numerical-and-library-contract-v57、F23-singleton-merge-specialization-v57和shared-expert-fusion-variants-and-interactions-v57。'
      'typedgraph路径BF16→DQ(INT8+scale)→G1 INT32→DSQ(INT8+scale)→G2 BF16→BF16Add；勿套routedMC2内部FP16。G1INT8dotK5120最大abs部分和83886080，G2K2304为37748736，均fitsINT32；后续FP32反量化/scale/SwiGLU/clamp/RINT仍保序。'
      '原166sharedcaller历史471db hash匹配，选2算子方法；3SDKoperator headers在latest/include/aclnnop实际存在，opp备用路径不可用不等于API缺失。当前text_config.swiglu_limit10，model101f传入及sharedcallerclamp参数已知；不以opaqueattr_N按API参数顺序强行映射语义。V2GroupedMatmulSwigluQuant(NZ)有头文件但缺clampLimit/swigluMode/GLu调参，公开standardformula与clamp10反例不同；不能直接替换。不要因此无限扫API或框架，条件变体需要compatibleclampedkernel。'
      '全root2560图层5阶段medianus=3.82/25.681/4.66/21.52/20.561。F23-ONEADD值得优先：_foreach_add([shared],[routed])[0],alpha1,可条件特化普通Add；同capture更大BF16Add3.88..4.35us只是跨shape/path参考，非同shape17us收益。64exactrootgraphs merge时段曝光比例median1.7103%(不是严格endtoend收益上限/利用率)。'
      '共享链Qmaterialized=3HI+(8I+2H)+12MH+18MI+16M，ideal全接口消除Q=3HI+(8I+2H)+6MH；上投影BF16scale可能额外读取4Iextent未当必死。M3delta216624B，referencecoldIO22.32931→22.19392us；M800delta57766400B但aggregate原/新floor都75.29668us(算力项)，serialstage参考floor108.7222us不能用LB比当实测gain。参考752INT8TOPS/1.6TB只是假设hardware容量，实际cache/Vectorbound未闭。'
      '行量化必须全Iamax等归约后定scale，不能每Ntile独立scale。F22-SPLIT可条件先G1epilogue FP32SiLUtile+amax，再rowquant：INT32gateup16MI往返→FP32act8MI但还有barrier/存储。Down+Add融合必须保DownBF16round；全shared提前overlap会让Down结果早于routed，要buffer/保持occupancy/延迟Down。只重叠前三阶段则preDownspanmedian34.501us，不能把完整57.301usoverlapcap与无Down暂存当独立收益。'
      '下一步离开已答共享链局部问题，推进HcPre/HcPost→norm/quant/残差的operatorwork/traffic/Vectorupperbounds、再SparseFlashMla/Indexer及其它尚缺定量ceil的operator。沿83类型/2082regimes/35model候选覆盖跨算子层卡迭代；不再独立研究framework/API/请求生命周期。'
      '原scope真实未完成项/849PMU缺/16unbound/3136module歧义保留，M21/M22语义不改。条件分析无需先native实现benchmark；全scope不因报告/ledger成稿完成。无模型/Torch/NPU/newprofile/health/process/kill/实现，Goal active。必要原文件只读可用ssh166-v56-xinyufeng.sock(显式override旧transport)，优先本地资料。')
    report=(
      '本轮分析共享专家 DQ→GateUp→DequantSwiGLUQuant→Down→Merge，覆盖42个捕获shape/role/context组合，每阶段4472次执行。仅补读了2个算子调用方法及3份本机SDK头文件。\n\n'
      '| M=3图层阶段 | 中位设备耗时 |\n|---|---:|\n| DynamicQuant | 3.82μs |\n| INT8 GateUp→INT32 | 25.681μs |\n| DequantSwiGLUQuant | 4.66μs |\n| INT8 Down→BF16 | 21.52μs |\n| ForeachAddList | 20.561μs |\n\n'
      '共享上投影保留INT32累加，再进行FP32反量化、clamp/SwiGLU和逐token量化；它不同于routed MC2内部FP16路径。完整行的量化尺度约束跨N-tile流式执行。可研究G1 epilogue先输出FP32激活与局部max，再完成行量化，将该中间接口从16MI字节减到8MI；仍需保留全行归约、NaN/zero及舍入条件。\n\n'
      '本机存在GroupedMatmulSwigluQuant及NZ版V2头文件，公开算子定义也包含GEMM、反量化、SwiGLU和量化。但当前调用使用clamp=10及SwiGLU mode1，而这些参数未出现在该V2融合接口。普通SwiGLU与clipped版本有明确反例，不能把“接口存在”当成当前模型可直接替换。[原始API说明](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/aolapi/context/ops-transformer/aclnnGroupedMatmulSwigluQuantWeightNzV2.md)\n\n'
      '优先候选是将单元素foreach合并特化为普通Add，或把同一BF16 Add融入Down epilogue。前者保留alpha=1及相同Tensor语义；同profile中更大BF16普通Add为3.88–4.35μs，支持优先验证，但不是当前shape实测提速。64个root图中merge设备时段曝光份额中位1.7103%，也不是端到端可实现收益。后者必须先保留Down的BF16舍入，再add并舍入；不能直接把未舍入累加器加到routed结果。\n\n'
      '| 理想接口消除模型 | M=3 | M=800 |\n|---|---:|---:|\n| 可减少的逻辑中间读写 | 216,624B | 57,766,400B |\n| 原链逻辑字节量 | 35,726,896B | 117,760,512B |\n| 理想融合后逻辑字节量 | 35,510,272B | 59,994,112B |\n\n'
      '用此前注明的参考752INT8TOPS/1.6TB/s作条件Roofline，M=800的聚合下界融合前后均为75.2967μs：计算项已高于访存项，所以省57.77MB不能直接报省36.10μs。原串行阶段的参考下界108.7222μs与理想聚合下界也不能相除当真实加速。缓存、向量计算、量化归约及本机容量尚需进一步绑定。\n\n'
      'Down+Merge融合与routed/shared重叠还存在取舍：共享Down先完成时，routed结果可能未就绪，需要暂存、占住资源或延迟Down。若只重叠Down之前三阶段，观测span中位34.501μs；不能同时领取完整57.301μs重叠预算和“无Down暂存”的全部收益。\n\n'
      '共记录7个条件方案：输入量化＋GateUp、G1＋激活量化、FP32 tile＋行量化拆分、受clamp条件约束的库融合复用、Down＋Add、两支量化输入共用、singleton Add特化。下一步转向HcPre/HcPost、norm/quant及其它热点算子的定量上限。整体仍未完成，没有实现或新增设备测试。\n')
    src=[Path(r['local_path'])for r in inputs['files']if r['status']=='READ']
    cfg=Path(c['model_text_config']['path'])
    jobs=[
      dict(kind='review',status='pass',description='Exactoriginal166sharedcallerhistoricalhashmatches;3installedoperatorheadersread,3alternatepathsunavailable. No framework/SDKwideaudit. Clampcurrentcasefromsource/config,notopaqueattrordinal.',artifacts=[O/e[1],O/e[4],N/'shared-operator-source166-v57.json',N/'read_shared_operator_contract_v57.py',cfg]+src),
      dict(kind='profile',status='pass',description='42M13regimes conserve4472executionsineach5stages;2560rootboundgraphstages and64graphmergeexposures exact. No crossshapeAddspeedupclaim.',artifacts=[O/e[0],O/e[3],N/'analyze_shared_expert_ceiling_v57.py',N/'enrich_shared_ceiling_and_exposure_v57.py']),
      dict(kind='design-check',status='pass',description='FiveMdomainsdenseINT8/IOmodels,whole-rowquant/clampwitnesses,safeINT32dotbounds andnonadditivecompute-vs-IO/overlapconstraints. Lowerboundratiosnotbenchmarkgains.',artifacts=[O/e[0],O/e[1],O/e[2]]),
      dict(kind='review',status='pass',description='Sevenconditionaloperatorfusion/replacementproposals refineF22/F23;singletonAdd andcompatibleclamp support separated fromgenericfusedAPIavailability. Allkeyphysical/otheroperatorscopegaps retained.',artifacts=[O/e[2],O/e[3],N/'refine_singleton_merge_candidate_v57.py']),
    ]
    persist_refinement(version=57,previous=56,title='Shared expert operator ceilings, quantization fusion and singleton merge specialization',
      summary=dict(filename='shared-expert-ceiling-summary-v57.json',previous_turn_classification='PROGRESS:v56MC2operator_receiver_and_communication_models',
        captured_M13_regimes=42,executions_per_stage=4472,root_bound_graph_layers=2560,conditional_variants=7,
        installed_operator_headers=3,drop_in_GMM_SwiGLU_clamp_equivalence=False,
        M3_removable_logical_interface_bytes=216624,M800_removable_logical_interface_bytes=57766400,
        M3_singleton_merge_median_ns=20561,M800_aggregate_roof_unchanged=True,
        target_implementation_changed=False,native_test=False,new_performance_result=False,
        remote_job_status='Onlyexisting166stage caller/headerdockercp readcompleted. Allprofile/modelanalysislocal;no model/Torch/NPU/newcapture/health/process/kill/implementation/backgroundanalysis.'),
      changes=['42sharedoperatorregimes modeled andINT32/FP32clampedquant path distinguishedfromMC2FP16.',
        'InstalledGMM-SwiGLUQuantV2 capabilityfoundbutclamp10drop-in notestablished;rowquantbarrier andDown-BF16merge readiness bound.',
        'SingletonForeachAddspecialization prioritized;7conditionalvariants/IOroofline/overlapinteractions recorded withoutgainclaims.'],
      candidate_updates=updates,framework_updates={},module_updates=mods,remaining=remaining,action=action,jobs=jobs,report=report,
      read_first=e,publication_rows={},gate_updates=dict(overall_complete=False),
      scripts=[N/x for x in ['read_shared_operator_contract_v57.py','analyze_shared_expert_ceiling_v57.py','refine_singleton_merge_candidate_v57.py','enrich_shared_ceiling_and_exposure_v57.py']],
      executor_prefix='zcode-shared-ceiling',main_script=Path(__file__),source_read=True)


if __name__=='__main__':main()
