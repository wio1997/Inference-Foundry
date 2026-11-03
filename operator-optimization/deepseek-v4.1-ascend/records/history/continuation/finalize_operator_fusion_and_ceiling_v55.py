"""Checkpoint the user's operator-only focus with real fusion and roofline evidence."""
from copy import deepcopy
from pathlib import Path
import json,hashlib
from finalize_origin_coverage_v9 import N,O,read,write
from persist_profile_refinement_v5 import persist_refinement


def main():
    assert not(O/'REVIEW-CHECKPOINT-v55.json').exists()
    prof=read(O/'operator-fusion-profile-v55.json');roof=read(O/'F04-router-fusion-roofline-v55.json')
    ledger=read(O/'operator-shape-ceiling-ledger-v55.json');scope=read(O/'OPERATOR-FOCUS-REQUIREMENTS-v55.json')
    assert prof['operator_types']==83 and prof['total_ComputeInfo_executions']==286950
    assert len(prof['graph_chains'])==3200 and roof['root_bound_chains']==2560 and roof['phase_unbound_chains']==640
    assert ledger['regime_count']==2082 and ledger['dense_model_regimes']==87
    assert len({r['regime_id']for r in ledger['observed_regimes']})==2082
    assert sum(r['executions']for r in ledger['observed_regimes'])==286950
    assert not ledger['all_operator_fusion_decisions_complete']and not ledger['all_theoretical_bounds_complete']
    e=['OPERATOR-FOCUS-REQUIREMENTS-v55.json','F04-router-fusion-roofline-v55.json','operator-shape-ceiling-ledger-v55.json',
       'operator-dense-reference-roofline-v55.csv','operator-fusion-profile-v55.json','hardware-reference-scope-v55.json']
    write(O/'PROFILE-CODE-QUESTION-GATE-v55.json',dict(status='OPERATOR_FUSION_AND_CEILING_ONLY_AFTER_EXPLICIT_USER_STEERING',
        previous_turn_classification='PROGRESS:v54request_provenance_and_output_consumers',
        profile_evidence=e[1:5],question='For each operatorchain,which materialized interfaces can disappear whilepreserving actualdataflow/numerics,and what compute/memory/communication floor applies to its exactshape?',
        reused_operator_source=['F04-precision-routing-and-row-contract-v29.json','F04-existing-fusion-and-extensions-v29.json'],
        allowed_new_source_rule='Only specificoperatorcallee/kernel/tiling orhardwarethroughput evidence that changes fusion legality or a theoretical bound. No independentHTTP/API/framework branch research.',
        before_user_steering_reads=dict(input_index='sampling-guard-inputs-v55.json',source_files_read=9,missing_generation_config=True,
            followup_generation_or_plugin_tree_stopped=True,role='Preservedsupportingmaterial;not described as completedoperatoranalysis.'),
        no_model_torch_NPU_newcapture_service_health_process_work=True,whole_module_closed=False,overall_complete=False))
    source=read(O/'sampling-guard-inputs-v55.json')
    write(O/'pre-steering-source-disposition-v55.json',dict(scope='Beforetheuseroperator-focuscorrection,9boundedsource/configfileswerecopied and1generationJSONwasunavailable. These are retained withoutcontinuing framework/default/plugin research.',
        inputs=str(O/'sampling-guard-inputs-v55.json'),all_source_semantics_reviewed=False,
        missing_file_not_neutral_default_proof=True,original_operator_goal_not_completed_by_this_read=True,overall_complete=False))
    updates={}
    for cid in ['F04','F20']:
        updates[cid]=dict(review_v55=dict(profile=str(O/e[4]),fusion_theory=str(O/e[1]),shape_ledger=str(O/e[2]),focus=str(O/e[0]),
            relation='Exact3200graphblocks bind norm/router/topK shapes;2560root-associated and640phase-unbound keptseparate. Existingnorm+castalreadyfused. ConditionalFP32plane andscoreinterface removal has8MH/8MElogicalbyte accounting;M3vsM800 differ strongly. No cache/FP32peak/nativegain claim.',implementation=False,native_test=False,new_performance_result=False),
            profile_review_state='OPERATOR_CHAIN_FUSION_AND_SHAPE_CONDITIONAL_ROOFLINE_ANALYZED',
            conclusion_changing_gap='Operator-only:all live consumers,nativeFP32router precision/tiling/cachelevel/capacity andsamecallrowpadding/generation/event obligations. Do not repeat alreadyknownroundedBF16→FP32relation orRMS+castbaseline as missing/newfusion. No framework/APIexhaustion is required as an independentresearchgoal.',
            next_profile_or_contract_action='Price exactoperatorchains andtheir transformedwork/traffic. Next prioritizeMC2/expert subgraph andprojection/attention/indexer operators,using2082regimeledger. Point-readonlyneededkernel/tiling/capacitycontracts;no broadframeworkdefault audit.')
    modules={'M12':dict(finding='Operatorfusion/ceiling focus:3200exactnorm→FP32router→topKgraphtriples,2560root-bound. M3medians3.16/17.36/4.82us;8MH+8MElogicalinterfaceelimination differs strongly fromlargeM. ExistingRMS+castalreadyfused;newboundaryrequiresroundedBF16,FP32weight/nonlinear/hash andliveBF16consumerpreservation.',
        evidence=e,candidates=['F04','F20'],gap='AllMoE/expert/communication variants andtypedphysicalcapacities/caches/necessarywork/lifetimes remain;publishedreference roofline is conditional,not measuredspeedup.')}
    prior=read(O/'module-coverage-gaps-v54.json')['remaining']
    write(O/'prior-coverage-gaps-before-operator-focus-v55.json',dict(previous_remaining=prior,
        scope='Preservealloperatorobligations;userclarifiedthatindependentframework/APIresearchisnotdeliverable. Thefollowingoperator-focusedgaps supersede oldmixed researchpriority,not missingoperator work.',overall_complete=False))
    remaining=[
      'Operatoruniverse stillincomplete:83observedoptypes/2082shape-role-context regimes/286950ComputeInfoexecutionsareinventoried,notallfusion decisionsortheoryclosed. Preserveall modeloperator branches andcrossoperator/layer/card/iteration boundaries;unobservedconditionalmodelpaths require explicit applicability.',
      '35modelcandidate variants need concreteoriginal/transformedsubgraphs,alltensorconsumers,numerics/ownership/synchronization/capacity andsame-denominatorcost. Ledgerexistence,candidateindexing andreportfinalization do not provecompletion.',
      '87plain2Ddense regimes nowhave2MNK/operand-result byte andpublishedreference roofline models. Remainingregimes need operator-specific necessarywork/IO/communication models;do notapplyGEMM formula tofractal/custom/comm tasks. 6belowcoldreference rows refute assumptions,not over100percenthardwareefficiency.',
      'Actualphysicalcapacity binding remains:reference910Cdie24Cube48Vector/64GB has376BF16TF/752INT8TOPS/1.6TB/s inHuawei paper;currentcachedclock1800MHz andnodeSKU/cachetopology must not be silentlymappedto that reference. FP32routerpeak/actualHF32mode andL2/HBMtraffic remainunproved. No BF16peak forFLOATrouter/MFU or inventedclockscale.',
      'F04/F20chain:existingRMS+castdualfused4472instances;graph3200M3triples (2560root/640unbound). Conditionalrouterwiden preservesroundedBF16,FP32gateweights/nonlinear/bias/hash/topK andliveBF16expert/shared consumers. M3logicalinterfaces132096B versus7864320Bweights;M800FP32interface32768000B. Coldreference transfer valuations are not physicaltraffic/criticalgain. M800norm versusM816routing padding isknownbutnot samecallpair bycount/time.',
      'MoE/MC2operatornext:DispatchFFNCombine,expertGEMMs/dequantSwiGLU/combine andsharedbranch need exact EP16route/validrow/work-wait/tiling/weightandactivationbyte models. Keepstatic/dynamicpayload/mapconditions andallnamedplane lifetimes frompriorF21;do notcontinuecheckpointloader/framework trees unless a namedoperatorchange needs it. CurrentMC2==1sharedmultistream disabled baseline isknown,notfreeoverlapgain.',
      'Projection/attention/indexer chains:Q/KVnorm-quant-RoPE-cache,outputprojection/collectives,C1/SWAjointGEMM andindexWAR/read-publication contracts alreadylocalized. Completeotherconsumers/nativeprecision/banksandtypedcapacity models. F33projectiongaps withotherworkoverlap are notidle;separateweights remain required.',
      'Compressor/KV/state:currentC2pool/update/norm andF34unchangedSWA-anchorallocation exclusions areknown. Completeoperatorrowvalidity/slot-generation/CPmetadata/otherobservers andmemory-tierbounds beforepruning/repacking/crossiterationreuse claims. Sameaddress/shape isnot value/stateidentity.',
      'Engram/UVA path:176prezero full-overwrite selectedpath,TProw/headexchange/cachepriority/writefootprint questions answered. Retainallotherbanks/consumers/historyrequest-slot/weightgeneration/nativeprecision andhost/NPUmemorytierbounds;48nativeTASK/32segments unbound retained,notreassignedbyancestry.',
      'Finalnorm/head/sampling/DSparkoperatorchains:F26/F30/F31/F35 haveboundlocalconsumers andalgorithms;completepartition/validrow/map/nativeprecision/generation andmandatoryinternalprobability/entropy/recovery readers. Newcachedsampler/defaultfiles are supportonly;no broadrequest/APIparentaudit. Greedycurrent positives do notexclude fullmodelalgorithmbranches.',
      'Evidenceuncertainty retained:849missingPMUgroups,16phase-unboundgraphs,3136module-membershipambiguousComputeInfoexecutions,2082regimes. CPU/frameworkwork isonlyinvestigated whenitcontrols a concreteoperatorfusion dependency orcriticalpath;independentHTTP/SSE fullbranchclosure isnot an operatorgoal.',
      'Allfulloperator/module theoreticalcomparisons stillnotclosed. Conditionalanalyticdecision neednotwaitforimplementation/nativebenchmark,whileactualunanalysedoperators/branches/consumers/keybounds must remainopen. No all-scope completion claim.'
    ]
    action=(
      '继续原Goal，严格按用户最新纠正：目标是算子能否融合/拆分/重组及理论上限，不研究框架。先读OPERATOR-FOCUS-REQUIREMENTS-v55、F04-router-fusion-roofline-v55、operator-shape-ceiling-ledger-v55和PROFILE-FINDINGS-v55。'
      'v54有实际参数/consumer进展，但优先级偏框架；v55已回归operator。此前9源码/config读取保留为背景，generationJSON缺失不能推默认1，不再追HF/plugin/API树；仅当影响命名operator方案才回看。'
      '本轮从完整原32DB导出286950执行/83算子类型用时、3200个图层norm-router-topK exactTASK三元组；用输出shape3x384筛选FLOATrouter，不能把同block其它MatMulV3也当router。2560rootbound/640unbound保持。'
      'RmsNormCast M3=3.16us/3blocks，router=17.36us/24blocks，gating=4.82us/3blocks；rootbound三核sum中位25.601us，span33.34us，gap7.66us含其它工作不能称idle。4472norm所有shape仍有M2/M800/M816/M834。'
      '理论model：norm4MH+M普通算术+Mrsqrt，双输出逻辑bytes8MH+2H；可移除FP32接口write/read8MH，logits接口8ME。M3共132096B比routeweight7864320B小，coldreference1.6TB/s对应0.08256us只是假设流量估值；M800单FP32接口32768000B/20.48us。实际cache/HBM流量/native合法性/临界路径不由这些数证明。BF16专家/shared输出必须留，FP32是roundedBF16widen，非unroundednormacc。'
      '原资源fulljson43.6MB已从原166缓存并核历史c8e2hash；新完整2082regime台账ID唯一且执行/PMU缺守恒，87个plain2D dense记录2MNK及条件reference roofline，6行低于cold假设floor不算hardware>100%。不将模板IOextent当自定义算子必写流量。'
      '硬件仅引用Huawei主论文CloudMatrix参考die(24Cube48Vector64GB,376BF16TF/752INT8TOPS,1.6TB/s)，当前1800MHz已缓存；FP32peak/actualSKU/cache/native模式未绑定，不用BF16peak算FLOATrouterMFU，不猜clockscale或端到端gain。'
      '下一步先按算子推进DispatchFFNCombine/专家GEMM-激活-通信子图的work/traffic/route/cut模型，再投影/attention/indexer/Engram/head等；点读的是相应kernel/tiling/数值/consumer/同步，不是框架生命周期。跨层/跨卡/跨迭代继续覆盖operator依赖和数据复用。'
      '原缓存150文件+v55资源fulljson优先本地使用。若必须远端读已有文件，新认证ssh166-v55-xinyufeng.sock验证166/host105root+4corehash；旧helper默认v53已过期，须在新script显式override transport.CONTROL，不改已pinhelper；不silentfallback其他主机。无模型/Torch/NPU/profile/health/process/kill/实现，Goal active。35model候选保留但不以数量完成；849PMU/16phase/3136歧义和各operator真实未分析范围保留。')
    model3=next(r for r in roof['shape_models']if r['M']==3)
    report=(
      '本轮已按用户纠正，将主线限定为算子融合、拆分、重组及理论上限。已有框架材料只作为必要的数据依赖或数值条件，不继续独立展开HTTP、SSE、请求与框架生命周期。\n\n'
      '完整profile现按83种算子、2082个shape/role/context组合建立台账，覆盖286950条ComputeInfo执行。87个满足形状条件的二维dense矩阵组合已填入2MNK计算量、逻辑操作数/结果字节量和条件Roofline；其余明确保留待建模，不以有表格代替分析完成。\n\n'
      '本轮算子链：RmsNormCast → FLOAT路由MatMulV3 → MoeGatingTopKHash。3200个图层通过原始block/TASK身份绑定，其中2560有root关联，640仍phase-unbound。以下M=3,H=5120,E=384：\n\n'
      '| 算子 | 中位设备耗时 | blockNum |\n|---|---:|---:|\n| RmsNormCast | 3.16μs | 3 |\n| FP32路由GEMM | 17.36μs | 24 |\n| TopK/Hash | 4.82μs | 3 |\n\n'
      'root关联链的三核累计中位为25.601μs，首尾span中位33.34μs。间隔包含其它ID/control/重叠工作，不是可直接删除的空闲。block数也不是实测物理利用率。\n\n'
      '融合结论：RMSNorm+双输出cast已经融合；后续可条件地将BF16结果在router内部widen为FP32，减少独立FP32平面；router GEMM与非线性/偏置/hash/TopK也可作为融合边界。必须保留BF16舍入后再widen的数值边界、FP32权重以及专家/shared分支的BF16消费者。不能直接删BF16平面、换成未舍入norm中间量、跨层复用同shape结果或按top-k不变性取消norm。\n\n'
      '| 逻辑接口 | M=3 | M=800 |\n|---|---:|---:|\n| FP32 norm平面写入+读取，8MH | 122,880B | 32,768,000B |\n| router logits写入+读取，8ME | 9,216B | 2,457,600B |\n\n'
      '小M两项仅132,096B，约为7,864,320B路由权重的1.68%；单靠省这部分访存，不能预期大收益，应重点看启动、归约和小M tiling。大M的中间平面更大，但800→816的padding域必须保留，不能按相邻时间或数量强配同一调用。上述是逻辑接口估值，cache可改变真实HBM流量。\n\n'
      '采用华为论文给出的910C参考die参数，BF16/FP16约376TFLOPS、INT8约752TOPS、HBM约1.6TB/s；它们用于条件模型，没有冒充当前服务器的实测容量。FLOAT router不能套用BF16峰值。模型公式为T≥max(F/P_dtype,Q_HBM/B_HBM,Q_L2/B_L2,其它pipeline/依赖下界)，每项必须确认资源与分母。M=800的FP32接口按全部穿过HBM估值20.48μs，不是实测可省时；M=3两接口同条件仅0.08256μs。[硬件原始论文](https://arxiv.org/html/2506.12708v1#S3.SS3.SSS1)\n\n'
      '下一步优先分析DispatchFFNCombine及专家GEMM、激活、dispatch/combine通信子图，再推进投影、attention、indexer、Engram与head/sampling链。整体算子覆盖与理论上界尚未完成，继续保持进行中；没有实施优化或新增设备测试。\n')
    source_paths=[Path(r['local_path'])for r in source['files']if r['status']=='READ']
    cache=read(O/'operator-shape-resource-cache-v55.json')['files'][0]
    jobs=[
      dict(kind='review',status='error',description='OldownedSSHmuxexpired;localguardstoppedbeforethefilequery. Newauthorizedsameaccountmux/root/corehashverified;nofallback.',artifacts=[O/'expired-owned-mux-reader-error-v55.json']),
      dict(kind='review',status='pass',description='Bounded authorizedoriginal166reauth+9source/config readscompletedbeforeusersteering;1generationJSONunavailable,notneutraldefaultproof. Followupframeworktree stopped;materialretainedonlyasbackground.',artifacts=[N/'original166-xinyufeng-recheck-v55.json',N/'reauth_original166_v55.py',N/'read_sampling_guard_sources_v55.py',N/'sampling-guard-source166-v55.json',O/'sampling-guard-inputs-v55.json',O/'pre-steering-source-disposition-v55.json']+source_paths),
      dict(kind='review',status='error',description='FirstoperatorchainselectorusedoptypealoneandencounteredotherMatMulV3withinablock;corrected exactFLOATroutershape3x384. Notabsence/nearest-timeproof.',artifacts=[O/'operator-chain-selector-error-v55.json']),
      dict(kind='review',status='error',description='Adhocdurationdisplayexpectednonexistent sumfield;correctedbyrecomputingsummednativeTASKdurationsfromfullpopulation,nofabricatedsum.',artifacts=[O/'operator-inspection-display-error-v55.json']),
      dict(kind='profile',status='pass',description='All286950nativecomputes/83optypes plus3200exactnorm-router-topKgraphblocks andselectedPMUbound;2560rootbound/640unbound separated. No newcapture.',artifacts=[O/e[4],N/'analyze_operator_fusion_roofline_v55.py']),
      dict(kind='design-check',status='pass',description='Exactshape logicalwork/traffic and conditionalfusion roofline;smallMvslargeM,roundedBF16/livefanout/padding/nonlinear andnonadditiveinterval constraints preserved. Publishedperdie reference is not actualFP32peak orMFU.',artifacts=[O/e[1],O/e[5],O/'F04-precision-routing-and-row-contract-v29.json']),
      dict(kind='review',status='pass',description='2082stableoperatorregimes/87conditionaldensemodels conserveexecutions/PMUgap;operator-onlyuserfocus andcomplete remainingoperator frontier reconciled. No maincandidate/modulecompletion/nativegain.',artifacts=[O/e[0],O/e[2],O/e[3],O/'operator-shape-resource-cache-v55.json',Path(cache['local_path']),N/'build_operator_scope_and_ceiling_ledger_v55.py']),
    ]
    persist_refinement(version=55,previous=54,title='Operator fusion and shape-specific theoretical ceilings',
      summary=dict(filename='operator-fusion-and-ceiling-summary-v55.json',previous_turn_classification='PROGRESS:v54request_provenance_and_output_consumers',
        user_focus='Operatorfusion/split/reorganizationandtheoreticalceilings;notindependentframeworkresearch',
        operator_types=83,shape_role_context_regimes=2082,native_compute_executions=286950,conditional_dense_roofline_models=87,
        graph_operator_chains=3200,root_bound_chains=2560,phase_unbound_chains=640,
        small_M3_FP32_interface_bytes=122880,small_M3_logit_interface_bytes=9216,large_M800_FP32_interface_bytes=32768000,
        hardware_reference_not_actual_capacity=True,target_implementation_changed=False,new_performance_result=False,
        remote_job_status='Authorizedsame166reauth andsavedfile/dockercp readscomplete;alloperatoranalysis local. No newmodel/Torch/NPU/capture/health/process/kill/implementation. Expiredmux,selector anddisplay errors retained;noanalysisjob left running.'),
      changes=['Useroperator-onlyfocusapplied;independentframework/APIresearchstopped,existingbackgroundevidencepreserved.',
        'Fullprofile83operators/2082regimes inventoried,87conditionaldensework/traffic roofs;3200exactnorm-router-topKchainspriced.',
        'SmallMvslargeMinterfacebytes andconditionallegalfusion/tiling bounds distinguish actualopportunityfromalreadyfusedbaseline;noactualcapacity/MFU/gain invention.'],
      candidate_updates=updates,framework_updates={},module_updates=modules,remaining=remaining,action=action,jobs=jobs,report=report,
      read_first=e,publication_rows={},gate_updates=dict(overall_complete=False),
      scripts=[N/'analyze_operator_fusion_roofline_v55.py',N/'build_operator_scope_and_ceiling_ledger_v55.py'],
      executor_prefix='zcode-operator-ceiling',main_script=Path(__file__),source_read=True)


if __name__=='__main__':main()
