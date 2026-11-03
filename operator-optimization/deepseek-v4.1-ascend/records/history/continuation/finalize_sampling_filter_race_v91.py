"""Persist operator cutoff/race proofs,typed costs,and a partial G7 graph supplement."""
from pathlib import Path
from copy import deepcopy
import hashlib
from finalize_origin_coverage_v9 import N,O,R,read,write
from persist_profile_refinement_v90 import persist_refinement


def main():
    assert not(O/'REVIEW-CHECKPOINT-v91.json').exists()
    assert read(O/'sampling-filter-race-analysis-preflight-v91.json')['status']=='PASS'
    names=['sampling-filter-race-source-contract-v91.json','sampling-filter-race-algebra-v91.json','sampling-filter-race-cost-models-v91.json',
      'sampling-filter-race-pointwise-validation-v91.json','operator-graph-G7-v91.json','operator-graph-node-coverage-v91.json',
      'operator-graph-edge-coverage-v91.json','operator-graph-candidate-bindings-v91.json','operator-graph-coverage-audit-v91.json']
    variants=[
      dict(id='F35-v91-CUTOFF-MASK',state='POINTWISE_CONDITIONAL_FUSION',change='RetainoriginalP/cutoffs,combine two-infmaskedwrites byOR;withorderednonNaNcutoffs,useoneP<max(cK,cP).',
        gates='SameoriginalP/sort/cutoff/CDFvalues,originalvalidationandnobetween-writesobserver. Do notsubstituteupstreamsequentialfilter.'),
      dict(id='F35-v91-CDF-SEARCH',state='EXACT_EXISTING_VALUE_CONDITIONAL_REORGANIZATION',change='ExistingorderednonNaNCDF:count=min(upper_bound(CDF,actual1-p),V-1). Avoidfullbooleanmask/sumforcount.',
        gates='PreserveactualCDFvalues/scanroundingandthreshold;nonmonotonic/NaNCDFrequiresoriginallogic. Newdependentsearchloadlatencyunpriced.'),
      dict(id='F35-v91-K-SELECTION',state='CONDITIONAL_ALGORITHM_SUBSTITUTION',change='TopKonly:selectsameprobabilityorderstatistic,retainallboundaryties;allkV/pNonecanbeexplicitmetadata-noop.',
        gates='Roundedprobabilitydomain,validmetadata/errors andsameall-tiebehavior;notexactk/logitTopKwithoutproof.'),
      dict(id='F35-v91-DIV-ARGMAX',state='POINTWISE_FIXED_OPERAND_CONDITIONAL_FUSION',change='FuseexistingP/Qdivisionwithargmaxstream,omitratiointermediateifPprivate;withgenericPmutationkeepwriteback.',
        gates='SameP/Q/FP32division/comparator,Qstreamwait/record_stream andallRNGcalls. NoimportedQzero/Inf recoveryguards.'),
      dict(id='F35-v91-MIXED-RACE-DEMAND',state='CONDITIONAL_CONSUMER_PROJECTION',change='KeepfullB-sizedQdrawsandotheroutputs,computeprivatePDiv/argmaxonlyrowswhosefinaltemperature<1e-5selectorusesrandomID.',
        gates='ExactbonusSamplerpredicate/flagsandrowplan;notautomaticallyrejectionEQ0mask. Fullscore/processorandgeneratorstatesretained.'),
      dict(id='F35-v91-SOFTMAX-RACE',state='UNPROVED_NATIVE_REDUCTION_FUSION',change='Potentiallystreampostfiltersoftmax→Div→argmaxandremoveP/ratiointerfaces.',
        gates='Whole-rowUBplanfails;pricepasses/tiling,exactnativeProunding/reductions,returnedscoreandaliasrequirements. ReusingzeroedoriginalP hasFP32winnercounterexample.'),
      dict(id='F35-v91-LOGPROB-STATS',state='UNPROVED_NATIVE_LOGSOFTMAX_STATISTICS_SHARING',change='Potentiallysharestatsforprocessedlog_softmaxandpostfiltersoftmax.',
        gates='log(roundedP)isnotlog_softmax;keepunderflow/precision,fullreturnedbankandsamefilteredvalueepoch. Noautomaticlosslessconversion.')]
    write(O/'sampling-filter-race-fusion-variants-v91.json',dict(candidate='F35',variants=variants,new_maincandidate=False,implementation=False,native_test=False,actual_speedup=None,overall_complete=False))
    rel=deepcopy(read(O/'candidate-relation-clauses-v90.json'))
    for row in rel['clauses']:
        if row['id']in ['REL73-22','REL73-26']:
            row['filter_race_refinement_v91']=dict(evidence=names,
              scope='Keepglobalroundedprobabilitycutoffs/ties andRNG/row/bankepochs;localepiloguecasesdonotprovepartitionednormalizationorhead/private-readerprotocol.',
              new_pair_or_all_variants_closed=False,critical_path_gain_us=None)
    rel.update(parent_relation_index='candidate-relation-clauses-v90.json',overall_complete=False);write(O/'candidate-relation-clauses-v91.json',rel)
    cov=deepcopy(read(O/'candidate-relation-coverage-v90.json'));cov.update(parent='candidate-relation-coverage-v90.json',parent_relation_index='candidate-relation-clauses-v91.json',
      filter_race_refinement_v91=dict(new_pairs=0,new_groups=0,evidence=names),overall_complete=False);write(O/'candidate-relation-coverage-v91.json',cov)
    fps=deepcopy(read(O/'candidate-operator-footprints-v90.json'));index={r['id']:r for r in read(O/'candidate-record-index-v90.json')['model_candidates']}
    for row in fps['candidates']:
        q=R/'candidates'/(row['id']+'.json');actual=hashlib.sha256(q.read_bytes()).hexdigest();assert actual==index[row['id']]['master_sha256'];row['captured_master_sha256']=actual
        if row['id']=='F35':
            row['operator_atom_union']+=['sampler.original_P_cutoff_fork','sampler.existing_CDF_upper_bound','sampler.privateP_Div_argmax','sampler.mixed_LT_EPS','sampler.Q_default_seeded_effects']
            row['G7_source_binding_v91']=dict(graph='operator-graph-G7-v91.json',node_rows=29,edge_rows=40,all_F35_primitives_or_variants_mapped=False)
    fps.update(parent='candidate-operator-footprints-v90.json',overall_complete=False);write(O/'candidate-operator-footprints-v91.json',fps)
    nav=deepcopy(read(O/'candidate-pair-navigation-v90.json'));fpmap={r['id']:r for r in fps['candidates']}
    for row in nav['pairs']:row['shared_operator_atom_union']=sorted(set(fpmap[row['a']]['operator_atom_union'])&set(fpmap[row['b']]['operator_atom_union']))
    nav.update(parent='candidate-pair-navigation-v90.json',filter_race_refinement_v91=dict(new_pairs=0,new_groups=0),overall_complete=False);write(O/'candidate-pair-navigation-v91.json',nav)
    op=deepcopy(read(O/'operator-analysis-coverage-ledger-v90.json'));op['sampling_filter_race_refinement_v91']=dict(evidence=names,
      source_files=3,named_functions_pinned=11,source_if_sites=27,source_if_arms=54,source_guard_counts_not_whole_model_coverage=True,
      local_hw_cap_filter_branches_equivalent=True,mask_cases=7650,CDF_cases=6375,race_cases=729,new_profile_regimes_or_executions=0,
      actual_random_path_or_batch_invariant_or_parameter_flags_attested=False)
    write(O/'operator-analysis-coverage-ledger-v91.json',op)
    branches=deepcopy(read(O/'M17-probability-branch-ledger-v90.json'));branches['rows'] +=[
      dict(branch='Ascend_default_originalP_k_p_filter_and_race',state='SCOPED_CUTOFF_RACE_AND_EFFECT_CONTRACT_REVIEWED',evidence=names[:4]),
      dict(branch='NPU_TOP_K_TOP_P_capability_choice',state='LOCAL_SOURCE_BRANCH_EQUIVALENCE_PROVED',evidence='BothselectedAscendfunctions call samePyTorchprobabilityfilter inthissnapshot;notfusednpuoperatorproof.'),
      dict(branch='batch_invariant_upstream_filter',state='DISTINCT_PYTORCH_CONTRACT;TRITON_BODY_OPEN',evidence='Upstreamsequentialk→P/Tritonselection cannotbeassigneddefaultAscendoriginalPcutoffsemantics.')]
    branches.update(parent='M17-probability-branch-ledger-v90.json',branch_universe_exhaustive=False,whole_module_closed=False,overall_complete=False);write(O/'M17-probability-branch-ledger-v91.json',branches)
    gap=('Actualrandom/topK/topP/batch-invariant/logprob/generatorflagsandnewCDForderedness/nativekernelmapping remainunbound. '
      'PreserveoriginalP/scan/cutoff/rounding/ties,raw/processedscoreandmutationobservers,allQdraws/objectidentities/streamwaits andbankepochs. '
      'Sourcefixedoperand/IEEEcasesdonotproveactualCANNprecision,RNGdrawsupport,mixedissue,physicaltraffic,launchlatency orspeedup. UpstreamTritonfilterbodyandremainingprocessors/readers remainseparateopenquestions.')
    matrix=read(O/'module-phase-branch-matrix-v90.json');m17=deepcopy(next(m['branches']for m in matrix['modules']if m['id']=='M17'))
    for row in m17:
        if row['id']in ['M17-OP-04','M17-OP-07','M17-OP-09']:
            row['evidence']+=names;row['filter_race_review_v91']=dict(default_originalP_cutoff_semantics=True,hardwarecap_alias_contract_resolved=True,
              mixed_LT_EPS_predicate_is_not_automatically_rejection_EQ0=True,nativeflags_and_allreaders_proved=False);row['gap']=gap
    m17.append(dict(id='M17-OP-12',branch='DefaultAscend original-probabilityk/p cutoffs,allboundaryties andpostfilterP/Qrace:merge-mask/CDFsearch/probabilityselection/privateDivArgmax;upstreamfilter andRNG/mixed-row contracts distinct.',
      state='PINNED_OPERATOR_CUTOFF_RACE_AND_CONDITIONAL_RESOURCE_DOMAINS_REVIEWED;WHOLE_MODULE_OPEN',evidence=names,
      gap=gap,scope='NamedNPU-nativeentry anddirectcomparison/helperboundaries;unreviewedTriton/otherprocessors notimplicitlyclosed.'))
    theory=deepcopy(read(O/'module-theory-capacity-coverage-v89.json'));row=next(m for m in theory['modules']if m['module']=='M17')
    row['extra_source_instruction_domains']=['FLOATCompare/Div declaredcutoff/racecases','ConditionalSort32firstpass,notallalgorithmSortbound','ExistingCDFcomparison-queryreference,notphysicalrate']
    row['state']='SCOPED_TYPED_INSTRUCTION_CASES_ADDED';theory.update(parent='module-theory-capacity-coverage-v89.json',this_version_new_modules=['M17'],refined_modules=11,
      outstanding=gap,whole_modules_closed=0,overall_complete=False);write(O/'module-theory-capacity-coverage-v91.json',theory)
    mids=['M16','M17','M21','M22','M24'];mods={mid:dict(
      finding='Operatorcutoff/race analysis separatesdefaultAscendoriginalPthresholds fromupstreamfiltersemantics,provesconditionalmask/CDF/ratio transformations andretainsRNG/row/outputepochs. GraphG7 supplements partialsourcecatalog;no native attributionchange.',
      evidence=names+['sampling-filter-race-fusion-variants-v91.json'],candidates=['F35'],gap=gap,
      profile_scope='Probabilityfilter/raceelementwise/reduction/selection/statecontracts;no independentframeworkresearch.')for mid in mids}
    update=dict(review_v91=dict(evidence=names+['sampling-filter-race-fusion-variants-v91.json'],
      defaultfilter_source_contract='OriginalPforbothk/p;keepallprobabilitycutoffties',local_hw_cap_alias_contract_resolved=True,
      fused_mask_and_existing_CDF_search_conditionally_proved=True,privateP_ratio_argmax_conditional=True,
      numerical_fences=['postKrenorm/filterbackendnonidentity','probabilityties/underflow','FP32common-scale-racewinner','logroundedPnotlogsoftmax','Qguard/RNGstream/mixedpredicate'],
      G7_source_rows=[29,40],all_F35_primitives_mapped=False,implementation=False,native_test=False),
      conclusion_changing_gap=gap,
      next_profile_or_contract_action='Do notrepeatdefaultPthreshold/seedguard discoveries. PrioritizeapplicableupstreamTritonfilterbody precision/tie/topP contracts orstillunclosedprocessor/readereffects;onlynamedoperatorquestions. Currentprofiledoesnotattestrandomflags. KeeponebaselineandRNGepochs;no automaticbackend substitution or physicalspeedupclaim.')
    write(O/'PROFILE-CODE-QUESTION-GATE-v91.json',dict(status='PROFILE_LED_DIRECT_FILTER_RACE_OPERATOR_QUESTIONS',
      questions=['AredefaultAscendandupstreamcutoffcontracts interchangeable? No.',
        'Canunchangedcutoffs feedonewriter andexistingCDFfeedsearch? Conditionalpointwiseproof.',
        'CanexistingP/Qrace avoidratiointermediate orunusedmixedrows? PrivateP/identicalvalues/RNG/streamconditions.',
        'Doesprobabilitynormalizationcancellation orlog(P) permitunconditionalsoftmax/logprob fusion? No.'],
      cached_pinpoint_source_files=3,named_functions_pinned=11,local_source_if_sites=27,local_source_if_arms=54,
      source_count_is_navigation_not_full_backend_closure=True,new_target_source_files=0,new_public_document_read=False,new_remote_source_read=False,
      no_framework_wide_research=True,no_Torch_model_NPU_capture_or_server_work=True,overall_complete=False))
    write(O/'sampling-filter-race-question-status-v91.json',dict(newly_resolved=[
      'BothNPUcapabilityselector arms leadtosameprobabilityfilter;runtimecapflagnotneededfor thislocalidentity.',
      'DefaultPcutoffs shareoriginalP;topK-topP maskcanmergeunderexactvaluecontract.',
      'Cutoffties/postKnormalization differfromupstreamPyTorchfilter.',
      'ExistingorderedCDFcount canuseclippedupper_bound withoutrewritingscan.',
      'PrivatePDivArgmax/mixedunusedrandomID paths haveexplicitoperand/effect/returncontracts.',
      'AllseededQomitsdefaultdrawhere;do notreusealways-defaultrejectionQrule.'],
      unresolved=[gap,'SourceTritonfilterbody/genericprocessor/logprob primitives andallremainingcrosscandidateobservercontracts.'],
      theoretical_work_traffic_and_native_time_not_conflated=True,whole_modules_closed=0,overall_complete=False))
    remaining=list(read(O/'module-coverage-gaps-v90.json')['remaining'])
    remaining.append('v91:3frozenmanifest-boundfiles11namedfunctionpins27if54armsnavigation,notwholebackendclosure. AscenddefaultoneoriginalPsoftmax/sort feedsbothk/p,valuecutoff keepsallties;upstreamPyTorchpostKrenorm/positionmaskdifferent. HWcapbranchesbothsamePYfilterlocallyresolved;batchInvariant/HAS_TRITONbodystillseparate.7650mask6375existingCDFsearch729racecases+8numeric/domainwitnesses.20costmodels/8RNGcases,B2/4V129280;mask8BVandprivateDivArgmax8AVinterfacecases,notactualHBM orspeedup. Qallseeded skipsdefaultdrawhere,globalstreamwait/record/objectorderretained. OrderedCDFupperbound17probes/row retainsnativeCDFvalues. G7source29/40,currentpartial708/1159rows;M17OP12=>353records24semantic.115/24/480 and2073/285598review9/1352pending849/16/3136 unchanged0wholeclosed.')
    report='''本轮从保存的采样profile范围出发，定点分析默认Ascend的top-k/top-p过滤、概率除指数噪声及argmax，并核对直接上游对照函数。三个文件均匹配已有冻结源码清单；当前profile仍未提供随机分支、k/p、batch-invariant或seeded参数的逐调用取值，没有新增服务器或NPU执行。

默认Ascend实现先做一次softmax，再将概率升序排序。top-k和top-p阈值都来自这份未变的原始概率；top-k写logits并不会更新用于top-p的概率。这个softmax和sort本来已共享，本轮没有将它们再记为新融合。

上游PyTorch实现不同：先按logits排序并截断top-k，再对截断结果softmax计算top-p，最后按排序位置mask。权重[1,2,3,4]、k=2、p=1/2的实数例子中，默认Ascend保留后两项，上游只保留最后一项；四项同概率、p=1/2时，Ascend的概率阈值比较保留全部同值项，上游位置mask只保留两项。同名函数不能直接替换数值合同。硬件能力选择在当前Ascend源码中的两个分支，实际上都进入同一个PyTorch过滤体；这项局部选择已厘清。batch-invariant会委托上游，Triton过滤器体仍需独立核对。

有明确可分析的融合范围。保留相同原P、sort、CDF和两个阈值后，两次masked_fill(-inf)等价于按两个条件OR进行一次写入。阈值有序且非NaN时，可进一步比较P<max(cK,cP)，减少一次整矩阵比较。无效元数据的原验证、gather错误顺序及中间读者条件仍需保留，不能趁此换成不同top-k/top-p语义。

top-p计数也可重组：若保留的既有CDF确实非递减且无NaN，原sum(CDF<=1-p)并将最后mask置False，等于min(upper_bound(CDF,同一阈值),V−1)。V=129280时，二分最多17次值探查/行。它保留原CDF的数值，不要求先改写cumsum；如果融合新的scan生产者，原生归约顺序和边界值还须证明。依赖读取的延迟也可能抵消比较数量下降。

top-k-only可研究在相同已舍入概率上找顺序统计量，替代完整排序并保留全部边界同值项；不能直接使用精确保留k项的logit TopK。所有k=V且p缺省，可在元数据与验证条件成立时作为no-op分支。p=1却不总能同样处理：FP32 softmax出现零概率时，当前实现可能把对应有限但极小的logit改为-inf；[-200,0]的数值模型即展示这一边界。

默认随机采样在global_stream生成指数Q，当前流等待并record_stream后，对probs原地除Q再argmax。保持同一P/Q、逐值除法和比较规则，可将Div与argmax流式融合，省去ratio中间平面。直接公开调用random_sample(P)有P被修改的语义，须保留写回；在forward内部P私有的范围，可不落地这个平面。processed_logits仍需过滤后的logits别名；processed_logprobs是独立log_softmax结果，不能用log(已舍入softmax)代替，后者在下溢时会变为-inf。

不能仅凭实数公因子消去最后一次softmax。两个保留权重为[2,3]时，原归一化总量6与截断后总量5产生的FP32 P不同；本轮固定正值Q数值例子中，复用原P得到并列第一，而重新归一化的正确P使第二项胜出。该例子是IEEE单步舍入模型，不冒充CANN输出或可复现的原生RNG种子。

RNG合同与拒绝恢复路径也不能混用：这里全部请求都有generator时跳过默认整银行draw，否则先draw全部B×V，再按字典顺序覆盖seeded行；v33的拒绝恢复wrapper则总会先draw默认Q。generator对象可能共用状态，不能仅凭相同seed重排。Q为0/Inf时，这里的源码没有恢复kernel的sentinel防护，也不能将那个防护直接移入。

mixed bonus采样最终按temperature<1e-5选择greedy或random ID。保留所有原Q draw和其他score工作后，私有P的Div/argmax可以只处理最终使用random ID的行。该谓词与拒绝helper的temperature==0不能在未证明调用者温度域时合并；小正温度是否能经上游正规化到达这里仍未认定。已存在的all_greedy提前返回也不重复计收益。

成本模型覆盖B=2/4、V=129280的20个声明参数/活动行场景及8种RNG draw情形。只看两段互不重叠的接口：同时有k/p时，两次logit读写参考16BV→8BV；P/Q已存在的race参考16BV→私有融合8AV，generic写回方案为12BV。softmax、sort/merge、CDF、返回分值、Q生成/存储、stream等待和新增launch均另计。这不是实测HBM流量或耗时。在声明的Sort32实现域内，完整首遍排序服务还在，elementwise融合不能凭空消掉它；换选择算法另建成本域。

检查通过7650组精确概率掩码用例、6375组保留CDF值的搜索用例、729组固定操作数race用例，并记录8项数值/域反例。均为源语义与数学验证，未导入Torch或进行NPU测试。

G7补上29行默认过滤/race节点、40行边；当前G1/G5/G6/G7部分图708/1159行已与覆盖入口绑定。计数不是唯一算子或全分支覆盖率。累计353条分支记录、24个语义模块；115条条件关系、24组联合合同，中央仍有480对待汇入或复核。2073个regime/285598次执行已作条件分析，9个regime/1352次执行仍待补；849条PMU缺失、16张阶段未绑定图、3136条模块归属歧义不变。整体目标继续，没有整模块关闭。
'''
    action=('继续原Goal，v91PROGRESS，v90亦PROGRESS。读sampling-filter-race-summary/PROFILE-FINDINGS-v91、source-contract/algebra/cost-models/pointwise-validation/fusion-variants/question-status与G7/coverageaudit/node/edge/candidatebindings。'
      '新证据默认Ascend initialP一次softmax/sort本已共享，k/p都用原P，strictprobabilitycutoff保allties；upstreamPY先k再softmax/topP-positionmask，实数和ties可不同。HWcapNPU_TOP_K_TOP_P两选择体实际都同PYfilter已local解决，别再要求runtimecapflag才能定此局部body。batchInvariant upstream HAS_TRITON时实际fusedfilterbody仍需namedkernelcontract，不开框架广泛研究。'
      '可条件merge两maskedFill OR为onewrite；orderednonNaNcutoffs可oneCompareP<max。Existing orderednonNaNCDF上count=min(upper_bound(CDF,sameactual1-p),V-1)，17probes/row，不改nativeCDF值；新scanproducer另证。TopKonly同roundedP orderstat可换fullsort但保ties，allkV/pNone条件noop，p1有FP32zeroP时非无条件noop。'
      'PrivateP DivArgmax融合保持原P/Q/division/comparator可省ratioplane；generic公开Pmutate要writeback，returnedprocessedlogits/logprobs要保。log(roundedP)非logsoftmax，复用originalP代posttruncP在F32race可翻winner，typedIEEEwitness不是nativeQseed/rounding attestation。'
      'Qglobalstream draws/currentwait/recordstream和generatorobject/dictorder保留。Ascendallseeded时不defaultdraw vsrejectionrecovery总defaultdraw不同，不能CSE两Qbank或借用Qzero/Inf sentinelguard。MixedbonusmaskLT1e-5 vsrejectEQ0，caller温度域未证别合并。保所有Qdraw只pruneunusedrandomID DivArgmax，otherlogit/score work不自动消失。'
      '3manifestboundsourcefiles11namedpins27if54arms只是导航。7650mask6375CDF729racecases+8witness、20costmodels/8RNGcases已PASS；source/math不是Torch/NPU。Mask16BV→8BV，race16BV→private8AV/writeback12BV仅twointerfacecostspans；softmax/sort/CDF/Qgen/returnedplanes/launch另外价，Sort32fullpass条件cut不是allalgorithmupper，不physicaltime/HBM。'
      'G7source29/40,currentpartial708/1159，M17OP12=>353records24semantic，35model7legacy0wholeclosed。115pairs24groups480pending和2073/285598review9/1352pending849PMU16phase3136ambiguity不变。F35otherprimitive/processor/readers/actualbackendflags仍open。'
      '下一步优先批不变上游Triton实际_topk_topp kernel的probability/tie/scan/threshold条件或仍影响融合的directprocessor/readereffects，不复读当前defaultfilter/RNG已解决事实当新优化，不盲增pair，不研究无关框架。仅savedprofile/pinpointfrozen/math/必要primary，无server/model/Torch/NPU/capture/service/upload。GoalACTIVE/TaskACTIVE/LoopEVALUATING；verify_sampling_filter_race_v91应PASS，不重跑finalizer不提前complete。')
    sourcepaths=[Path(r['path'])for r in read(O/names[0])['source_files']]
    oldgraphs=[Path(p)for k,p in read(O/names[-1])['graph_catalog'].items()if k!='G7']
    jobs=[dict(kind='review',status='pass',description='Bind3frozen sourcefiles11namedfunctions anddefaultAscendoriginalP cutoff/race/seed/streamcontracts;differentupstreamfilter andTritonbodyboundary explicit. G7partial29/40,current708/1159.',
      artifacts=[O/n for n in names[:1]+names[4:]]+sourcepaths+oldgraphs+[R/'fusion-scan.json']),
      dict(kind='design-check',status='pass',description='7650cutoff6375existingCDF729racefixedoperand cases,8numeric/domainwitnesses,20costmodels/8RNGcases;conditionalmask/search/private-racefusion,notnativeprecisionorspeedup.',
        artifacts=[O/n for n in names[1:4]]+[O/'sampling-filter-race-fusion-variants-v91.json',O/'sampling-filter-race-analysis-preflight-v91.json',O/'typed-capacity-primary-contract-v89.json'])]
    persist_refinement(version=91,previous=90,title='Probability cutoff fusion,existing CDF search and exponential-race effects',
      summary=dict(filename='sampling-filter-race-summary-v91.json',previous_turn_classification='PROGRESS:v90greedyfallbackdemand',
        cached_pinpoint_source_files=3,new_target_source_files=0,new_public_primary_documents=0,named_functions_pinned=11,local_source_if_sites=27,local_source_if_arms=54,
        exact_probability_mask_cases=7650,existing_CDF_search_cases=6375,fixed_operand_race_cases=729,numerical_domain_witnesses=8,cost_models=20,RNG_draw_cases=8,
        new_G7_node_rows=29,new_G7_edge_rows=40,finite_graph_node_rows=708,finite_graph_edge_rows=1159,
        all_F35_primitives_or_readers_mapped=False,refined_candidates=1,refined_modules=5,new_inventory_families=1,branch_records=353,semantic_modules=24,
        new_pairs=0,new_groups=0,relation_clauses=115,higher_order_groups=24,pending_central_pair_reconciliation=480,
        scoped_reviewed_regimes=2073,scoped_reviewed_executions=285598,pending_regime_reconciliation=9,pending_execution_reconciliation=1352,
        new_regimes_or_executions_promoted=0,actual_random_path_flags_or_native_speedup_proved=False,implementation=False,native_test=False,
        remote_job_status='Alllocal source/math/structuralchecks finished;no server/model/Torch/NPU/capture/servicework.'),
      changes=['ResolveoriginalPcutofffork,tiebehavior andlocalhardwarecapalias;upstreamsequentialfilter remainsdifferent.',
        'Proveconditionalmerge-mask/existingCDFupperbound/privateDivArgmax transformations withprecision/RNG/mixedpredicatefences.',
        'Verify7650/6375/729algebra cases,8witnesses and20costmodels;partialG7sourcegraph29/40.',
        'Update353branchrecords24semantic;no pair/group/nativepromotion orwholemoduleclosure.'],
      candidate_updates={'F35':update},module_updates=mods,semantic_branches={'M17':m17},remaining=remaining,action=action,jobs=jobs,report=report,
      read_first=names+['sampling-filter-race-fusion-variants-v91.json','sampling-filter-race-question-status-v91.json'],
      publication_rows={'F35':'| F35 | 目标采样行选择、greedy/概率过滤、race与拒绝输出重组 | 条件 | 默认Ascend以同一原P求k/p概率阈值并保同值项；可条件合并mask、复用CDF搜索、融合私有P/Q Div→argmax。上游filter、FP32归一化/下溢、RNG/流及mixed谓词不同，禁止直接替换；实际后端与收益待证。 |'},
      gate_updates=dict(overall_complete=False),scripts=[N/'analyze_sampling_filter_race_v91.py',N/'verify_sampling_filter_race_v91.py',N/'persist_profile_refinement_v90.py'],
      executor_prefix='sampling-filter-race',main_script=Path(__file__),source_read=False,
      additional_shared_writes={R/'fusion-scan.json':read(O/'fusion-scan-proposed-v91.json')})


if __name__=='__main__':main()
