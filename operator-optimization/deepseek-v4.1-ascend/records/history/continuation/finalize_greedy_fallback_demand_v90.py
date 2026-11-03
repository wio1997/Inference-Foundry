"""Persist greedy-fallback operator contracts and a version-bound G6 supplement."""
from copy import deepcopy
from pathlib import Path
import hashlib
from finalize_origin_coverage_v9 import N,O,R,read,write
from persist_profile_refinement_v90 import persist_refinement


def main():
    assert not(O/'REVIEW-CHECKPOINT-v90.json').exists()
    check=read(O/'greedy-fallback-analysis-preflight-v90.json');assert check['status']=='PASS'
    names=['greedy-fallback-source-contract-v90.json','greedy-fallback-demand-algebra-v90.json',
      'greedy-fallback-demand-cost-models-v90.json','greedy-fallback-pointwise-validation-v90.json',
      'operator-graph-G6-v90.json','operator-graph-node-coverage-v90.json','operator-graph-edge-coverage-v90.json',
      'operator-graph-candidate-bindings-v90.json','operator-graph-coverage-audit-v90.json']
    variants=[dict(id='F35-v90-N1-CAPACITY',state='CONDITIONAL_FULL_WRITE_REORGANIZATION',
      change='Actualone-draftspecialization caninlinecol0/1andthefullS-capacitysuffixinitialization.',
      requires='Callerfilledbank,allgreedy,allactualn1,validS>=1 andsameoutputdtype/bonusplan. Functionname isnotS==1;S>1suffixmust bewrittenorretainoldfill.',native_occurrence=False),
      dict(id='F35-v90-SYNTHETIC-ARGMAX-DEMAND',state='POINTWISE_CONDITIONAL_PRODUCER_CONSUMER_SPLIT',
        change='Compute existinguniform/rateacceptancefirst;onlythefirstrejectedrowpergreedyrequest needs exacttargetArgmax. Allaccepted requestsneed0targetArgmaxrows.',
        requires='SameU/rate/draftvalidityanddraworder,sameargmaxcomparator/rounding/rowplan,allworking-scoreprocessors andotherreaders retained. ArgmaxpruningdoesnotautomaticallypruneheadGEMM/bonus or scorework.'),
      dict(id='F35-v90-ORDINARY-PREFIX',state='REFINE_EXISTING_V65_OPPORTUNITY_NOT_NEW_DISCOVERY',
        change='Directper-request scan substitutesrepeat_interleave/B*Sscatter-min/gather preparation andrequestsargmaxonlythroughfirstmismatch.',
        requires='Exactn/cu/orderedrows,Sbounds,oldbankretention,maskedindexbeforegather;newserialization/parallelism/launchcosts remain.'),
      dict(id='F35-v90-COMPOSED-PURE-ID',state='CONDITIONAL_JOINT_BASELINE_REPRICING',
        change='Compose row-demand with guardedv65Index/Cast/clone/argmax fusion usingone22NVbaseline and2AVnew score-read reference.',
        requires='Allgreedy,allraw/processor/entropy/probability/logprob readersclosed inthisspan,same retainedBF16entry/indexepoch,widening/NaN/tie semantics. Noindependent20NV+4V(N-A)credits.'),
      dict(id='F35-v90-GENERIC-TAIL',state='UNCONDITIONAL_FILL_DELETION_RULED_OUT',
        change='Generichelpersretainarbitraryoldoutputtail/nongreedycells;replacingthemwithPLACEHOLDERwithoutcallerbankproofchangesresults.',
        requires='Preservecaller validation/B0error,alloriginalRNGdraws andlaternongreedyrandomphase writes/captureepochs.')]
    write(O/'greedy-fallback-fusion-variants-v90.json',dict(candidate='F35',variants=variants,new_maincandidate=False,
      implementation=False,native_test=False,actual_speedup=None,overall_complete=False))
    rel=deepcopy(read(O/'candidate-relation-clauses-v89.json'))
    for c in rel['clauses']:
        if c['id']in ['REL73-22','REL73-26']:
            c['greedy_fallback_refinement_v90']=dict(evidence=names,
              scope='Argmaxproducer rowdemand and output/RNGepochs;doesnotprove lazyhead-GEMM or allsource-row/readers private.',
              new_pair_or_all_variants_closed=False,critical_path_gain_us=None)
    rel.update(parent_relation_index='candidate-relation-clauses-v89.json',overall_complete=False);write(O/'candidate-relation-clauses-v90.json',rel)
    cov=deepcopy(read(O/'candidate-relation-coverage-v89.json'));cov.update(parent='candidate-relation-coverage-v89.json',
      parent_relation_index='candidate-relation-clauses-v90.json',greedy_fallback_refinement_v90=dict(new_pairs=0,new_groups=0,evidence=names),overall_complete=False)
    write(O/'candidate-relation-coverage-v90.json',cov)
    fps=deepcopy(read(O/'candidate-operator-footprints-v89.json'));registry={r['id']:r for r in read(O/'candidate-record-index-v89.json')['model_candidates']}
    for row in fps['candidates']:
        q=R/'candidates'/(row['id']+'.json');actual=hashlib.sha256(q.read_bytes()).hexdigest();assert actual==registry[row['id']]['master_sha256'];row['captured_master_sha256']=actual
        if row['id']=='F35':
            row['operator_atom_union']+=['sampler.actual_n1_not_output_S1','sampler.synthetic_target_argmax_demand','sampler.greedy_output_epoch']
            row['G6_body_source_binding_v90']=dict(graph='operator-graph-G6-v90.json',node_rows=47,edge_rows=69,all_F35_primitives_or_variants_mapped=False)
    fps.update(parent='candidate-operator-footprints-v89.json',overall_complete=False);write(O/'candidate-operator-footprints-v90.json',fps)
    nav=deepcopy(read(O/'candidate-pair-navigation-v89.json'));byid={r['id']:r for r in fps['candidates']}
    for pair in nav['pairs']:
        pair['shared_operator_atom_union']=sorted(set(byid[pair['a']]['operator_atom_union'])&set(byid[pair['b']]['operator_atom_union']))
    nav.update(parent='candidate-pair-navigation-v89.json',greedy_fallback_refinement_v90=dict(new_pairs=0,new_groups=0),overall_complete=False)
    write(O/'candidate-pair-navigation-v90.json',nav)
    op=deepcopy(read(O/'operator-analysis-coverage-ledger-v89.json'));op['greedy_fallback_refinement_v90']=dict(
      evidence=names,previously_unexpanded_direct_greedy_bodies=2,new_native_executions_or_regimes=0,
      source_guard_question_answered='all actualn1 doesnot implyS1;S>1suffix remainsoldbank',
      fixed_input_general_cases=95920,fixed_input_special_cases=160,all_native_backend_or_flag_fields_proved=False)
    write(O/'operator-analysis-coverage-ledger-v90.json',op)
    branchledger=deepcopy(read(O/'M17-probability-branch-ledger-v49.json'))
    branchledger['rows'].append(dict(branch='greedy_PyTorch_specialized_and_general',
      state='TWO_DIRECT_GREEDY_BODIES_CONDITIONALLY_REVIEWED',evidence='greedy-fallback-source-contract-v90.json;actualcounts1vsScapacity,sourceif6/arms12,fixedinputequivalence andargmaxdemand'))
    branchledger.update(parent='M17-probability-branch-ledger-v49.json',six_direct_acceptance_recovery_helpers_reviewed_within_named_scope=True,
      four_random_recovery_body_review_first_version=49,two_greedy_body_refinement_version=90,branch_universe_exhaustive=False,whole_module_closed=False,overall_complete=False)
    write(O/'M17-probability-branch-ledger-v90.json',branchledger)
    gap=('Sourcehelpercontracts areconditional onvalidn/cu/S,row/index/bonusplan,same input/output storage epochs and exact argmax values/comparator. '
      'Actualfallback/synthetic/allgreedyflags and any changed-Nsourcecase remain unobserved;current72greedyTritonTASKs donotattestPyTorch fallback. '
      'OriginalRNGdraws/processor/raw/score/probability/logprob/bonusreaders,errorsandcapturelifetime remain. Nativefusedprecision/parallelism/launch/crosscard costs andphysicalargmax capacity/speedup areunproved.')
    matrix=read(O/'module-phase-branch-matrix-v89.json');m17=deepcopy(next(m['branches']for m in matrix['modules']if m['id']=='M17'))
    for row in m17:
        if row['id']in ['M17-OP-04','M17-OP-05','M17-OP-06','M17-OP-08','M17-OP-09']:
            row['evidence']+=names;row['greedy_body_refinement_v90']=dict(
              actual_n1_not_S1=True,synthetic_zero_or_one_target_argmax_rows_per_greedy_request=True,
              ordinary_prefix_opportunity_preexisting_v65=True,native_backend_flags_proved=False);row['gap']=gap
    m17.append(dict(id='M17-OP-11',branch='Two PyTorchgreedy fallback bodies:actualn1 specialization retainsS-capacitytail;generalraggedscatter/min canbe directscan;syntheticacceptance needs targetArgmax onlyatfirstreject.',
      state='PINNED_BODY_AND_FIXED_INPUT_DEMAND_EQUIVALENCE_REVIEWED;WHOLE_MODULE_OPEN',evidence=names,
      gap=gap,scope='Directnamedgreedyhelpers andimmediatecaller/output/RNGedges;not allsamplingprocessors/random backends ornativefusionproof.'))
    mids=['M16','M17','M21','M22','M24'];mods={mid:dict(
      finding='Pinned two greedy fallback bodies bind actual draft-count versus output-capacity guard,conditional argmax demand and full-output/metadata reorganization. G6 primitive source graph supplements prior partial source catalog;no new native attribution.',
      evidence=names+['greedy-fallback-fusion-variants-v90.json'],candidates=['F35'],gap=gap,
      profile_scope='Operator row-demand/fusion/store and state-effect conditions;no independent framework research.')for mid in mids}
    update=dict(review_v90=dict(evidence=names+['greedy-fallback-fusion-variants-v90.json'],
      direct_greedy_body_review=True,actual_count1_is_not_capacity1=True,synthetic_selected_argmax_projection=True,
      source_graph_supplement='G6:47nodes69edges;notallF35variants',fixed_input_general_cases=95920,
      pure_ID_joint_baseline='22NV→2AV conditional;noadded20NVand4V(N-A)credits',implementation=False,native_test=False),
      conclusion_changing_gap=gap,
      next_profile_or_contract_action='Prioritize remaining all-reader/processor primitive mappings and effect-sensitive branchconditions. Newfallbackcasesdo notchangeobserved72Tritongreedyflags. PreserveS-capacitytails,RNGandoldbankepoch;priceargmax demand separatelyfromheadGEMM/bonus. No actualspeedup orMFUclaim.')
    write(O/'PROFILE-CODE-QUESTION-GATE-v90.json',dict(status='PROFILE_LED_NAMED_GREEDY_FALLBACK_OPERATOR_QUESTIONS',
      questions=['Doesactualn1guardimplyoutputcapacityS1? No.',
        'Canmetadata scatter/min and conditional argmax demand be fused/reorganized without changing outputs? Pointwiseconditionals verified.',
        'DoesexistinggreedyTritonprofileprovePyTorchfallback orsynthetic flags? No.'],
      pinpoint_cached_source_files_read=1,direct_function_bodies=2,local_source_if_sites=6,local_source_if_arms=12,
      no_new_target_source_files=True,new_public_document_read=False,new_remote_source_read=False,
      no_framework_wide_research=True,no_Torch_model_NPU_capture_or_server_work=True,overall_complete=False))
    write(O/'greedy-fallback-question-status-v90.json',dict(
      newly_resolved=['Actualcount1specialization doesnot requireS1;tailinitializationmustremain.',
        'Two exactgreedyhelperbodies and6sourceconditions nowhaveexplicitoperatorcontracts.',
        'Syntheticgreedy targetArgmax demand is0or1perrequest,withsameprovidedU/rate/IDinputs.',
        'JointtargetIndex/Cast/clone/argmax readbudget usesone22NVbaseline;independentcredits canovercount.'],
      prior_facts_reused=['Ordinarylazyargmaxprefixv65','Randomfullresiduallazyrecoveryv33','Fourrandom/recoveryfallbackbodiesv49','A3UBcapacityv80/v89'],
      unresolved=[gap,'AllF35processor/logprob/random primitives andallcallerreader/alias protocols;G6ispartial.'],
      changed_N_scenarios_not_claimed_native=True,new_native_regime_promotions=0,whole_modules_closed=0,overall_complete=False))
    remaining=list(read(O/'module-coverage-gaps-v89.json')['remaining'])
    remaining.append('v90:2directgreedyTorchfallbackbodies newlocal review6ifs12arms;actualn1guardnoS1constraint soS5suffixstillneedsfill. Synthetic targetArgmax0/allacceptor1/firstreject pergreedyrequest;ordinarylazyargmaxpreexistingv65 refinednotnew.95920general160special fixedinputcasesnotnativeTorch/NPU.18declaredcostscenariosfrom3B/Tcapacity/V geometries;changedNmodelsunobserved notcurrentcapturedshapes. PureIDcompose22NV→2AVonebaseline not20NV+4V(N-A). FullFLOAT517120/BF16258560row>UB196608 excludesoneAIVwhole-row planonly. G6source47node69edge supplement,currentpartialG1/G5/G6coverage679/1119structuralrowsnotuniquenativeopsorallbranches.1M17family=>352records24semantic. No newpairs/groups/promotions115/24/480,2073/285598review9/1352pending849/16/3136 unchanged0wholeclosed.')
    report='''本轮围绕尚未展开的贪心回退算子，定点核对两个已有冻结函数体及其直接调用条件。当前profile仍只有72次greedy Triton正向证据，没有把本轮PyTorch回退、synthetic或其他运行标志认定为已执行。

一个新的融合边界是：`rejection_greedy_sample_spec_len_1_pytorch`的调用条件检查每个请求实际draft数均为1以及all_greedy，没有检查输出容量`max_spec_len==1`。函数本身也只断言B==N。容量S=5、实际n=1的合法分支中，它仅写第0/1列，后四列依赖原初始化。因此不能凭函数名删除Fill。固定输入例子正确输出为`[7,9,-1,-1,-1,-1]`；省去初始化且仍只写两列，会留下`[7,9,99,99,99,99]`。这个分支可能具有不同于当前记录T=10/20的输入N，本轮没有把它冒充已观测shape。

通用回退先构造request/token映射，再生成位置、mismatch、sentinel三张B×S逻辑平面，经min找首个失败，最后按greedy和copy长度掩码索引。可以用直接按请求扫描替代这一准备过程，并融合完整输出写入，但必须先掩码后读取、保持bonus列、旧尾部和非greedy行。

普通贪心按首个不匹配位置查询argmax的方向已在v65提出，本轮只补函数体级证明。新的细化是synthetic分支：接受判定只看原uniform、rate和draft有效性；接受位置输出draft，首次失败位置才需要target argmax。每个greedy请求至多查询一个target argmax行，全部接受时查询零行。例子draft=`[3,4]`、target argmax=`[8,9]`、uniform=`[0.25,0.75]`、rate=0.5，应输出`[3,9,-1]`，只查询第二行；不能把接受位置替换成target的8。

这是greedy target-argmax的需求投影，与v33已经分析过的随机拒绝后全词表残差采样不同。后者仍须其完整概率/RNG合同。这里保留全部原uniform生成、seeded覆盖和后续mixed随机路径的独立draw，不能把前后两份uniform按“相似计算”合并。logits processors、raw/logprob/entropy/概率读者以及bonus生成也没有自动获得删减许可。

成本按三个既有B/Tcapacity/V几何建立18个声明场景，区分记录容量与改变N的未观测分支。仅删除不再需要的FLOAT argmax查询时，分值读取参考为4NV→4AV。若还满足v65的纯ID、原BF16/index epoch、精度和全部读者条件，Index/Cast/clone/argmax联合跨度应按同一个22NV旧基线计价，新参考为2AV；不能再把20NV旧融合额度与4V(N−A)额外相加。N=10、V=129280、A=0时，错误相加会超过整个旧跨度的参考成本。

Fill的计价同样按完整新图重算。通用回退原来是整银行Fill加局部更新；完整写入新kernel仍须写全部输出格，收益不是机械减去所有Fill字节。实际n1专用体无论接受与否都会写第1列，同时保留S>1尾部，也按这个实现单独计价。

理论约束保留算法范围：每个仍需查询的、无额外索引/证书的任意分值行，在最坏情况下仍需检查完整V；不能把高层argmax直接套用Vector Max的速率。V=129280时，整FLOAT行517120B，整BF16行258560B，均大于单AIV 196608B UB。这只排除整行同时驻留方案，分块、流式和多AIV方案仍可分析。普通逐请求扫描还可能损失跨行并行度，不能把读取量下降当作耗时同比下降。

固定输入的独立“展平映射/scatter/min”和“逐请求需求扫描”模型通过95920组通用用例，实际n1专用体与通用体通过160组一致性用例，涵盖S0..5、ragged、负placeholder、mixed掩码、synthetic和任意旧尾部。它们是源语义/数学验证，没有导入Torch或运行NPU；原生归约、NaN/tie、alias、capture、启动与并行成本仍需证明。

新增G6将直接调用、两条函数体和输出/RNG边界展开为47行节点、69行边；当前G1/G5/G6部分图合计679/1119行，已更新版本绑定的覆盖入口。它们是结构行数，不是唯一算子或全分支覆盖率，F35其他primitive和读者仍未全部映射。

累计352条分支记录、24个语义模块；115条条件关系、24组联合合同，中央仍有480对待汇入或复核。2073个regime/285598次执行已作条件分析，9个regime/1352次仍待补；849条PMU缺失、16张阶段未绑定图、3136条模块归属歧义不变。整体任务保持进行，没有整模块关闭。
'''
    action=('继续原Goal，v90PROGRESS，v89亦PROGRESS。先读greedy-fallback-summary/PROFILE-FINDINGS-v90、source-contract/demand-algebra/demand-cost-models/pointwise-validation/fusion-variants/question-status及operatorgraphG6/node/edge/candidatebindings/coverageauditv90。'
      '本轮真实新证据为2directgreedyTorchfallback bodies6ifs12arms，caller min/maxactualcounts1+allgreedy没有max_spec_len1；special只写col0/1，S5suffix仍靠fill。N1/S5是合法unobservedshape分支，N=B与当前Tcap10/20不一致时不能冒充nativecase。'
      '普通lazyargmaxprefix早在v65是旧方向，本轮bodyrefine；randomlazyfullresidualv33/fourfallbackbodyv49也不算新。新syntheticgreedyaccept只依赖sameU/rate/draft>=0，acceptedemitDraft，firstfailed才targetArgmax；每greedyrequest0/allaccept或1/rejectquery。保全部RNGcalls/序列，mixed lateruniform/Q是独立draw不是CSE。保processors/score/raw/entropy/prob/logprob/bonus和validation。'
      '95920general+160special fixedinputs纯Python独立源语义/需求scan对照PASS，不是Torch/NPU/nativeprecisiontest。18costmodels/3recordedB/Tcap/V几何；changedNcase标unobserved。仅Argmax4NV→4AV；pureID与v65compose用one22NVbaseline/new2AV，别加20NV和4V(N-A)重计。Fullwriter仍写B(S+1)，按原Fill+helperstores vsnewfullwrites重价。'
      'FLOATfullrow517120和BF16258560>singleAIVUB196608只排除整行驻留，chunk/stream/multiAIV仍可。任意score rowexactargmaxworstcase需全V值unlesscertificate，不把Max64当wholeArgmaxcap。普通prefix可能串行化，不能推runtimegain。'
      'G6source47nodes69edges supplement，canonicalfusionScan更新G1v88/G5v88/G6v90，currentpartialledger679nodes1119edges不是uniqueops或allbranches。F35wholeprimitive/mapping/allreaders remainopen；35candidates仍conditional。新增M17OP11 =>352records24semantic；115pairs24groups480centralpending，2073/285598review9/1352pending849PMU16phase3136ambiguity0wholeclosed不变。'
      '后续继续尚未闭合且影响算子融合/理论上限的primitive/allreader/conditionalbranch问题，用保存profile入口，只有定案需要才查明确source；不盲增pairs，不复读旧方向当新优化，不研究无关框架。仅offline/pinnedsource/math/primary必要；no server/model/Torch/NPU/capture/service/upload。GoalACTIVE/TaskACTIVE/LoopEVALUATING，不重跑finalizer；verify_greedy_fallback_demand_v90应PASS。')
    graphpaths=[O/'operator-graph-G1-v88.json',O/'operator-graph-G5-v88.json']
    jobs=[dict(kind='review',status='pass',description='Pin2directgreedyfallbackbodies,6ifs12arms;actualcount1 guarddoesnot implyS1;explicitG6source graph supplement47/69 andversionboundpartialcoverage679/1119.',
      artifacts=[O/n for n in names[:1]+names[4:]]+graphpaths+[Path(read(O/names[0])['path']),R/'fusion-scan.json']),
      dict(kind='design-check',status='pass',description='95920general/160special fixedinputproofs,18costcases,synthetic0/1targetargmax-demand andonejoint22NVbaseline;noTorch/NPU/nativeeffectorspeedupproof.',
        artifacts=[O/n for n in names[1:4]]+[O/'greedy-fallback-fusion-variants-v90.json',O/'greedy-fallback-analysis-preflight-v90.json',O/'A3-local-capacity-and-ceiling-scope-v80.json'])]
    persist_refinement(version=90,previous=89,title='Greedy fallback output capacity and conditional argmax row demand',
      summary=dict(filename='greedy-fallback-summary-v90.json',previous_turn_classification='PROGRESS:v89typedcapacityevidence',
        cached_pinpoint_source_files=1,new_target_source_files=0,new_public_primary_documents=0,newly_reviewed_named_greedy_bodies=2,local_source_if_sites=6,local_source_if_arms=12,
        fixed_input_general_cases=95920,fixed_input_special_cases=160,cost_models=18,profile_capacity_geometries=3,
        new_G6_node_rows=47,new_G6_edge_rows=69,finite_graph_node_rows=679,finite_graph_edge_rows=1119,
        all_F35_primitives_or_readers_mapped=False,refined_candidates=1,refined_modules=5,new_inventory_families=1,branch_records=352,semantic_modules=24,
        new_pairs=0,new_groups=0,relation_clauses=115,higher_order_groups=24,pending_central_pair_reconciliation=480,
        scoped_reviewed_regimes=2073,scoped_reviewed_executions=285598,pending_regime_reconciliation=9,pending_execution_reconciliation=1352,
        new_regimes_or_executions_promoted=0,actual_fallback_or_synthetic_flags_proved=False,native_precision_or_physical_speedup_proved=False,
        implementation=False,native_test=False,remote_job_status='Alllocal source/math/structuralchecks complete;no server/model/Torch/NPU/capture/servicework.'),
      changes=['Review2greedyfallbackbodies;proveactualcount1doesnotimplyS1 andretainS-capacitytails.',
        'Conditional syntheticgreedy targetArgmax requires0/1row perrequest;ordinaryprefixdirectionv65 onlyrefined.',
        'Verify95920general/160special fixedinputs and18symboliccostmodels;composeone22NVbaseline,notindependentcredits.',
        'VersionbindG6sourcegraph47/69,currentpartial679/1119;352branchrecords24semantic withnativeuncertaintiesunchanged.'],
      candidate_updates={'F35':update},module_updates=mods,semantic_branches={'M17':m17},remaining=remaining,action=action,jobs=jobs,report=report,
      read_first=names+['greedy-fallback-fusion-variants-v90.json','greedy-fallback-question-status-v90.json'],
      publication_rows={'F35':'| F35 | 目标采样行选择/转换/argmax与拒绝输出重组 | 条件 | 两条greedy回退体已核对：实际n1不等于容量S1，尾部初始化保留。Synthetic接受后仅首拒位置需要target argmax，全接受时该请求为零行。保RNG/读者/精度，联合22NV旧基线重价；95,920组固定输入检查不等于原生加速证明。 |'},
      gate_updates=dict(overall_complete=False),scripts=[N/'analyze_greedy_fallback_demand_v90.py',N/'verify_greedy_fallback_demand_v90.py',N/'persist_profile_refinement_v90.py'],
      executor_prefix='greedy-fallback',main_script=Path(__file__),source_read=False,
      additional_shared_writes={R/'fusion-scan.json':read(O/'fusion-scan-proposed-v90.json')})


if __name__=='__main__':main()
