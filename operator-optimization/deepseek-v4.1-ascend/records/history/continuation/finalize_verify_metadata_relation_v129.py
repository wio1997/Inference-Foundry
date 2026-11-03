"""Persist one previously-unreconciled operator-pair relation; all-runtime open."""
from copy import deepcopy
from pathlib import Path
from finalize_origin_coverage_v9 import N,O,read,write
from persist_profile_refinement_v90 import persist_refinement


def main():
    assert not (O/'REVIEW-CHECKPOINT-v129.json').exists()
    pre=read(O/'verify-metadata-preflight-v129.json');assert pre['status']=='PASS'and pre['checks']==6465
    names=['verify-metadata-inputs-v129.json','verify-metadata-profile-scope-v129.json','verify-metadata-source-contract-v129.json',
        'verify-metadata-value-and-epoch-witnesses-v129.json','verify-metadata-layout-models-v129.json',
        'verify-metadata-tie-reference-validation-v129.json','verify-metadata-generation-graph-v129.json','verify-metadata-joint-contract-v129.json']
    detail=dict(evidence=names,profile_databases=32,recorded_profile_APIs_selected=183813,
        saved_seed_publication_TASKs=104,saved_next_ID_publication_TASKs=520,
        next_ID_count_surface_TASKs=1232,next_ID_count_typed_TASKs=1144,next_ID_count_without_COMPUTE_definition=88,
        dynamic_scheduler_or_confidence_surface_API_hits=0,actual_dynamic_flag_attested=False,
        prefix_layout_cases=2834,TopK_candidate_subsets=65028,valid_TopK_subsets=14182,
        selected_position_not_prefix_witness_cases=8938,new_scoped_pair_relations=1,
        new_pair_clauses=1,new_groups=0,new_main_candidates=0,new_native_promotions=0,
        new_device_work=False,whole_requirement_closed=False)
    contract=(
        'Keep current target accepted-state next-ID/count,current draft input/metadata andlater proposalconfidence/verifykeep lengths in distinct value/generation namespaces. '
        'GlobalTopK positions feed a request histogram;published tokenlists are capped prefixes,not the selected positions. '
        'Compose finalkeep/actualtrimmedlength with nexttargetmetadata onlyunder fullscheduledrequest/row/CP/group/attrs/capture/bankreader identity;shape,totalQ andbudget alone are insufficient. '
        'Current noncausaldraft rows are not retroactively dead;acceptedcount comes from futuretargetsample/rejection,notconfidence. '
        'Preserve existingperiodicbudget/numerics/ties/RNG/hostint/zip/zeroquery/frontier andoldreader effects. NewABI logical content is notbaselineHBM/time savings.')
    rel=deepcopy(read(O/'candidate-relation-clauses-v128.json'))
    assert not any({r['a'],r['b']}=={'F30','F31'}for r in rel['clauses'])
    rel['clauses'].append(dict(id='REL129-01',a='F30',b='F31',
        relation='CROSS_ITERATION_PREFIX_DEMAND_AND_ACCEPTED_STATE_NAMESPACE',operator_claim=contract,evidence=names,
        applicable_domain='Selected padded DSpark merged first-pass/early-return,optionaldynamic policy andeligible puredecode next scheduled prefix;other branches retain explicit gates.',
        conditional_compatibility=True,all_runtime_gates_proved=False,speedup_additive=False,critical_path_gain_us=None,
        verify_metadata_composition_v129=deepcopy(detail)))
    for g in rel['higher_order_groups']:
        if g['id']in ['HG73-TAIL','HG85-VERIFY-CONTEXT-DRAFT']:g['verify_metadata_composition_v129']=dict(**detail,contract=contract,whole_group_closed=False)
    rel.update(parent_relation_index='candidate-relation-clauses-v128.json',new_pair_clauses=1,new_higher_order_groups=0,
        clause_count=136,selected_pair_count=136,verify_metadata_composition_v129=detail,overall_complete=False)
    write(O/'candidate-relation-clauses-v129.json',rel)
    cov=deepcopy(read(O/'candidate-relation-coverage-v128.json'))
    row=next(r for r in cov['pairs']if {r['a'],r['b']}=={'F30','F31'})
    row.update(scope_state='SCOPED_OPERATOR_RELATION_REVIEWED_CONDITIONAL',clauses=['REL129-01'],whole_pair_all_variants_closed=False,
        new_critical_path_gain_us=None,verify_metadata_composition_v129=deepcopy(detail))
    cov.update(parent='candidate-relation-coverage-v128.json',parent_relation_index='candidate-relation-clauses-v129.json',
        scoped_conditional_pairs=136,remaining_pair_relations=459,verify_metadata_composition_v129=detail,
        scope='136 scoped conditionalpairs/26groups/35modelcandidates;459centralpairs remain. F30-F31 value/generation/prefix/metadata composition reviewed;allvariants/runtime legality remain open.',overall_complete=False)
    write(O/'candidate-relation-coverage-v129.json',cov)
    nav=deepcopy(read(O/'candidate-pair-navigation-v128.json'));row=next(r for r in nav['pairs']if {r['a'],r['b']}=={'F30','F31'})
    row.update(conditional_clause_ids=['REL129-01'],semantic_pair_reconciliation_done_in_stated_scope=True,
        whole_pair_all_variants_closed=False,verify_metadata_composition_v129=deepcopy(detail),evidence=names)
    nav.update(parent='candidate-pair-navigation-v128.json',parent_footprints='candidate-operator-footprints-v129.json',
        parent_relation_index='candidate-relation-clauses-v129.json',scoped_conditional_pairs=136,
        remaining_central_pair_reconciliation=459,verify_metadata_composition_v129=detail)
    write(O/'candidate-pair-navigation-v129.json',nav)
    fp=deepcopy(read(O/'candidate-operator-footprints-v128.json'))
    for row in fp['candidates']:
        if row['id']in ['F30','F31']:row['verify_metadata_composition_v129']=dict(evidence=names,
            no_new_native_operator_atoms=True,local_generation_graph='verify-metadata-generation-graph-v129.json',all_runtime_or_native_parameters_bound=False)
    fp['parent']='candidate-operator-footprints-v128.json';write(O/'candidate-operator-footprints-v129.json',fp)
    op=deepcopy(read(O/'operator-analysis-coverage-ledger-v128.json'));op['verify_metadata_composition_v129']=detail
    write(O/'operator-analysis-coverage-ledger-v129.json',op)
    obligations=deepcopy(read(O/'coverage-current-obligations-v128.json'))
    for row in obligations['obligations']:
        if row['id']in ['G00','G02','G05','G06','G07','G08','G09','G11','G13']:row['verify_metadata_composition_v129']=deepcopy(detail)
        if row['id']=='G07':row['remaining']=(
            'Reconcile allreaders/newwork/packing/recompute/publication/resource competition for eachchosenrepresentation. '
            'F30-F31 nowhas REL129-01:currentacceptedstate,currentdraftmetadata,laterproposalkeep andnexttargetverificationmetadata aredistinct. '
            'GlobalTopKposition selection isnotpublishedprefix;usecompletedrequestcounts plusactualwidth/cap/rowgeneration. '
            'OnlynewABI logical content4B+4(B+1)+4Q ismodeled,no currentdevicek-to-QSLscan baselineorindependentgain.136conditionalpairs/26groups,459centralpairs andallvariants remainopen.')
        if row['id']=='G13':row['remaining']=(
            'v129 completes one genuinelypending F30-F31 conditional composition review;do notrepeatnext-IDpredicate/confidencepolicyor genericadapter work. '
            'Newall32rawcallchainquery selects183813APIs,104seed/520IDpublication TASKs and1232next-IDsurface TASKs(1144typed+88withoutCOMPUTE). No recordedconfidence/scheduler/retrievalsource hits,notproofdisabled. '
            'Keep rawTopKpositions,count-derivedpublishedprefix,actualtargetacceptedcount andcurrent/futuremetadata generationsdistinct. '
            'Actualdynamic/B/D/listwidth/consumer cap,requestrowmap,periodicstate/numeric/ties/RNG,opaqueplanpurity/bankreaders/capture/update remain open. '
            'Next select a remainingrelation withspecificoperatorproducer/reader orrepresentation evidence;F29-F32 contextualKVprojectionversusSPinputgather/earlyprojection iscentrallypending andshouldfirstbecheckedagainstexistingprofile/ownershipgraphs. '
            'All9/1352nativefields,physicalcapacity/MFU/timebounds/fullbranchgraphscope remain. No newdevice/model/capture/replay authorization orpendingapproval;v120Pad grantconsumed.')
    obligations.update(parent='coverage-current-obligations-v128.json',new_native_or_whole_closures=0,overall_complete=False)
    write(O/'coverage-current-obligations-v129.json',obligations)
    audit=deepcopy(read(O/'whole-scope-requirement-audit-v128.json'))
    audit.update(previous_goal_turn='PROGRESS:v128 native disk scalar plan/8081words/147records;postcheck PASS8619',input_checkpoint=128,followup_checkpoint=129,verify_metadata_composition_v129=detail)
    for row in audit['requirements']:
        if row['id']=='CROSS_OPERATOR_LAYER_CARD_ITERATION':row.update(conditional_pair_relations=136,central_pairs_not_reconciled=459,higher_order_groups=26)
        if row['id']in ['ALL_MODULE_CONDITIONAL_BRANCHES','CROSS_OPERATOR_LAYER_CARD_ITERATION','THEORY_AND_MEASURED_COMPARISON','CRITICAL_EVIDENCE_AND_FINAL_CLOSURE']:row.setdefault('evidence',[]).extend(names)
    write(O/'whole-scope-requirement-audit-v129.json',audit)
    write(O/'PROFILE-CODE-QUESTION-GATE-v129.json',dict(status='ALL32_CACHED_PROFILE_OPERATOR_ANCESTRY_THEN_TARGETED_KEEP_COUNT_PREFIX_AND_METADATA_DATAFLOW',
        questions=read(O/names[0])['questions'],source_scope='Only selected frozenproposer/scheduler/worker valueproducer-consumer segments plus existingmetadatafrontier contract. No frameworksurvey ornewSDK work.',
        profile_zero_hits_not_runtime_false_flags=True,no_model_operator_implementation=True,
        no_remote_Torch_CANN_NPU_model_or_replay=True,Pad_authorization='GRANTED_AND_CONSUMED',overall_complete=False))
    write(O/'verify-metadata-validation-history-v129.json',dict(events=[
        'Predecessorv128 revalidated PASS8619/manifest69;previous turn producedtargetednative disk scalar evidence,notwait/status-only.',
        'Readall32SHA-pinnedcachedDB Torchcallchains inro/immutable mode. OriginalSQLitefiles/stats unchanged.183813selectedAPIs;exactnativeCIDarrayjoin preserves104seed/520IDpublication TASKs and1232next-IDsurfaceTASKs.',
        'Next-IDsurface includes1144typedTASKs and88withoutCOMPUTEdefinition;do notinventtheir oporphysicalcopybytes. Confidence/scheduler/retrieval/head source selectorshave0recordedAPIs,notallflagsdisabled proof.',
        'Onlyafterprofileinsufficiency,reuse3hash-pinnedsourcefiles andexistingv36/v37contracts tobindcurrentacceptedstate→draftprepare→mergeddecoder→Markov→optionalconfidence→keeptrim→nextverification namespaces.',
        'GlobalTopK equal-position witness:one rowfour survival1values,chosenextrapos3 yieldsk2andpublishedtokenprefix0,1,notselected0,3. Count-firstprefixsemantics matters evenwithfiniteprobabilities.',
        'SameB2D3budget2 canyieldkeep3,1/querylength4,2 versusuniform2,2/querylength3,3;total6same,butQSL/owner differ. Requestreorder changesQSL too. Actualreturnedwidth/cap/zip andepoch constraints preserved.',
        '2834newABI prefixlayout references,65028candidateTopKsubsets/14182valid/8938nonprefixselectedsets checked. Thesearepuremath/reference domains,notnativeTopKtie/output/precision tests.',
        'Keep kINT32 observerbank,query_start_loc andowner content4B+4(B+1)+4Q separatefromcurrentbaselineallocations/HBM. No existingdevicek-to-QSLscan removal orphysicalspeedup credited.',
        '6465preflightchecksPASS:allselectedAPIrole summaries,32DBSHA andfirst/lastpositive rawAPI samples,sourcepins/layout/counterexamples/localgenerationDAG andpredecessormanifest verified. Alllocaljobs terminal,no remote/devicework.',
        'OnecentralpairF30-F31 reviewed inconditional scope=>136pairs/136clauses/26groups/459pending. No newmaincandidate/nativeatom/globalgraphnode/semanticfamily orwholemodule closure.'],overall_complete=False))
    gap=(
        'Need actualdynamicflag,B/D/returnedwidth/cap/requestidentity,periodicbudgetstate,tie/precision/RNG andfullopaqueplan input/storage/readergeneration. '
        'Laterkeep cannotprune currentnoncausaldraftmetadata orsubstituteacceptedcount. RawTopKpositions cannotreplacethecount-derivedpublishedprefix. '
        'Newkeep-to-nextmetadata composition needsnewcompatibleABI andallstablebanks/frontiers/oldreaderretirement;currentasync work isbaseline. '
        'No actualreadtraffic/cycle/physicalcapacity/criticalpathgain orwholepair/allbranchclosure established.')
    remaining=[r['id']+': '+r['title']+'. '+r['remaining']for r in obligations['obligations']]
    remaining.append('v129F30-F31 conditionalrelation reconciled:136pairs/26groups/459centralpairs remain.2834layouts and65028TopKsubsets notruntimeproof.358/26semantic,854/1346/48partialgraph,9/1352nativepending,849PMU/16phase/3136ambiguity,0whole unchanged.')
    action=(
        'Continue original Goal;v129 PROGRESS on F30-F31 crossiteration keep/prefix/acceptedstate/metadata relation,not hardware/nativeclosure. '
        'Read verify-metadata summary/profile-scope/source-contract/value-epoch-witnesses/layout-models/tie-reference/generation-graph/joint-contract/current-obligations-v129;verify_verify_metadata_relation_v129 mustPASS. '
        'v128postPASS8619. All32cachedDB exactrawcallchain/CIDarrayjoins select183813APIs;104seed/520IDpublication TASKs,1232next-IDsurface=1144typed+88withoutCOMPUTE. No scheduler/confidence/keepretrievalsourceAPIhits;notfalseflagproof. No remote/model/Torch/CANN/NPU orreplay. Queryexec41129terminal0. '
        'Onlytargetedpinnedproposer/utils/workersegments reusedafterprofile. CurrenttargetY→literalnext-ID/count→inputprepare/metadata→draftdecoder→Markov→optionalconfidence/globalbudgetTopK→keepbank→hostcappedprefix→nextscheduledtargetmetadata→futuretargetacceptedcount. Currentdraftmetadata/fullnoncausalrows cannotretroactivelydie. '
        'RawglobalTopKpositions arenotpublishedprefix afterhistogram:tieone-rowD4 chooseextraactualpos3⇒keep2⇒publishpositions0,1. 65028mathsubsets/14182valid/8938nonprefixcases,notnativeTopKtiepolicy. '
        'Keep3,1 vs2,2 totalqueries6butdifferentQSL/owner;requestreorder alsochangesmetadata. Actualreturnedwidth/cap/zip/currentrequestgeneration mandatory;zero-Dreturnsbeforedynamicupdate,nonparallelcontinuation remainsseparate. '
        'ConfidencekINT32 notcurrent/futureacceptedcountINT64. Samek permits0..k+1validcounts withdiscard/rejection;literalholes[7,-1,9] givesnext-1,count2;negatives<-1stillvalidbelowV. Preserveoriginalsummary/backup/publication observers. '
        '2834newABI B0..4,D0..5 layoutcases:4B+4(B+1)+4Q logicalk/QSL/optionalownermapcontent;notbaselineHBM/latency. Existingk/int(k)/hosttrim andopaqueSMLAQLI/frontiers remainunlessexplicitcompatibleinterface replacesallobservers. Range-specific readiness only,no inventedglobalbarrier. '
        'REL129-01 addedF30-F31;HG73TAIL/HG85VERIFYCONTEXTDRAFT refined,nonewgroup/candidate/nativeatom.136conditionalpairs/136clauses/26groups/459centralpending;allvariantsopen. No globalcatalogchange854/1346/48 orsemanticgrowth358/26. '
        'Next inspect existingprofile andownershipgraphs for remainingF29-F32 relation(contextKVprojection versusSPgather/earlyprojection),thenonlytargetedoperatorcode ifneeded;do notfillblankpairsbyemptyatom/noedge assumptions orrepeatalreadyansweredconfidence/next-ID work. '
        '2073/285598conditional9/1352pending849PMU16phase3136ambiguity0whole/noallmodulephysicalMFU/lowest unchanged. ActualoriginalMemSet/BOOL/lifetimegates stillopen;no freshdevice/model/capture/replayauthorization;Padv120consumed,no pendingapproval. '
        'Do noteditsealedv129/rerunfinalizer. Versionedpostcheckauthority;CURRENT-POSTCHECKoverlayvalidonlymatchingRECOVERY. TaskACTIVE Loop004EVALUATING GoalACTIVE.')
    report='''本轮完成一条此前未进入中央关系台账的F30–F31条件分析：DSpark置信度/验证预算与metadata构造、发布。它不是新增主候选，也没有宣称动态裁剪在原负载启用。先只读查询32份固定SHA的缓存profile，保留183813条相关Torch API记录及原始键/源码栈；精确CID数组关联得到104个seed publication、520个ID publication和1232个next-ID/count范围TASK，其中1144个有COMPUTE定义、88个没有。对confidence、scheduler、keep-length retrieval及confidence head的源码位置查询均为0命中；这不是配置为False或没有消费者的证明。没有服务器、模型、Torch/CANN/NPU或replay运行。

profile不能单独决定值语义，随后仅复用3份固定hash的proposer/utils/worker局部源码及既有metadata协议证据。选中padded路径中，当前target sampled IDs先产生next-ID和valid-count，再进行拒绝修正、draft输入和metadata准备；merged draft先运行decoder，再完成Markov序列，最后才可调用confidence scheduler。因而本轮晚出的confidence keep长度不能反过来裁掉已经需要的当前draft metadata或noncausal query/KV行。当前target接受状态、当前draft、下一次target验证的proposal长度、下一次验证后的接受状态是不同代的数据。

全局TopK的结果也不能直接当成待验证token位置。当前实现把TopK索引换算成request IDs、scatter-add得到keep-count，取出结果时按每个request截取token前缀。例如一行4个survival都为1，min_k=1、额外选中原位置3，是一个允许的平分选择；代码最终keep=2，发布位置0、1，而不是0、3。枚举65028个数学候选子集，其中14182个符合TopK值顺序，8938个选中位置集不同于最终发布前缀。这些是源契约下的反例，不是native TopK实际tie策略或当前数值的恢复。

metadata必须基于最终前缀和请求映射。B=2、D=3、budget=2可得到keep=[3,1]，对应纯decode query长度[4,2]、QSL=[0,4,6]；用统一[2,2]代替会得到[0,3,6]，总token数同为6，owner行归属却不同。将请求顺序反转也会改变QSL。相同总量、shape或图bucket不能证明可以复用原metadata。实际token-list宽度、consumer cap、zip长度、request ID/generation、调度后row map、CP/DP/group及全部plan输入均需匹配。零draft提前返回、scheduler None/关闭、宽度变化、非parallel continuation分别保留条件，未推定旧keep bank已更新。

confidence keep与验证后的valid-count也不可互换。同一个待验证k可以因后续拒绝/丢弃得到0..k+1个有效结果；它不能预测下一轮接受状态。当前next-ID helper还按字面谓词x!=-1且x<V计数，再取count-1位置：有洞的[7,-1,9]给出count2、next=-1，不能改成rightmost-valid的9；小于-1的负数也仍可能通过原谓词。候选必须保留原INT32 next-ID、INT64 count、backup、discard及独立GPU/CPU发布消费者。

可研究的跨边界组合是：在动态分支确实启用、完整request/width/epoch契约成立时，让最终keep-count生产者同时生成兼容的下一次验证prefix-scan/owner输入，或向metadata producer传播精确需求。它不等于把opaque SMLA/QLI callback直接塞进confidence kernel。既有异步submit、input-ready/reuse/frontier、capture固定地址、零query写入和旧reader退役仍须满足；只需真实值/range依赖，不新增一个无证据的全局barrier。周期预算重算要先得到本轮正确budget，其他步可使用已知budget，不能一律等同处理。

2834组B0..4、D0..5的布局参考用例核对了前缀、QSL和owner。明确选择的新ABI若保留INT32 k[B]、QSL[B+1]和可选owner[Q]，逻辑内容为4B+4(B+1)+4Q；这不是实际分配、HBM必要流量或现存device scan成本。当前host int(k)/prefix retrieval仍是观察者，尚未证明可以删除它，也没有凭新ABI认领4B读取、host同步或关键路径收益。原FP32 confidence/sigmoid/cumprod、globalTopK ties、periodic state、RNG及score消费者约束均保留；单独的embedding复用和F31 summary融合是既有候选，收益不重复累加。

中央新增REL129-01并补充既有HG73-TAIL和HG85-VERIFY-CONTEXT-DRAFT，条件关系135→136，待核对460→459，联合组仍26。M20-OP-10及M21-S10等既有分支获得跨代证据，语义行仍358/26；局部13节点/15边的代际DAG未当成新增native算子或全图补齐，部分全图仍854/1346/48提示。6465项前置核验通过，全部本地查询已结束。原9类/1352次字段缺口、849缺PMU、16个phase未绑定图、3136项含糊归属均不变，整模块关闭0，物理MFU及最低时间下界未建立，整体目标继续。
'''
    inputs=[Path(r['path'])for r in read(O/'verify-metadata-inputs-v129.json')['inputs']]
    updates={cid:dict(review_v129=dict(evidence=names,pair=['F30','F31'],relation_clause='REL129-01',
        current_draft_metadata_not_dead_from_later_keep=True,TopK_positions_not_published_prefix=True,
        accepted_count_is_distinct_value_and_generation=True,actual_runtime_guards_or_joint_fusion_proved=False,
        no_new_native_or_physical_gain=True))for cid in ['F30','F31']}
    persist_refinement(version=129,previous=128,title='DSpark verification prefix and metadata generation composition',
        summary=dict(filename='verify-metadata-summary-v129.json',previous_turn_classification='PROGRESS:v128 native disk scalar plan;PASS8619',
            **detail,refined_candidates=2,refined_modules=2,branch_records=358,semantic_modules=26,relation_clauses=136,higher_order_groups=26,
            pending_central_pair_reconciliation=459,partial_graph_node_rows=854,partial_graph_edge_rows=1346,structural_pair_hints=48,
            scoped_reviewed_regimes=2073,scoped_reviewed_executions=285598,pending_regime_reconciliation=9,pending_execution_reconciliation=1352,
            model_operator_implementation=False,native_test=False,actual_MFU_numeric_physical_ceiling_or_speedup_proved=False,
            Pad_authorization='GRANTED_AND_CONSUMED',remote_job_status='Alllocal32cachedDB/profile/source/reference checks complete;no remote/device/Torch/CANN/model/replay orserviceswork.'),
        changes=['Reconcile onegenuinependingF30-F31operatorrelation acrosscurrentacceptedstate/currentdraft/laterkeep/nextverification metadata.',
            'Establish counterexamples toTopKposition reuse,uniform/same-total/request-reordered metadata andkeep-as-accepted-count;retain exactprefix/cap/zip/state semantics.',
            'Validate2834layouts/65028TopKsubsets anddeclare newABIlogicalcontent only;preserveallruntime/numeric/bank/frontier andnative/fullscope gates.'],
        candidate_updates=updates,module_updates={
            'M20':dict(candidates=[],finding='GlobalconfidenceTopK publishesrequestcounts thenprefixes;laterkeepcannotretroactivelyremove currentdraftdecoder metadata andcannotpredict targetacceptedstate. Numeric/policy/native applicability remainopen.',evidence=names,gap=gap),
            'M21':dict(candidates=[],finding='Nextmetadata maycomposewith finalkeep onlyafteractualprefixwidth/cap/rowmap/request/bank generation. Equalshape/totalQ isinsufficient;existingaccepted-count/next-ID andfrontiers retained.',evidence=names,gap=gap)},
        remaining=remaining,action=action,
        jobs=[dict(kind='profile',status='pass',description='All32pinnedcachedDB source ancestry/CIDjoins:183813APIs/104seed/520IDpublication/1232next-IDsurface;absence ofconfidence notflagproof. Localqueryterminal0.',artifacts=[N/'verify-metadata-profile-run-v129.json',O/'verify-metadata-profile-scope-v129.json']+inputs),
            dict(kind='design-check',status='pass',description='2834layouts,65028TopKsubsets,conditionalgenerationDAG/counterexamples and6465preflightchecksPASS;onecentralpairreconciled,whole/nativegatesopen.',artifacts=[O/'verify-metadata-preflight-v129.json',O/'verify-metadata-validation-history-v129.json']+[O/n for n in names]+[O/'coverage-current-obligations-v129.json',O/'whole-scope-requirement-audit-v129.json'])],
        report=report,read_first=names+['coverage-current-obligations-v129.json'],publication_rows={},gate_updates=dict(overall_complete=False),
        scripts=[N/'query_verify_metadata_profile_v129.py',N/'verify_metadata_relation_models_v129.py',N/'analyze_verify_metadata_relation_v129.py',N/'verify_verify_metadata_relation_v129.py',N/'persist_profile_refinement_v90.py'],
        executor_prefix='verify-metadata',main_script=Path(__file__),source_read=True)


if __name__=='__main__':main()
