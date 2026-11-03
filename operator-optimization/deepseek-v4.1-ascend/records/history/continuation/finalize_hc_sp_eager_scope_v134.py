"""Seal remaining eager HC/SP evidence; do not count old relations as new pairs."""
from pathlib import Path
from copy import deepcopy
from finalize_origin_coverage_v9 import N,O,read,write
from persist_profile_refinement_v90 import persist_refinement


def main():
    assert not(O/'REVIEW-CHECKPOINT-v134.json').exists()
    pre=read(O/'hc-sp-eager-preflight-v134.json');assert pre['status']=='PASS'and pre['checks']>100000
    names=['hc-sp-eager-inputs-v134.json','hc-sp-eager-intake-v134.json','hc-sp-eager-native-frontiers-v134.json','hc-sp-eager-coverage-reconciliation-v134.json',
        'hc-sp-eager-observed-timing-and-geometry-v134.json','hc-sp-eager-typed-carrier-and-padding-models-v134.json','hc-sp-eager-operator-source-contract-v134.json']
    detail=dict(evidence=names,new_eager_HC_SP_frontiers=960,CPU_forward_roots=24,total_saved_attention_SP_frontiers=4472,remaining_saved_attention_SP_frontier_joins=0,
        positive_QA=960,positive_SWA_KV=960,positive_C1=24,positive_C2_KV=72,positive_C2_score=72,positive_Engram_guard_boundaries=48,aux_reader_boundaries=72,
        new_pair_clauses=0,new_groups=0,new_main_candidates=0,new_native_promotions=0,new_device_work=False,
        actual_tensor_parameter_native_new_M_parity_or_crossrank_episode_bound=False,whole_requirement_closed=False)
    def fork(n):return deepcopy(read(O/(n+'-v133.json')))
    def save(n,d):write(O/(n+'-v134.json'),d)
    touched=['F01','F02','F03','F06','F17','F32','F33'];clause_ids=['REL73-13','REL73-16','REL74-14','REL133-01','REL133-02','REL133-03']
    rel=fork('candidate-relation-clauses')
    for r in rel['clauses']:
        if r['id']in clause_ids:r['eager_HC_SP_evidence_v134']=deepcopy(detail)
    for g in rel['higher_order_groups']:
        if g['id']in ['HG93-HC-ATTENTION-FORK-DEADLINE','HG88-SWA-C2-TYPED-PROJECTION','HG74-C1']:g['eager_HC_SP_evidence_v134']=dict(**detail,whole_group_closed=False)
    rel.update(parent_relation_index='candidate-relation-clauses-v133.json',new_pair_clauses=0,new_higher_order_groups=0,clause_count=142,selected_pair_count=142,eager_HC_SP_evidence_v134=detail,overall_complete=False);save('candidate-relation-clauses',rel)
    selected={tuple(sorted((r['a'],r['b'])))for r in rel['clauses']if r['id']in clause_ids}
    cov=fork('candidate-relation-coverage');nav=fork('candidate-pair-navigation')
    for table in [cov,nav]:
        for r in table['pairs']:
            if tuple(sorted((r['a'],r['b'])))in selected:r['eager_HC_SP_evidence_v134']=deepcopy(detail)
    cov.update(parent='candidate-relation-coverage-v133.json',parent_relation_index='candidate-relation-clauses-v134.json',eager_HC_SP_evidence_v134=detail,
        scope='142scopedconditionalpairs/26groups/453remaining. v134closesonlythe960eagertargetHC/SPsource-frontierjoinso4472savedsuchcallsarecovered;no newpairsornative/wholeclosure.',overall_complete=False);save('candidate-relation-coverage',cov)
    nav.update(parent='candidate-pair-navigation-v133.json',parent_footprints='candidate-operator-footprints-v134.json',parent_relation_index='candidate-relation-clauses-v134.json',eager_HC_SP_evidence_v134=detail);save('candidate-pair-navigation',nav)
    fp=fork('candidate-operator-footprints')
    for r in fp['candidates']:
        if r['id']in touched:r['eager_HC_SP_evidence_v134']=dict(evidence=names,no_new_native_operator_atoms=True,source_frontiers=960,all_runtime_parameters_or_new_ABI_closed=False)
    fp['parent']='candidate-operator-footprints-v133.json';save('candidate-operator-footprints',fp)
    op=fork('operator-analysis-coverage-ledger');op['eager_HC_SP_evidence_v134']=detail;save('operator-analysis-coverage-ledger',op)
    obligations=fork('coverage-current-obligations')
    for r in obligations['obligations']:
        if r['id']in ['G00','G01','G02','G05','G06','G07','G08','G09','G10','G11','G13']:r['eager_HC_SP_evidence_v134']=deepcopy(detail)
        if r['id']=='G07':r['remaining']='142conditionalpairs/26groups/453centralpairsandallvariantsopen. The4472savedHC/SPfrontiersarenowsource-scoped,notnewABI/allreader/actualcrossrankclosure. PriceQAraw/ownerroute,C1secondBF16andC2twoFLOATplanes,pack/collective/numericM/paddingandHC/Postbanksinonegraph. SWAonly.1,C2.5orwithQA.75arenamedpayloadmodelsnotunconditionalcarrierorlatencygains.'
        if r['id']=='G13':r['remaining']=(
            'v134completesnamed960eagertargetHC/RMS/SP/Post/premixsourcejoinwithin24CPUforwardranges. 3512+960=4472allv66attentionSPkeys,zeroLEFTinthissavedpopulation. All960COMMretainnoCOMPUTE_TASKnode;no missingcostzero/pointer/clientstep/crossrankpromotion. '
            'Positive960QA/960SWA/24C1/72C2KV/72C2scoreand48Engram/72auxguardsbound. LargeM72SWA-C2triplesarenowsource-rolejoined;priorv88formulaeareprioranswers. C1eager24doesnotresolvecapturedgraph80SWA/C1ambiguities. '
            'N6668reportedKVvsA6672and40QA830recordswithHC834needpadding/trim/owner/numeric/errorcontracts. AllglobalrawIndexer/QA/C1/C2/rstdreaders,loadedweights/HF32/FP32/nativeM/alias/bank/captureepochsremain. '
            'Nextprofile-firstreviewpendingF01-F33:HCcoeff/YforkcombinedwithC1+SWAproducer/carrier choices,usingnowpositive24eagerC1recordsandexistingC1weight/norm/round/readercontracts;do notclaimsameweightCSEfromequalshapeorfilltheold80graphroles. '
            'All9/1352nativeoperandfields/globalbranch/graph/physicalcapacity/MFU/lowestgatesremain. No newNPU/model/capture/replayauthorization;Padv120consumed,nopendingapproval.')
    obligations.update(parent='coverage-current-obligations-v133.json',new_native_or_whole_closures=0,overall_complete=False);save('coverage-current-obligations',obligations)
    audit=fork('whole-scope-requirement-audit');audit.update(previous_goal_turn='PROGRESS:v133HC/SP3relationsandgenerations;postcheckPASS231297',input_checkpoint=133,followup_checkpoint=134,eager_HC_SP_evidence_v134=detail)
    for r in audit['requirements']:
        if r['id']in ['ALL_MODULE_CONDITIONAL_BRANCHES','CROSS_OPERATOR_LAYER_CARD_ITERATION','THEORY_AND_MEASURED_COMPARISON','CRITICAL_EVIDENCE_AND_FINAL_CLOSURE']:r.setdefault('evidence',[]).extend(names)
    save('whole-scope-requirement-audit',audit)
    write(O/'PROFILE-CODE-QUESTION-GATE-v134.json',dict(status='SAME_TID_FORWARD_CONTAINMENT_40_SOURCE_ROLE_CALLS_EXACT_CID_THEN_TYPED_OPERATOR_PLANES',questions=read(O/'hc-sp-eager-operator-source-contract-v134.json')['questions'],
        source_scope='OnlypinnedtargetHC/Engram/aux/CP/compressor71/C2projection530-531segments;noframeworksurvey/newSDK/remote/deviceexecution.',
        no_model_operator_implementation=True,no_remote_Torch_CANN_NPU_model_or_replay=True,Pad_authorization='GRANTED_AND_CONSUMED',overall_complete=False))
    write(O/'hc-sp-eager-validation-history-v134.json',dict(events=[
        'Predecessorv133revalidatedPASS231297/manifest134;previousgoalturnchanged3HC/SPrelationsandM19F01link,notwait/statusonly.',
        'Intake960COMMhasnoCOMPUTE_TASKbutpositiveHcclAllGatherCID/sourcechain. FounduniqueCPUforwardancestorworker2553withsameTid/containmentandhostexecute_modelmarker;24rootsx40orderedsourcecalls.',
        'Firsthelperpy_compilehadtuplecomprehensionwhitespaceerrorbeforeI/O;archivedscript/note,correctedthenprofilequerycompleted. Notatargetcodechangeorfailedfusionexperiment.',
        '960exactHC/RMS/SP/RS/Post/FFNPre/Postfrontiers;new960QA/960SWA,24C1fromcompressor71and72eachFLOATC2KV530/score531.48positiveEngram/72auxguardsand24initialHCgroupsretained.',
        'OriginalCPUforwardrangesandCIDsbindlocaloperatorcalls,notdeviceintervalcontainment,formalclientrequests,crossrankepisodes,parametersorhistoricaltensoraddresses.',
        'NativeNKV6400/6528/6668versussenderM800/816/834;QA834280timesandQA83040timesinlastgeometry. Uniformownerpadding/trimcostnowexplicit;SP/CPmapstillnotrecovered.',
        'Drafttimingscopetextinitiallyassumed196internalCOMMrecords;full960populationhas196x935,207x19,210x4,193x2. Correctedbeforeseal;nativeintervalstatsnotbandwidth/cycle/criticalpathorbenchmarkgains.',
        'v88largeMtripleshadbeenreferencegeometries;72nowhavesource-rolejoinedSWA+twoFLOATC2tasks. Existing.5typedcontentformula notnewdiscovery;addingQAroutegives.75partial-planecontent. C1distinctBF16planesplusQA.45onlyunderallreadergates.',
        'Preflight'+str(pre['checks'])+'PASS. Union3512old+960newkeys=4472allv66attentionSPcalls;namedjoincomplete,wholegoalnotcomplete. No newcentralpairs/nativepromotions/semanticfamilies/globalgraphnodes.142pairs/26groups/453pendingunchanged.'],overall_complete=False))
    gap=('All4472savedattentionSPsourcefrontiersnowjoined;needactualSP/CProwmaps/allrawQA/Indexer/C1/C2/rstdreaders,matchingweight/gamma/eps/HF32-FP32/nativeM/rounds/errorsemanticsandpackedtypedcollective/capture/bankevents. '
        'N6668/A6672andQA830/M834paddingpricedonlyasdeclaredreference;C2twoFLOATplanesandC1distinctBF16outputsremain. PositiveeagerC1doesnotattestweightidentityorresolvethe80capturedgraphroles. '
        'COMMnoTASKisnotzerocost;intervalsarenotexclusivephysicalcapacity/causalgain. No nativeoperand9/1352,physicalMFU/lowestorwholepair/allbranchclosure.')
    remaining=[r['id']+': '+r['title']+'. '+r['remaining']for r in obligations['obligations']]
    remaining.append('v134savedHC/SPsourcejoincomplete4472/4472,960largeeagerin24forwardroots. No newpairs:142/26/453.358/26semantic854/1346/48partialgraph9/1352native849PMU16phase3136ambiguity0wholeunchanged.')
    action=(
        'ContinueoriginalGoal;v134PROGRESScompletes960eagertargetHC/SPfrontierjoinandlargeMtypedplanecosts,notnewnativeoptimizationorwholeclosure. Readhc-sp-eager-summary/frontiers/coverage/observed-timing/typed-carrier/source/currentobligations-v134;verify_hc_sp_eager_scope_v134mustPASS. '
        '24sameTidCPUforwardranges(worker2553)insidehostexecute_modelmarkerscontain40orderedsourceHC889/RMS896/AG898/RS901/Post902/FFNPre905/FFNPost919calls. ExactCID→TASKorCOMM;all960AGCOMMnoTASKpreserved. CPUcontainmentnotNPUwholebarrier/clientrequest/crossrankepisode/pointeridentity. '
        '960QA/SWA,24C1compressor71BF16,72C2KV530FLOAT/72C2score531FLOAT;48Engram/72aux/24initialguards. Sourceblock20C1separatefromSWA;C1eagerdoesnotresolveold80capturedgraphambiguitiesorproveweightCSE. '
        'NKV6400/6528/6668,M800/816/834,A6400/6528/6672;lastgeometryQA834280x/83040x. OwneruniformQAadds4rowsat40records;SP/CPmapsandnativeM/trim/errorsemanticsstillunbound. '
        '4472/4472savedattentionSPsourcefrontiersnowcovered(3512prior+960new);remainingnamedjoin0,notallbranches/loadedABI/allreaders. No newpairs:142conditional/26groups/453pending. '
        'NamedpartialcarrierSWA.1;plusQA.35;SWA+C1.2/withQA.45;SWA+twoC2FLOAT.5/withQA.75. Existingv88.5notnewformula;72largeMtriplesnowboundbeyondpriorgeometryreference. RawIndexer/QA/C1/C2/rstdotherreadersstillneedclosure;retainedrawHmakesplanesadditionalratherthansavings. Multipletypedcollectives/bitpreservingpackhascosts;novalue-destroyingFLOAT→BF16conversion. '
        'NativeCOMMdurationsvarynonmonotonicallywithgeometry;internalrecordcounts193/196/207/210preserved,notfixed196. NoCOMM_TASKdoesnotmeanzerotime. IntervalsnotHBM/wire/latencygain/physicalcapacity/MFU/lowest. '
        'NextpendingF01-F33HCcoeff/YforkwithC1+SWAproducer/carrier,usingpositive24eagerC1andpriorC1weight/norm/rounding/readerfacts. Do notblindlyre-countoldsplit/typedcarrieranswersorfillold80graphambiguities. '
        '358/26semantic854/1346/48partialgraph2073/285598conditional9/1352pending849PMU16phase3136ambiguity0whole/noallmodulephysicalMFU/lowestunchanged. Alllocaljobscomplete,no remote/Torch/CANN/NPU/model/capture/replay/implementation. Padv120consumed,nopendingapproval. Do noteditsealedv134/rerunfinalizer. Versionedpostcheck/CURRENTmatchingRECOVERY;TaskACTIVE Loop004EVALUATING GoalACTIVE.')
    report='''本轮补齐剩余960次eager target通信与HC/RMS/Post的关联，使原始v66清单中的4472次attention SP均有相应源码角色前沿。这个完成范围仅指该固定profile清单，不能代表全部模块、条件分支、实际融合合法性或整体目标完成。

这些960次通信没有唯一COMPUTE_TASK节点，但有HcclAllGather的精确CID和调用栈。在同一globalTid上找到唯一的forward CPU父区间，再按40层HC889、RMS896、AG898、RS901、Post902、FFNPre905、FFNPost919的源码调用序列配对。24个forward各有40组，均处在正面的host execute_model记录内。随后通过CONNECTION_IDS关联原生TASK或COMM，未按设备时间最近补造节点。CPU区间包含关系也不等同于设备区间、正式客户端step或跨卡同次迭代。

新关联包含960次Q_a、960次SWA KV、24次C1 BF16投影、各72次C2 KV与score FLOAT投影，并保留48处Engram正面调用、72处辅助状态读者和24组初始HC。C1的compressor71与SWA的multistream475是不同源码角色；C2在hidden.float()后分别执行530/531两次投影。相同形状不能证明C1与SWA权重相同或输出可CSE；本轮eager证据也不能替代旧的80处captured graph角色歧义。

每组320次通信对应sender M800/816/834，SWA KV报告的行数分别6400/6528/6668，而均匀gather行数为6400/6528/6672。最后一组中，280次Q_a为834行、40次为830行。统一按owner的834行投影，会在后40次增加4行Q_a计算；不能机械领取8倍工作减少。真正的SP/CP row map、receiver trim、padding错误行为及新M下数值等价性仍须满足。这里的N是算子报告行数，不直接等于客户端有效token数。

前移投影的接口内容必须覆盖全部活跃分支。以每行原BF16 hidden为基准，SWA512平面是10%；若还要传Q_a1280则为35%。保留第二份C1 BF16平面时，分别为20%或45%。C2必须保留两份FLOAT512：与SWA合计50%，再加Q_a为75%。v88已有不含Q_a的50%公式，本轮新增的是72组大M原生投影的正面角色关联，以及Q_a路由/padding一并计价。其它Indexer、raw输入、rstd、C1/C2和cache读者尚未闭合时，这些不是完整合法载荷；若仍需传原hidden，新平面甚至成为额外内容。

不同dtype可以研究多次typed collective或保位packed ABI，但不能把FLOAT直接转BF16以取得表面节省。新布局、对齐、打包、队列、ready事件、接收者行序、pool/state/RoPE/cache世代和原始kernel舍入都需计价。原始COMM内部记录数实际为196×935、207×19、210×4、193×2，不能固定假设每次196条。保留了HC、RMS、QA、SWA、AG/RS及Post的native区间分布；这些包含原调用的等待/传输及竞争，不是普适带宽、互斥计算时间、请求关键路径或优化前后benchmark。载荷比例不能直接变成延迟比例。

本轮没有新增中央pair，而是补强现有HC/SP、QKV/SP、C2/SP与C1/SP关系：142组条件关系、26个联合组、453组待核对仍保留。35主候选、7历史框架记录、358语义行/26模块和854节点/1346边/48提示的部分图不变。9类/1352次原生操作数字段、849个PMU缺口、16个phase未绑定图、3136项归属歧义均未被本轮关联消除，整模块关闭0。全程未执行设备任务或修改目标模型，整体继续。
'''
    inputs=[Path(r['path'])for r in read(O/'hc-sp-eager-inputs-v134.json')['inputs']];refs=read(O/'hc-sp-eager-native-frontiers-v134.json')['databases'];inputs += [Path(r['raw_database']['path'])for r in refs]+[Path(r['cached_native']['path'])for r in refs]
    old_candidates={c['id']:c for c in read(O/'candidate-profile-ledger-v133.json')['candidates']}
    updates={cid:dict(review_v134=dict(evidence=names,saved_attention_SP_frontiers=4472,new_large_eager_frontiers=960,positive_C1=24,positive_C2_pairs=72,all_saved_HC_SP_source_joins_done=True,
        actual_pointer_weight_native_new_ABI_or_all_readers_closed=False,new_pair_clauses=0,no_native_or_physical_gain=True),previous_conclusion_changing_gap_before_v134=old_candidates[cid].get('conclusion_changing_gap'),conclusion_changing_gap=gap)for cid in touched}
    persist_refinement(version=134,previous=133,title='Complete saved large eager HC/SP frontier joins and typed carrier costs',
        summary=dict(filename='hc-sp-eager-summary-v134.json',previous_turn_classification='PROGRESS:v1333HC/SPrelationsandgenerationguards;PASS231297',**detail,
            refined_candidates=7,refined_modules=4,branch_records=358,semantic_modules=26,relation_clauses=142,higher_order_groups=26,pending_central_pair_reconciliation=453,
            partial_graph_node_rows=854,partial_graph_edge_rows=1346,structural_pair_hints=48,scoped_reviewed_regimes=2073,scoped_reviewed_executions=285598,pending_regime_reconciliation=9,pending_execution_reconciliation=1352,
            model_operator_implementation=False,native_test=False,actual_MFU_numeric_physical_ceiling_or_speedup_proved=False,Pad_authorization='GRANTED_AND_CONSUMED',remote_job_status='Alllocalrawprofile/source/typed-content/verifierworkcomplete;no remote/device/Torch/CANN/model/replay/serviceswork.'),
        changes=['Complete960largeeagerHC/RMS/SP/Postjoinsvia24CPUforwardrangesandexactCID;4472/4472savedattentionSPfrontiersnowcoveredinsourcescope.',
            'Bind960QA/SWA,24C1BF16and72eachC2FLOATKV/score;preserveN6668/A6672andQA830padding,Engram/aux/premix/readergates.',
            'Refineexistingtypedcarrier/HCcompositionswithoutnewpairs:142/26/453;allnative/global/wholephysicalgatesretained.'],candidate_updates=updates,
        module_updates={mid:dict(candidates=[],finding=f,evidence=names,gap=gap)for mid,f in {
            'M06':'960largeeagerHC/SP/Post/premixsourcefrontiersand24initial/48Engram/72auxguardsnowbound;actualpointer/reader/epochandnewpartialhandoffremainconditional.',
            'M07':'960QA/SWAprojectionslinklocal800/816/834andglobal6400/6528/6668with40QA830cases. OwnerQA/trim/fullnorm/nativeMcostsremain,notunconditional8xfactor.',
            'M08':'24C1BF16and72two-FLOATC2projectionpairsboundbypositivesourcecalls;no weightCSEorFLOATreencodingclaim. Globalpool/state/cacheandrawreadercontractsretained.',
            'M23':'960COMM-onlycallsretainnativecommunicationcostwithnoCOMPUTE_TASK;all4472savedattentionSPsourcefrontiersjoined. Typedcarrier.1/.35/.2/.45/.5/.75contentmodelsnotlatencygains.'}.items()},
        remaining=remaining,action=action,jobs=[
            dict(kind='design-check',status='error',description='Retainedlocalhelperpy_compilesyntaxerrorbeforeprofileI/O;whitespacecorrected,no target/environment/devicechanges.',artifacts=[N/'analyze_hc_sp_eager_profile_syntax_v134a.py',O/'hc-sp-eager-build-note-v134a.json']),
            dict(kind='profile',status='pass',description='960eagerHC/SP/Postfrontiersin24CPUroots;exactrawCID/TASK/COMM/sourcecallsandtypedQA/SWA/C1/C2/Engram/auxrolesverified.',artifacts=[O/'hc-sp-eager-native-frontiers-v134.json']+inputs),
            dict(kind='design-check',status='pass',description='4472savedSPsourcejoinunion,padding/typedcarrier/nativeintervalscopeand'+str(pre['checks'])+'preflightPASS;existingrelationsrefined,no newpair/nativepromotion.',artifacts=[O/'hc-sp-eager-preflight-v134.json',O/'hc-sp-eager-validation-history-v134.json']+[O/n for n in names]+[O/'coverage-current-obligations-v134.json',O/'whole-scope-requirement-audit-v134.json'])],
        report=report,read_first=names+['coverage-current-obligations-v134.json'],publication_rows={},gate_updates=dict(overall_complete=False),
        scripts=[N/'prepare_hc_sp_eager_intake_v134.py',N/'analyze_hc_sp_eager_profile_v134.py',N/'analyze_hc_sp_eager_scope_v134.py',N/'verify_hc_sp_eager_scope_v134.py',N/'persist_profile_refinement_v90.py'],executor_prefix='hc-sp-eager',main_script=Path(__file__),source_read=True)


if __name__=='__main__':main()
