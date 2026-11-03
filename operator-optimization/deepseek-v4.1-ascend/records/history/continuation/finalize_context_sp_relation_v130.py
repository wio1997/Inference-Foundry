"""Seal scoped F29-F32 owner/packet/order analysis; no native fusion closure."""
from copy import deepcopy
from pathlib import Path
from finalize_origin_coverage_v9 import N,O,read,write
from persist_profile_refinement_v90 import persist_refinement


def main():
    assert not (O/'REVIEW-CHECKPOINT-v130.json').exists()
    pre=read(O/'context-sp-preflight-v130.json');assert pre['status']=='PASS'and pre['checks']>=26180
    names=['context-sp-inputs-v130.json','context-sp-profile-frontiers-v130.json','context-sp-source-contract-v130.json',
        'context-sp-hybrid-owner-models-v130.json','context-sp-recipient-owner-models-v130.json',
        'context-sp-order-and-sharing-witnesses-v130.json','context-sp-joint-graph-and-conditions-v130.json']
    detail=dict(evidence=names,source_order_frontiers=104,draft_SP_collectives=312,
        proposal_query_KV_typed_TASKs=264,positive_dummy_None_metadata_zero_TASKs=48,
        retained_main_feature_gathers=88,context_owner_options=19637,model_Pareto_cases=2460,recipient_cases=9330,
        actual_cache_recipient_or_weight_or_bank_identity_bound=False,
        new_scoped_pair_relations=1,new_pair_clauses=1,new_groups=0,new_main_candidates=0,
        new_native_promotions=0,new_device_work=False,whole_requirement_closed=False)
    contract=(
        'Same draftlayer sourceuses WKV/norm fordistinctcontext/query rows;shareweight work bycompatibleperlayer rowpacking,notCSE ofvalues. '
        'Afterqueryproject-before-SP,choose replicatedcontext,ownercontext orhybrid s underactualcache-recipient/communicator/row/parameter/numeric identity. '
        'Keep existingglobal5120context featuregather andlocalqueryraw readers;BF16-512 carrier isdraftSWA-only,not targetC1/C2 replacement. '
        'Forreplicatedreceiver fixed-countP8carrier,new per-sender content1024*(m_q+ceil(s/8)) versus10240*m_q;physicalwire/time unknown. '
        'Preservequery/contextpadding,role-specificRoPE,per-slotcontext-before-queryWAW,noncausalreadiness andextendedcontext-banklifetime. '
        'Earlyjointcontext3maps anddelayedperlayerC/Qpacking arecoupledschedules,notindependentcredits.48dummyNone-metadatazeros needseparate startup/collectiveguards.')
    rel=deepcopy(read(O/'candidate-relation-clauses-v129.json'));assert not any({r['a'],r['b']}=={'F29','F32'}for r in rel['clauses'])
    rel['clauses'].append(dict(id='REL130-01',a='F29',b='F32',relation='JOINT_CONTEXT_QUERY_WEIGHT_OWNER_AND_PROJECTED_CARRIER',
        operator_claim=contract,evidence=names,applicable_domain='Selected draftSWA-only/proposal paths withper-layerWKV/normidentity andnewlocalQ/projectedKV interface;cache-demand/runtimeprecision/loadedobjects remainconditional.',
        conditional_compatibility=True,all_runtime_gates_proved=False,speedup_additive=False,critical_path_gain_us=None,
        context_SP_composition_v130=deepcopy(detail)))
    group=next(g for g in rel['higher_order_groups']if g['id']=='HG85-VERIFY-CONTEXT-DRAFT')
    assert 'F32'not in group['members'];group['members'].append('F32')
    group['context_SP_composition_v130']=dict(**detail,contract=contract,whole_group_closed=False)
    rel.update(parent_relation_index='candidate-relation-clauses-v129.json',new_pair_clauses=1,new_higher_order_groups=0,
        clause_count=137,selected_pair_count=137,context_SP_composition_v130=detail,overall_complete=False)
    write(O/'candidate-relation-clauses-v130.json',rel)
    cov=deepcopy(read(O/'candidate-relation-coverage-v129.json'));row=next(r for r in cov['pairs']if {r['a'],r['b']}=={'F29','F32'})
    row.update(scope_state='SCOPED_OPERATOR_RELATION_REVIEWED_CONDITIONAL',clauses=['REL130-01'],whole_pair_all_variants_closed=False,
        new_critical_path_gain_us=None,context_SP_composition_v130=deepcopy(detail))
    cov.update(parent='candidate-relation-coverage-v129.json',parent_relation_index='candidate-relation-clauses-v130.json',
        scoped_conditional_pairs=137,remaining_pair_relations=458,context_SP_composition_v130=detail,
        scope='137 scopedconditionalpairs/26groups/35modelcandidates;458centralpairs remain. F29-F32 jointcontext/queryowner/content/order reviewed inselecteddraftSWA domain;allruntime/allvariants open.',overall_complete=False)
    write(O/'candidate-relation-coverage-v130.json',cov)
    nav=deepcopy(read(O/'candidate-pair-navigation-v129.json'));row=next(r for r in nav['pairs']if {r['a'],r['b']}=={'F29','F32'})
    row.update(conditional_clause_ids=['REL130-01'],semantic_pair_reconciliation_done_in_stated_scope=True,
        whole_pair_all_variants_closed=False,context_SP_composition_v130=deepcopy(detail),evidence=names)
    nav.update(parent='candidate-pair-navigation-v129.json',parent_footprints='candidate-operator-footprints-v130.json',
        parent_relation_index='candidate-relation-clauses-v130.json',scoped_conditional_pairs=137,remaining_central_pair_reconciliation=458,context_SP_composition_v130=detail)
    write(O/'candidate-pair-navigation-v130.json',nav)
    fp=deepcopy(read(O/'candidate-operator-footprints-v129.json'))
    for r in fp['candidates']:
        if r['id']in ['F29','F32']:r['context_SP_composition_v130']=dict(evidence=names,no_new_native_operator_atoms=True,
            local_graph='context-sp-joint-graph-and-conditions-v130.json',all_runtime_or_native_parameters_bound=False)
    fp['parent']='candidate-operator-footprints-v129.json';write(O/'candidate-operator-footprints-v130.json',fp)
    op=deepcopy(read(O/'operator-analysis-coverage-ledger-v129.json'));op['context_SP_composition_v130']=detail
    write(O/'operator-analysis-coverage-ledger-v130.json',op)
    obligations=deepcopy(read(O/'coverage-current-obligations-v129.json'))
    for r in obligations['obligations']:
        if r['id']in ['G00','G01','G02','G05','G06','G07','G08','G09','G11','G13']:r['context_SP_composition_v130']=deepcopy(detail)
        if r['id']=='G07':r['remaining']=(
            '137conditionalpairs/26groups,458centralpairs andallvariants remainopen. '
            'F29-F32 jointgraph mustchoosecontextreplication/owner/hybrid andearlyjointcontext versusperlayerC/Qpacking underactualreceiver/communicator/parameter/row/numeric contracts. '
            '88mainfeaturegathers remain;newquery/contextpadding,localQArawreaders,RoPEroles,per-slotWAW andextendedcontext-banklifetime arecosts. '
            'Sender/endpointcontentsanddenseworkmodels are notphysicalnetwork/HBM/cycles orindependentspeedups.')
        if r['id']=='G13':r['remaining']=(
            'v130 completes targetedF29-F32 conditionalcomposition:104exact-keysource-orderfrontiers,312draftSPcalls,264positiveproposalKV tasks and48positiveNone-metadata dummyzeros. '
            'Do notresurrectthefailedall-SP-has-query-KV assumption orbindtheseBF16[20,5120]zeros tothe80pendingBOOL24graphfields. '
            'Cache recipients/actualslots/groupmembership/rowmaps/loadedinstances/weightreplicas/numeric/generation andnewABIallreaders remainunbound. '
            'Forreplicatedreceiver model,thefullcontextowner path mayincreasecarriercontent greatly;hybridthreshold s<=72m_q isnotahardwareoptimum. Context-localrecipient alternative needsitsownactualdemand proof. '
            'Next reviewremaining F27-F32 relation aroundthe retainedmainfeaturegather,checkpointrowownership/full5120norm andupstreamSP/auxchoices,usingexistingprofilefirst;keepv34row-owner/feature-consumer alternatives asprioranswers,notnewdiscovery. '
            'All9/1352nativefields,globalbranch/graph/physicalcapacity/MFU/lowest-timegates remain. No newNPU/model/capture/replay authorization orpendingapproval;v120Pad grantconsumed.')
    obligations.update(parent='coverage-current-obligations-v129.json',new_native_or_whole_closures=0,overall_complete=False)
    write(O/'coverage-current-obligations-v130.json',obligations)
    audit=deepcopy(read(O/'whole-scope-requirement-audit-v129.json'))
    audit.update(previous_goal_turn='PROGRESS:v129F30-F31 keep/prefix/acceptedstate relation;postcheck PASS6502',input_checkpoint=129,followup_checkpoint=130,context_SP_composition_v130=detail)
    for r in audit['requirements']:
        if r['id']=='CROSS_OPERATOR_LAYER_CARD_ITERATION':r.update(conditional_pair_relations=137,central_pairs_not_reconciled=458,higher_order_groups=26)
        if r['id']in ['ALL_MODULE_CONDITIONAL_BRANCHES','CROSS_OPERATOR_LAYER_CARD_ITERATION','THEORY_AND_MEASURED_COMPARISON','CRITICAL_EVIDENCE_AND_FINAL_CLOSURE']:r.setdefault('evidence',[]).extend(names)
    write(O/'whole-scope-requirement-audit-v130.json',audit)
    write(O/'PROFILE-CODE-QUESTION-GATE-v130.json',dict(status='EXACT_CACHED_CONTEXT_DRAFT_SP_AND_QUERY_OR_ZERO_BRANCH_THEN_ONLY_ROW_PROJECTION_PUBLISH_CONTRACTS',
        questions=read(O/names[0])['questions'],source_scope='Onlypinneddraft/model/V41/legacyCP projection/norm/RoPE/write/None-metadata segments;noframeworksurvey,newSDKorremoteexecution.',
        all_SP_calls_not_assumed_to_have_projection=True,no_model_operator_implementation=True,
        no_remote_Torch_CANN_NPU_model_or_replay=True,Pad_authorization='GRANTED_AND_CONSUMED',overall_complete=False))
    write(O/'context-sp-validation-history-v130.json',dict(events=[
        'Predecessorv129 revalidated PASS6502/manifest63;previousgoalturn completedonependingoperatorrelation,notwait/statusonly.',
        'Reusev66nativeCOMMroles andv113context/Tilefrontiers;all104source-orderwindows haveexactly3draftSPcalls. 88precedingmainprojectionwindows eachcontainonefeatureAG,count640*C.',
        'Initialexploratorycommand hadasyntaxerrorbeforeI/O. Firstpersistedprofilequery thenincorrectlyrequiredonequeryKVforeverySPcall;failedondummywindow0. Retainedscript/assumption-errorreceipt,notusedasabsenceproof.',
        'Correctedqueryfinds264proposalBF16KVMatMuls and48positiveZerosLikecalls fromdsa_v41.py698None-metadata branch. All312COMMkeys/264queryTASKkeys/48zeroTASKkeys unique;rawCID/sourceframeverificationkeepsdummy/proposal separate.',
        'Sourceper-layercontext/query shareWKV/normexpressions butdifferentinputvalues;draftroleSWA-only sourceexcludesC1/C2extraoutputs. NativeprojectionM/numerics/parameter/cachemembershipstillgated.',
        '19637s-options/2460two-countParetooptions across5positivegeometries,9330contextreceiver/computechoices andcontext-before-queryslot counterexample checked. No HCCLwire/cycle/latency ornewimplementation claimed.',
        'ExistingmainfeatureAG retained. NewC/Qrowpacking canextendcontext-bankreaders intodraftlayers andcompeteswithearlyjointcontextprecompute;onegraphmustincludeweights/packs/queues/oldreaders.',
        'Preflight'+str(pre['checks'])+'checksPASS across32rawDBpins,exactrawCOMM/TASK/Compute/sourcebranchrows,sourcepins/formulas/witnesses andpredecessormanifest. Alllocaljobs terminal,no remote/devicework.',
        'REL130-01 addsF29-F32 andextendsHG85membershiptoF32;137pairs/26groups/458pending. No newmaincandidate/nativeatom/globalgraphrow/semanticfamily orwholemodule closure.'],overall_complete=False))
    gap=(
        'Need actualcommunicatormembers/rankorder,per-layerweight/gamma/eps/numericidentity,context/queryrow/slot/position maps,cache recipients/allreaders andcapture/bank generations. '
        'Keepdistinctinputvalues,localqueryrawconsumers,existingglobalcontextfeatureAG,role-specificRoPE andper-addresscontext-before-queryWAW. '
        'OwnercontextaddsKVdeliveriesinthe replicatedreceiver domain;recipient-localcontext needsactualdemand;allpadding/newpacks/extendedbanklifetimesandcompetingearly/late schedules mustbepriced. '
        'No hardwareoptimal s,nativeparity,wire/latency/MFU/lowest-time orallbranches/wholepairclosure established.')
    remaining=[r['id']+': '+r['title']+'. '+r['remaining']for r in obligations['obligations']]
    remaining.append('v130F29-F32 conditionalrelation reconciled:137pairs/26groups/458pending.104frontiers/312SP split264proposalKV+48positiveNone-metadatazeros;88mainAGretained.19637owner/9330recipientmodels notruntimeproof.358/26semantic,854/1346/48partialgraph,9/1352nativepending,849PMU/16phase/3136ambiguity,0whole unchanged.')
    action=(
        'ContinueoriginalGoal;v130PROGRESS isF29-F32 draftcontext/queryweight-owner/carrier/order composition,notnativeoptimizationorwholeclosure. '
        'Readcontext-sp summary/profile-frontiers/source-contract/hybrid-owner/recipient-owner/order-sharing/joint-graph/current-obligations-v130;verify_context_sp_relation_v130 mustPASS. '
        'v129postPASS6502.104source-orderfrontiers/312draftSPcalls:264proposalwindows eachoneBF16M10/20x5120-by512x5120KVtask;48dummywindows positiveZerosLikeBF16[20,5120]atdsa_v41.py698(metadataNone). Not80pendingBOOL24fields. Firstall-SP-has-KVassumptionfailed/preserved;canonicalquerysession3781terminal0. '
        '88mainfeatureAGs exactwindow/count640*C/dtypeBF16/P8retained. ContextC12/24/6400/6528/6668,queryQ10/20,senderrowsmq2/3. GPUkeys/orderarenotrequest/layerinstance/pointer/weight/cache identityorcriticalpath. '
        'ContextmainnormH5120 andcurrentlayerqueryhidden aredifferentvalues;sourcewkv/normsameper-layerobjectsupportsrowpackingwithsharedweights only. DraftroleSWA-only,noC1/C2carriercredit. PreserveBF16GEMM/normanddistinctcontextmoduleRoPEversusqueryinplaceRoPE. '
        'Aftergloballyavailablecontextmainnorm,chooseC-sreplicatedcontext+ceil(s/8)ownercontext+mqquery rowspercard. UniformBF16-512carrier per sender=1024*(mq+ceil(s/8));oldqueryhidden=10240mq. Budgetcut s<=72mq onlycontent,notphysicaloptimum. '
        '19637options/2460countPareto rows;9330recipientmodelcases. Ractualcache/readers unknown:recipient-localcontext canavoidadditionalKVdeliveriesifrawcontextavailable;replicatedcacheownerpath addsoutputdelivery. Groupcount/namealone notmembershipproof. '
        'C24/Q20/mq3fullowner:packet30720→6144B,retainedmain+3packets122880→49152B;projectionrows44→6. C6400/Q10/mq2fullowner:packet20480→821248B,combined8253440→10655744B;hybrids144retainspacket20480B/rows6410→6276. Logicalcontent/densework,notHBM/wire/latency/speedup. Paddingq16/24forvalid10/20retained,nofree8xfactor. '
        'Rank-majorC/Qscattercanreverseper-slotC→QWAW(witnessslot7finalquery29vscontext11);newABIrequiresrole-awarewriterorder/rangeRAW/oldreaders/noncausalready. DelayingCuntileachquerylayerextendscontextbanklifetimethroughlastlayer. Earlyjoint3contextmaps andlateC/Qpacking arealternative/coupledschedules,notadditivecredits. '
        'REL130-01F29-F32;HG85addsF32member,allsubsets/runtimejointlegalityopen.137conditionalpairs/26groups/458pending;35model7legacy,358/26semantic,854/1346/48partialgraph unchanged. '
        'Next target remainingF27-F32 retainedmainfeaturegather/checkpointrowownership/full5120norm interplay. Reuseexistingv34row-owner/feature-consumer answers andprofile first;onlynewoperator-specificcoupling/legality/upper-work evidence meritsnewfindings. '
        'No remote/model/Torch/CANN/NPU/capture/replay/implementation;alllocaljobscomplete. Original2073/285598conditional9/1352pending849PMU16phase3136ambiguity0whole/noallmodulephysicalMFU/lowest unchanged;Padv120consumed,nopendingnewpermission. '
        'Do noteditsealedv130/rerunfinalizer. Versionedpostcheckauthority;CURRENT-POSTCHECKoverlayvalidonlymatchingRECOVERY. TaskACTIVE Loop004EVALUATING GoalACTIVE.')
    report='''本轮补齐F29–F32在draft SWA路径上的条件组合：上下文KV预计算与SP gather前移投影如何共同选择计算、通信和发布方式。没有运行服务器、模型、Torch/CANN/NPU、capture/replay或优化实现。

先复用固定profile中的104个context→Tile顺序前沿，精确定位312次draft SP gather和88次主上下文feature gather。初始“每次SP后都有query KV”的假设被第一组dummy记录否定，原脚本与错误已保留。补查得到264次proposal窗口各有一个BF16[M,5120]×[512,5120]的KV投影；48次dummy窗口则有来自dsa_v41.py(698)的正面ZerosLike记录，对应None-metadata清零分支。它们是BF16[20,5120]，不能拿来绑定原80次BOOL24字段。没有因缺少投影记录就猜测分支；每条dummy结论有源码栈/CID关联证据。

源码进一步限定了可共享内容：上下文来自三份target辅助状态经main projection和完整5120维norm；当前query来自本层mHC pre/input norm，两者数值不同。选中draft层的context和query投影表达式都使用该层self_attn.wkv/kv_norm，因此可以研究按行打包、共享weight tiles；不能把两份KV值当成公共子表达式。draft role明确是SWA-only；目标C1/C2的额外投影、FLOAT或quant/index平面仍属其他域。context原RoPE模块与query inplace partial RoPE也须保留各自位置、dtype、舍入及读者，不能因共同512维就合并数值边界。

88次主上下文feature gather仍保留，count=640×Nc、P=8、BF16。这里只把已经全局可用的normed context送到新的KV计算owner，不删除上游gather，也不假装获得完整main weight的免费复制。仍需绑定实际communicator成员/顺序、每层参数副本、row map和全量读者；相同group label或rankSize本身不足以证明这些条件。

在明确的新ABI模型中，每层保留Nc-s行context在各卡重复计算，另有s行按owner分发，query在SP owner投影后交换512维BF16平面。每sender的carrier内容为1024×(m_q+ceil(s/8))字节；原query hidden gather内容为10240×m_q字节。不超过原carrier内容的条件是s<=72m_q。检查了5类正面profile几何的全部19637个s选项，得到2460个计算行数/接口内容的非支配选项；这不是硬件性能Pareto曲线或最优s。

小例Nc24、query20、m_q3：全部context分发时，每层carrier为30KiB→6KiB；保留一次main feature gather并计入3个不同层的carrier后，是122880→49152字节。每卡每层的dense投影行数为44→6。query实际20行却需按8×3=24行发送，新增计算不能凭空忽略padding；也不能把变化概括成无条件8倍工作减少。

长例Nc6400、query10、m_q2：全部context分发时，carrier为20KiB→802KiB；加上保留的main gather和3层carrier，8253440→10655744字节，接口内容反而增加。与此同时dense投影行数6410→802。只分发144行context时，可维持原20KiB carrier，投影行数变为6276。此处是计算与通信的取舍，不能单凭小维度平面宣布整体更快；GEMM小M效率、weight工作集、Vector/Cube/link竞争、打包与通知成本尚未测定。

实际cache接收者是关键条件。若原始context已在所有相关卡可用，一行需由R个卡消费、其中k个卡各自计算，则新接口需要R-k份512维endpoint值交付；k=R可不新增KV传送。9330组接收者/计算者参考用例覆盖了这个计数关系。接收者local方案、全复制cache方案和部分分发方案属于不同图，profile没有记录足以选择它们的真实cache需求集合。endpoint内容也不是网络wire下界。

联合打包还须保留每个重叠slot上的context→query写顺序。两次原调用分别把context值11和query值29写入同一slot7，原最终值为29；按rank排列打包后，一个合法串行顺序可能把context放到最后，得到11。这个有限反例不是native race或实际slot的发现，而是否定“同一批行可以任意scatter”的推断。可选方案必须保持role-aware写入、真实RAW/WAW范围、旧读者退役和noncausal query-KV就绪，不凭图增加全局barrier。

按层推迟context投影直到query输入ready，还会把context bank的最后读者从早期precompute延长到后面的draft层，影响跨迭代复用与队列容量。既有“3份context投影合成一个1536输出、分别做512 norm”和“按层C/Q row packing”是耦合的调度选择，不能同时领取两套完整launch/IO收益。所有数值都是指定表示的逻辑内容或当前/候选dense工作计数，不是HBM必要字节、HCCL wire流量、实测延迟、物理MFU或理论最低时间。

中央新增REL130-01，条件关系136→137、待核对459→458；HG85补入F32成员，联合组仍26且全部子集/运行时合法性未关闭。主候选35、历史框架记录7、语义358行/26模块、部分图854节点/1346边/48提示不变。原9类/1352次操作数字段、849个PMU缺口、16个phase未绑定图、3136项归属歧义仍保留，整模块关闭0。前置核验通过，整体目标继续。
'''
    inputs=[Path(r['path'])for r in read(O/'context-sp-inputs-v130.json')['inputs']]+[Path(r['path'])for r in read(O/'context-sp-profile-frontiers-v130.json')['databases']]
    updates={cid:dict(review_v130=dict(evidence=names,pair=['F29','F32'],relation_clause='REL130-01',
        same_weight_not_same_input_value=True,proposal_and_dummy_KV_domains_separated=True,
        retained_main_feature_gathers=88,hybrid_owner_vs_replicated_context=True,
        actual_cache_recipient_or_native_numeric_or_joint_fusion_proved=False,no_new_native_or_physical_gain=True))for cid in ['F29','F32']}
    updates['F32']['conclusion_changing_gap']='SelecteddraftSWA F29-F32 composition nowhaspositive264queryKV/48None-metadatazero branches and88retainedmainfeaturegathers. Actualcommunicatormembers/rankorder,CP/PCP/DCP/factory/draftclone/loadedobjects,weightreplicas/nativeM/numerics andcache-recipient/row/slot/bank/readergenerations remainunbound. TargetC1/C2 andotherallbranch/resource cases stayseparate. Current358semanticrows/26modules andconditional137pairs/26groups do notclosewholemodel/nativephysicalcapacity/MFU/lowesttime;9/1352 and849PMU/16phase/3136ambiguity remain.'
    persist_refinement(version=130,previous=129,title='Draft context and SP projected-KV joint owner/carrier analysis',
        summary=dict(filename='context-sp-summary-v130.json',previous_turn_classification='PROGRESS:v129F30-F31 generation/prefix relation;PASS6502',
            **detail,refined_candidates=2,refined_modules=3,branch_records=358,semantic_modules=26,relation_clauses=137,higher_order_groups=26,
            pending_central_pair_reconciliation=458,partial_graph_node_rows=854,partial_graph_edge_rows=1346,structural_pair_hints=48,
            scoped_reviewed_regimes=2073,scoped_reviewed_executions=285598,pending_regime_reconciliation=9,pending_execution_reconciliation=1352,
            model_operator_implementation=False,native_test=False,actual_MFU_numeric_physical_ceiling_or_speedup_proved=False,
            Pad_authorization='GRANTED_AND_CONSUMED',remote_job_status='Alllocalcachedprofile/source/arithmetic/verifier work complete;no remote/device/Torch/CANN/model/replay/serviceswork.'),
        changes=['Bind104context/draftSPwindows whilecorrectingall-SP-has-KVassumption:264proposalKV and48positiveNone-metadatazero branches;88mainfeaturegathers retained.',
            'ReconcileF29-F32 sharedweight/distinctvalues withreplicated/owner/hybridcontext,querypadding andrecipient-awareKVcontent;19637owner/9330recipient cases.',
            'Preserveper-slotC→Qorder,roleRoPE,localrawqueryreaders andextendedcontextbanklifetime;137centralpairs/26groups/458pending,allnative/wholegates open.'],
        candidate_updates=updates,module_updates={
            'M06':dict(candidates=[],finding='DraftqueryKV has264positiveBF16projectionwindows;48dummySPwindows insteadpositivelyreturnzero. Per-layerWKV/norm sharingisrowpackingwithnumerical/rowdomain guards,notvalueCSE.',evidence=names,gap=gap),
            'M19':dict(candidates=[],finding='ContextKV replication/recipient-owner/hybrid choices couplewithqueryproject-before-SP;main5120featuregather staysanddelayingcontext extendsbankreaderlifetime.',evidence=names,gap=gap),
            'M23':dict(candidates=[],finding='NewBF16-512carrier mayreduce orincrease sendercontent dependingcontext/queryratio ands. DummyNone-metadata,globalrecipient andlocalcacheowner domains requireseparate group/readiness contracts.',evidence=names,gap=gap)},
        remaining=remaining,action=action,
        jobs=[dict(kind='profile',status='error',description='Retaininitialselectorassumptionfailure:everydraftSPwindowwasexpectedtohavequeryKV;dummyNone-metadatazero branchrequiresseparatedomain.',artifacts=[O/'context-sp-query-assumption-error-v130a.json',N/'analyze_context_sp_profile_v130a.py']),
            dict(kind='profile',status='pass',description='104savedkeywindows/312SP,264proposalKV/48positiveNone-metadatazeros,88retainedmainAG;rawkeys/sourceframes verified,no remote/devicework.',artifacts=[N/'context-sp-local-run-v130.json',O/'context-sp-profile-frontiers-v130.json']+inputs),
            dict(kind='design-check',status='pass',description='19637owneroptions/2460countPareto/9330recipientcases,writerordercounterexample and'+str(pre['checks'])+'preflightchecksPASS;oneconditionalcentralpairadded.',artifacts=[O/'context-sp-preflight-v130.json',O/'context-sp-validation-history-v130.json']+[O/n for n in names]+[O/'coverage-current-obligations-v130.json',O/'whole-scope-requirement-audit-v130.json'])],
        report=report,read_first=names+['coverage-current-obligations-v130.json'],publication_rows={},gate_updates=dict(overall_complete=False),
        scripts=[N/'analyze_context_sp_profile_v130.py',N/'context_sp_models_v130.py',N/'analyze_context_sp_relation_v130.py',N/'verify_context_sp_relation_v130.py',N/'persist_profile_refinement_v90.py'],
        executor_prefix='context-sp',main_script=Path(__file__),source_read=True)


if __name__=='__main__':main()
