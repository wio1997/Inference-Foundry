from copy import deepcopy
from datetime import datetime,timezone
from collections import Counter
from finalize_origin_coverage_v9 import N,O,read,write
from finalize_Engram_whole_frontier_v30 import packet,graph
from persist_profile_refinement_v3 import persist_refinement


def main():
    assert not (O/'REVIEW-CHECKPOINT-v41.json').exists()
    stamp=datetime.now(timezone.utc).isoformat()
    scope=packet('M21-scope166-v41',False);union=packet('M21-union166-v41',True)
    gap=read(O/'M21-original-profile-gap-reconciliation-v41.json');audit=read(O/'M21-original-requirement-completion-audit-v41.json');sem=read(O/'M21-semantic-branch-inventory-v41.json')
    assert len(sem['branches'])==18 and len(sem['static_branch_sites'])==268 and audit['result']=='NOT_COMPLETE'
    assert len(audit['requirements'])==8 and not any(x['completion_proved']for x in audit['requirements'])
    assert union['result']['union_TASKs']==84297 and union['result']['overlap_TASKs']==0
    profilefile='M21-expanded-native-navigation-v41.json';initfile='H06-dummy-and-proposal-initialization-contract-v41.json';variantsfile='H06-M21-initialization-variants-v41.json';graphfile='M21-initialization-and-capture-boundary-graph-v41.json'
    cs=deepcopy(read(O/'M21-profile-chain-coverage-frontier-v40.json')['selected_chains'])
    for h in gap['outside_chains']:
        text='\n'.join(h['project_frames']);ctx='DUMMY_ANCESTRY'if'dummy_run'in text or'_dummy_run'in text else'DRAFT_PROPOSAL_ANCESTRY'if'/llm_base_proposer.py('in text else'OTHER_RETAINED'
        cs.append(dict(chain_key=h['chain_key'],TASKs=h['TASKs'],contexts=[ctx],ops=sorted({x['signature']['op']for x in h['typed_shapes']}),leaf_project_frames=h['project_frames'][:3],state=h['review_state'],evidence=h['evidence'],new_to_v37_selector=True))
    assert len(cs)==209 and len({c['chain_key']for c in cs})==209
    write(O/profilefile,dict(generated_utc=stamp,selected_chains=cs,exact_TASK_union={k:v for k,v in union['result'].items()if k!='additional_TASK_keys'},
        state_counts=dict(Counter(c['state']for c in cs)),previous_selector_chains=186,expanded_chains=209,
        outside_scope_categories=gap['categories'],scope='Currentnativeentrynavigationexpandedfrommetadata/inputselectorusingoriginalM21entryhints;notallhost/process/graph/branchuniverse.',
        completion_audit='M21-original-requirement-completion-audit-v41.json',whole_module_closed=False,overall_complete=False))
    write(O/initfile,dict(generated_utc=stamp,framework_candidates=['H06'],related=['F31','H02','F30'],
        profile=dict(newly_point_reviewed_chains=9,TASKs=592,dummy_token_indices_zero_calls=16,dummy_query_slot_PAD_calls=16,dummy_context_slot_PAD_calls=16,
            proposal_slot_tail_calls=88,proposal_seq_tail_calls=88,proposal_QSL_tail_calls=88,proposal_token_index_tail_calls=88,fallback_QSL_last_Sub=16,used_update_positions_Index_TASKs=176,
            positive_fill_extents=dict(dummy_query_slots=20,dummy_context_slots=8192,dummy_token_indices=8192,proposal_slot_tail=[8188,8198],proposal_seq_tail=[8196,8198],proposal_QSL_tail=[8195,8197],proposal_token_indices_tail=[8172,8182])),
        facts=[
            'DummyfirstsyncsmetadataacrossDPandusesresultingnum_input_tokens. Itzeroestoken_indices_to_sample,thenfillsallpergroupquery/contextslotbankswith-1,beforeprofilemodelpathor_runnable. Capture/profilemodesareexplicit;unused_returnisnotnodataeffect/nocollectives.',
            'Dummyqueriesandcontextslotsareindependentdictbuffers/extent;do notinferphysicalgraphreadextentfromlogicalB orprofiledtype/shapealone. PADpreventsundesiredcachewritesafterarealproposalmayhavepopulatedvalidslots.',
            'Proposalpublishesgroup0slottail-1,seqtail0,query_start_loc_group0tail0andlatertoken-index-tail0. ThisQSLsentinel0differsfromworkerQSLtail-1(v39). Differentbanks/casesmustnotbemergedbydtype/shape.',
            'Proposalclearsthreeearlytailfieldsaroundprefixpublication;finaltoken-index-tailclearhappenslaterat1343. Acombinedproducerrequiresallrelevantprefixcopiesreadyandnoconsumerbeforethejointclear;donotmove1343earlywithoutitsownfirst/last-readerproof.',
            'Dummyzero-indexselectioncanstilldriveMarkov/decoderandcollectives. Thisauditdoesnotelidethosepasses. M20exactchainreusecovers4472TASKofstepwork;M22mustbindactualgraph/replayobservability.',
            'Scopednewdirectionsareoneproducerinitializingmultipleoriginalbanksormaintainingdirtyrangesunderepochproof. Originaladdresses,perstep/pergroupowners,sentinels,DPcoordinationandallcaptureconsumersremain.'],
        source='M21-positive-outside-selector-initialization-source-v41.json',existing_proofs=['H06-initialization-write-ownership-review-v39.json','M21-slot-rejection-state-and-packing-contract-v38.json','M20-Markov-confidence-and-verification-contract-v36.json'],
        implementation=False,critical_path_gain_us=None,whole_module_closed=False,overall_complete=False))
    variants=[
        dict(id='H06-v41-DUMMY-MULTIBANK-INIT',parent='H06',transformation='Combinecompatibletoken-indexzeroandpergroupquery/contextPADinitializationintooneproducerwithoutchangingbankaddresses.',
             conditions='Keepdifferent0/-1values,perbufferdtype/capacity/allgroupsandDPsync;allcapture/fullbankreadersmustseereadydata. Noassumedinterbufferaliasorpermissiontoskipdummycollectives.',implemented=False,native_validation=False,critical_path_gain_us=None),
        dict(id='H06-v41-PROPOSAL-TAIL-INIT',parent='H06',transformation='Combinecompatibleearlyproposal-tailinitializersorapplygeneration-awareinactive-rangeupdates.',
             conditions='Slot-1/seq0/proposal-QSL0differentbanks;prefixpublicationsmustprecedejointproducer. Keepindependentstepstateandlate1343index-tailuntilseparateproof. Refinesv39-v40dirtyrange,notadditivegain.',implemented=False,native_validation=False,critical_path_gain_us=None)]
    cond=[
        ('SCOPE','23outsidechainsare8264distinctTASKonexactrowidunion,notanewrun/newfunctionalM21resourcebucket.'),
        ('OWNERS','PerKVgroupquery/contextslotbanksandperstepgroup0banksretaindistinctaddresses,capacityandalllogicalowners.'),
        ('VALUES','Dummy/proposalslotsuse-1;seq/token-indices/proposalQSLtailsuse0;workerQSLtail-1isdifferent.'),
        ('DTYPE','Preserveeachactualbufferdtype/stride/outputconversion;currentpositiveFillI32notgenericall-layoutproof.'),
        ('PREFIX','Allprefixcopiesandinputsreadybeforecombinedclear,andnoearlierconsumerofanyfield.'),
        ('ALIAS','Nooverlappingbankswithconflictingfillvalues/orderunlessnewproducerexplicitlypreservesoriginalstoreordering.'),
        ('GROUPS','Visitallactualdictgroupsandbothquery/contextclasses;onepositiveprofilegroupcountnotuniversalconfiguration.'),
        ('CAPACITY','Keepfullbank/capture/DP-paddedreadextents,notonlycurrentlogicalquerycount.'),
        ('DPSYNC','Dummy_sync_metadata_across_dpandrequiredcollectiveparticipationremain;unused_returnnotpermissionforlocalearlyexit.'),
        ('CAPTURE','Retainprofile/capture/warmupallocation/parameterbank/frontiersemantics;actualgraphbodyauditstillpending.'),
        ('STATE','Arealproposalcanmakedummybanksnon-PAD. Cachehit/skiprequiresbanklast-writergenerationattestation.'),
        ('DIRTY','Initialfullsentinelclear/allwriterfootprints/shrink-grow/alias/resettrackingfromv39retained.'),
        ('LATE','Token-index-tail1343isnotautomaticallymovableto1212-1220initialization;preserveitsread/writeepoch.'),
        ('STEP','Independentstepbuffers/seed/count-1/reject/CPownershipcontractsretained;equalshapeisnotstepidentity.'),
        ('READERS','Allreplay/forward/cachewrite/Markovfirstandlastreadersandbufferreusefencesretained.'),
        ('PHASE','OriginalEP3IndexFill[3,1]notcurrent166condition;currentemptyindexcallsdonotdisablegenericnonemptybranch.'),
        ('OVERLAP','Multibankinit/dirtyrange/fusedslotPAD/sourceproducerfusionoverlapoldH06/F31variants;donotsumtheirbenefits.'),
        ('COST','Fillcounts/shapes/logicalpayloadarenotmeasuredHBMtraffic,criticalpathorattainablegain.')]
    write(O/variantsfile,dict(generated_utc=stamp,variants=variants,conditions=[dict(id='M21-v41-'+k,requirement=q,actual_current_legality_proved=False)for k,q in cond],implementation=False,whole_module_closed=False,overall_complete=False))
    g=graph(['DP_metadata_sync','prefix_and_old_state_ready','original_dummy_or_proposal_banks','conditional_joint_init','actual_eager_or_captured_consumers','all_last_readers','bank_reuse'],[
        ('DP_metadata_sync','prefix_and_old_state_ready','KEEP_ALL_RANK_PARTICIPATION'),('prefix_and_old_state_ready','conditional_joint_init','NO_READ_BEFORE_WRITE'),('original_dummy_or_proposal_banks','conditional_joint_init','SAME_BANKS_DIFFERENT_SENTINELS'),('conditional_joint_init','actual_eager_or_captured_consumers','PUBLISH_ALL_FIELDS_READY'),('actual_eager_or_captured_consumers','all_last_readers','CACHE_MARKOV_CAPTURE_OBSERVERS'),('all_last_readers','bank_reuse','EPOCH_FENCE')])
    write(O/graphfile,dict(generated_utc=stamp,conditional_initialization=g,graphs_acyclic=True,note='Aconditionaldata/protocolgraph,notaproventiming/alias/capturegraph.',implementation=False,whole_module_closed=False,overall_complete=False))
    evidence=['M21-original-scope-and-static-branch-index-v41.json','M21-original-static-index-hash-validation-v41.json','M21-positive-outside-selector-initialization-source-v41.json','M21-original-profile-gap-reconciliation-v41.json','M21-semantic-branch-inventory-v41.json','M21-original-requirement-completion-audit-v41.json',profilefile,initfile,variantsfile,graphfile]
    common=dict(scope_audit=str(O/'M21-original-requirement-completion-audit-v41.json'),semantic_branches=str(O/'M21-semantic-branch-inventory-v41.json'),outside_profile=str(O/'M21-original-profile-gap-reconciliation-v41.json'),initialization=str(O/initfile),conditions=str(O/variantsfile),implementation=False,new_performance_result=False)
    candidate_updates={'F31':dict(review_v41=dict(common,relation='Originalmoduleauditextends186nativechainselectorwithpositiveoutsideentries;formal8gatesremainpartial,insteadofrepeatinglocalfieldquestions.'),profile_review_state='SEMANTIC_MODULE_SCOPE_AUDIT;209CHAIN_NAVIGATION;EXPLICIT_DISPATCH_GUARD_CAPTURE_RESOURCE_GAPS',
        conclusion_changing_gap='FrozenconfigurationPCP/packed/groupdispatch,adjacentimplementationguardcrosswalk,actualgraph/parameterbank/allreaders andresourceboundsremain. Selectorcoveragealoneisnotfullmoduleproof.',
        next_profile_or_contract_action='Bindfixedconfigurationdispatchandneighbor-specificguardrecordsusingexistingcaptures;M22graph/bank/QLIbodyauditnext. Usev3semanticbranchpreservingpersistence.',all_consumers_closed=False),
        'F30':dict(review_v41=dict(common,relation='10outsideM21entrychains4472TASKexactlymatchM20v36Markov/stepbufferreview;notnewworkornewgain. Dummyinitmustretainstepbank/readiness/collectives.')),
        'F20':dict(review_v41=dict(common,relation='4outsidechains3200TASKareM12prepare/padding/IDABI alreadyreviewedv29;sourceassociationexpanded,notresourceassignmentornewM21savings.'))}
    fw_updates={'H06':dict(review_v41=dict(common,variants=variants),variants_v41=variants,implementation=False,new_performance_result=False),
        'H02':dict(review_v41=dict(common,relation='Host13field/globalcopy/scalar/asyncpublicationreviewretained;finitescopeauditstillneedscompleteprocess/queue/capturebindings.'))}
    module_updates={
        'M21':dict(finding='Originalscopeaudit:18stable semantic families replace20chronologicalrows;268existingstaticbranchsitesassignedbutnotautoproved.209nativechains/84297TASKnavigationincludes23outside/8264distinctTASK;9newinit592TASKpointreviewed.8originalgatesremainpartial/incomplete.',evidence=evidence,candidates=['F31','F30','F20'],gap='S01dispatch46sites/S13neighbor63guardmatch/S14PCP42casebinding plusS15graph/lifetime/S16bounds/S18processscope genuinelyremain. Assignmentisnotcompletion.',scope_audit=str(O/'M21-original-requirement-completion-audit-v41.json'),profile_scope=str(O/'M21-original-profile-gap-reconciliation-v41.json')),
        'M18':dict(finding='PositiveDSparkdummy16query/contextPAD/zero-indexinitializationandproposal-tail592TASKnowpointbound;keepallgroupbanks/DPsync/captureeffect.',evidence=evidence,candidates=['F31','F30'],gap='Actualcapturedreaderextent/aliasepochandnewinitnativecostremainconditions;M22actualbodyauditneeded.'),
        'M20':dict(finding='OriginalM21entrygapcontains10exactM20v36chains4472TASK;linksexistingstep/logit/argmax/copycontractswithoutdoublecounting.',evidence=evidence,candidates=['F30'],gap='OriginalM20finitephase/branch/observerauditisstillseparatefromM21navigation.'),
        'M22':dict(finding='Newdummyinit/sourceauditshowstoken-index0/query-contextPADandindependentstepbankreaders;actualgraphbody/replaybindingsremainunproven.',evidence=evidence,candidates=['F31'],gap='Needactualgraph/parameterbanks/first-lastreader/mode/16unboundgraphaudit;localsourceprotocolnotwholecaptureclosure.')}
    old_remaining=deepcopy(read(O/'module-coverage-gaps-v40.json')['remaining'])
    removed=[s for s in old_remaining if s.startswith(('M21v37','M21v38','M21v39','M21v40','NextM21'))]
    write(O/'M21-remaining-history-before-v41.json',dict(previous_version=40,retired_progress_paragraphs=removed,reason='PriorM21progressparagraphsarehistory,notactivependingrequirements;formal8-gateauditand18semanticfamiliesreplace them.',overall_complete=False))
    remaining=[s for s in old_remaining if s not in removed]
    remaining.insert(16,'M21v41currentfiniteaudit:original3files268staticbranchsites→18semanticfamilies;20chronologicalmatrixrowsarchived,notactivebranches. Current209chain/84297TASK exactnavigation extends186by23outside8264distinctTASK;10Markov4472exactv36/4M12prepare3200sourcev29/9newdummy-proposalinit592pointreviewed. 8originalmandatorygatesNOT_COMPLETE;0unassignedoldselectorisnotwholemodule.')
    remaining.insert(17,'NextM21S01fixeddispatch/groupcapacity(46sites)andS14PCPcasebinding(42sites)fromfrozencapturedconfiguration;S13adjacentM07-M11/M19guardreconciliation(63sites)useexistingproofs,notwholefileaudit. ThenM22S15actualgraph/QLIbody/parameterbank/first-lastreader/reuse/modes/16unboundgraphs;S16criticalsame-denominatorbounds andS18completeprocess/transportwithM23-M25/global26moduleaudit. Localconditionalimplementationnotrequiredforopportunityanalysis. Usepersist_profile_refinement_v3 tokeepprogresshistoryoutofsemanticbranches.')
    summary=dict(filename='M21-original-scope-audit-summary-v41.json',previous_turn_classification='PROGRESS:v40fieldreconciliation',original_static_sites=268,semantic_branch_families=18,original_chronological_rows=20,original_mandatory_gates=8,completion_audit='NOT_COMPLETE',
        expanded_native_chains=209,expanded_native_TASKs=84297,expanded_native_APIs=64476,outside_selector_chains=23,outside_selector_distinct_TASKs=8264,outside_categories=gap['categories'],newly_point_reviewed_chains=9,newly_point_reviewed_TASKs=592,
        conditional_variants=len(variants),condition_rows=len(cond),refined_model_candidates=[],related_model_candidates=['F31','F30','F20'],refined_framework_candidates=['H06'],related_framework_candidates=['H02'],new_model_candidates=[],
        remote_jobs=2,remote_pass=2,remote_retained_errors=0,
        remote_job_status='2saved-profileCPUjobs terminal/pass:original36entrynativecensusandexactrowidunion. SameSHAfrozenpositivegapreviewandmainsemantic8-gateauditpass. No model/torch/NPU/service/new-health work.',
        evidence_refs=[dict(receipt=str(N/(k+'.json')),full_result=p['full_result'],sha256=p['result_sha256'])for k,p in [('M21-scope166-v41',scope),('M21-union166-v41',union)]],no_new_model_NPU_service_or_health_work=True)
    report=f'''本轮对照原始M21范围审计，而非继续把报告版本当条件分支。旧矩阵20行大多是历次进展，其中“slot未分析”“108partial”等已被后续证据替代。本轮将旧行完整归档，改为18个稳定语义范围；原3文件268个已有静态branch-site均有家族/责任模块/证据导航，但没有因分配完成就自动标记语义闭合。未重新生成仓库AST或扫描全部源码。

审计找到了真实selector盲区：原36个EP3入口在当前166全量native资料中，有23链落在v37的186链selector之外。按(database_index,task_record_rowid)核对后，外部8264TASK与原76033TASK无重叠，当前导航并集为209链/84297TASK、64476API。它不是M21功能资源bucket。10链4472TASK与M20v36的Markov/stepbuffer chain ID完全匹配；4链3200TASK属于已审v29M12 prepare/pad/ID接口。复用这些证据，不重复宣称新发现或收益。

另9链592TASK补了定点语义：dummy的16次采样索引清0、16次query slot PAD和16次context slot PAD；proposal四种tail fill各88次，以及fallback QSL-last与位置gather。dummy先做DP metadata同步，清完整组bank后才进入profile/normal runnable；未使用返回值不证明没有cache状态/图体/通信效果。query/context容量不同，不能按当前B推断所有capture读者。

初始化值域也分开保留：slot用−1，seq/token-index与proposal QSL tail用0，worker QSL tail则用−1。三个早期tail可研究在prefix发布后合并producer；1343的后期index-tail不能无证明提前移动。记录两项H06条件细化：多bank dummy初始化、proposal tail联合初始化/dirty范围维护，{len(cond)}条约束。保持原地址、每组/每步所有权、sentinel、旧读先于覆盖和DP参与，不宣称所有Fill可删或已测加速。

原始8个mandatory gates逐项审计结果仍为PARTIAL_PROOF或INCOMPLETE_WORK，M21不关闭。明确剩余：S01 constructor/backend/group dispatch46个site的固定配置绑定；S14 PCP42个site的相关性/排除证明；S13邻接attention/index/cache实现63个site与M07-M11/M19证据的逐guard对照；S15实际graph/QLI body/参数bank/first-last reader与16未绑定图；S16关键同分母资源上界；S18完整进程/队列/回传边界。条件优化未实现/未测速本身不被当成未分析缺口。

原IndexFill[3,1]是旧EP3入口提示，未混入当前166执行数；当前88次空index与v36通用discard数学合同分别保留。历史进展段也从当前M21 pending清单归档，避免已回答问题反复成为下一步。

2次远端保存profile CPU作业通过，本地定点review与正式审计通过，共4runs全部pass。新persist v3只支持把语义分支与进展历史分开，原helper/v2保持不变。35模型候选、7框架候选、0整模块关闭，26理论比较INCOMPLETE、3136candidate资源行、849缺PMU、16phase-unbound图保持。无新模型/torch/NPU/服务/健康工作。下一步补固定配置/邻接guard证据后推进M22-M25和全26模块完成审计，Goal active。
'''
    jobs=[
        dict(kind='profile',status='pass',description='Pinnedall32nativefullchain/TASKidentityscancompares36originalM21entryhintswith186selector;23outsidechains8264TASKlocated.',artifacts=[N/'query_M21_original_entrypoint_gap_v41_166.py',N/'M21-scope166-v41.json']),
        dict(kind='profile',status='pass',description='Exact(database_index,task_record_rowid)/APIrowidunionoftwoSHA-pinnedcensuses:0overlap,84297TASK64476API209chainnavigation.',artifacts=[N/'query_M21_scope_union_v41_166.py',N/'M21-union166-v41.json']),
        dict(kind='review',status='pass',description='Positive9newinitializationinterfaces592TASKsameSHApointreview;10exactM20chainreuse and4M12sourceABIreuse;all268existingbranch-indexSHAchecked.',artifacts=[O/'M21-positive-outside-selector-initialization-source-v41.json',O/'M21-original-static-index-hash-validation-v41.json',N/'M20-profile166-v36.json',N/'F04-MC2-alignment166-v29.json']),
        dict(kind='review',status='pass',description='Formal8originalrequirementaudit/18semanticfamilyinventorywith268navigationassignments,2conditionalinitvariants/18conditions;noautomaticcompletion.',artifacts=[O/q for q in evidence]+[N/'build_M21_semantic_scope_audit_v41.py',N/'persist_profile_refinement_v3.py'])]
    action=('继续Loop004，v41是原scope实质审计。先读M21-original-scope-audit-summary-v41.json、M21-original-requirement-completion-audit-v41.json、M21-semantic-branch-inventory-v41.json、M21-original-profile-gap-reconciliation-v41.json、H06-dummy-and-proposal-initialization-contract-v41.json、module-coverage-gaps-v41.json。'
        '原M21范围3文件268staticbranchsites、8mandatorygates；20chronologicalrows已archive改18稳定semanticfamilies。268只是导航分配不是语义proof，M21仍open。Usepersist_profile_refinement_v3保semanticbranches，别用v2再次把每轮报告追加成branch；原helpers/v2未改。'
        '36原EP3入口current166native gap=23outsidechains/4072API/8264TASK，exactrowidunion old76033+8264 overlap0→84297TASK，64476API209chains，不重写v20M21功能76943+80/529missingPMU。10Markov4472 exactchain匹配v36已审；4M12prepare/pad/ID3200 sourceABI v29复用；9新init592已pointreview。'
        'Dummy361/363pergroupquery/contextfullPAD-1+392token_indices full0各16，DPsync前置后再profile/runnable，return未用不是无cache/comm；当前positiveextents query20/context8192/index8192不当全部captureextent。Proposal1212 slotTail-1/1216seqTail0/1220QSLTail0/1343laterIndexTail0各88，WorkerQSLTail-1另bank。可联合早3tail或dummy多bankinit需所有prefixready/addresses/owner/dtype/alias/disjoint/mask/lifecycles，late1343不盲提前；2H06variants18conditions refinev39/40。'
        '2remoteprofileCPUjob+localpointreview+mainaudit=4runs4pass；35model7framework0module，26theoryINCOMPLETE/3136资源/849missingPMU/16unbound图；无model/torch/NPU/service/health,lastv12post。'
        '下一步有限真缺口：S01 fixedconfiguration绑定backend/constructor/group dispatch46sites；S14 PCP42sites是否本case相关/排除的源cfg证据；S13邻接attention/index/cache实现63sites与现有M07-M11/M19合同逐guard对照，勿读整个DSA。再S15/M22实际80graph/16unbound/QLIbody/参数bank/inputready-lastreaderreuse/producer模式和queue-transport；S16同分母关键bounds，S18/M03/M25全进程回传，随后M23-M25/global26moduleaudit。旧IndexFill[3,1]不新当current166执行；已回答的v37-40M21 pending历史归档，别重读。Implementation未完成不自动等于analysisgap，Goalactive因上述原scope真未证。')
    persist_refinement(version=41,previous=40,title='v41 M21 original-scope and semantic branch audit',summary=summary,
        changes=['Originalscopeauditfinds23outsidechains8264distinctTASK;209chainexactnavigationincludes10M20reuse/4M12reuse/9newinit592TASK.',
            '20chronologicalM21rowsarchivedandreplacedby18stable semanticfamilies;268existingbranchsitesassignedbutnotautoproved,8originalgatesexplicitlyNOT_COMPLETE.',
            '2conditionalmultibank/tailinitrefinements18constraintsrecorded;fixeddispatch/PCP/neighborguard/graph/process/boundsremainrealfrontiers.'],
        candidate_updates=candidate_updates,framework_updates=fw_updates,module_updates=module_updates,semantic_branches={'M21':sem['branches']},source_read=True,
        remaining=remaining,action=action,jobs=jobs,report=report,read_first=['M21-original-requirement-completion-audit-v41.json','M21-semantic-branch-inventory-v41.json','M21-original-profile-gap-reconciliation-v41.json',initfile],
        publication_rows={'F31':'| F31 | Metadata、slot、draft共享与输入输出发布重组 | 条件 | v41对照原scope，将20条进展记录归档为18个语义范围，268静态site仅作导航。原selector外23链8264TASK经精确并集补入209链导航，9链592TASK初始化合同新补，其余复用M20/M12。8原mandatory gates仍partial，dispatch/PCP/邻接guard/capture-bank/资源与进程边界未全证，整体不关闭。 |'},
        gate_updates=dict(status='ORIGINAL_SCOPE_AND_POSITIVE_PROFILE_GAP_AUDITED',answered_by=evidence,closed_scope='Exactoriginalentryprofilenavigationgap/localdummy-proposalinit/semanticinventoryandformal8gateaudit.',not_closed='OriginaldispatchPCP/neighborguard/capturebank/allobserver/resource/processdomains explicitly open.',stop_rule='Do notconfuseselectororstaticindexassignmentwithwholemodulecompletion;advancefiniteopenfamiliesusingexistingcfg/neighborproofs.'),
        scripts=[N/'query_M21_original_entrypoint_gap_v41_166.py',N/'query_M21_scope_union_v41_166.py',N/'build_M21_semantic_scope_audit_v41.py'],executor_prefix='zcode-M21-',main_script=N/'finalize_M21_scope_audit_v41.py')


if __name__=='__main__':
    main()
