"""Persist selected-token→Engram history/hash composition without runtime promotion."""
from pathlib import Path
from copy import deepcopy
import hashlib
from finalize_origin_coverage_v9 import N,O,R,read,write
from persist_profile_refinement_v90 import persist_refinement

def main():
    assert not(O/'REVIEW-CHECKPOINT-v114.json').exists()
    assert read(O/'sample-hash-preflight-v114.json')['status']=='PASS'
    names=['sample-hash-inputs-v114.json','sample-hash-profile-scope-v114.json','sample-hash-source-contract-v114.json',
        'Engram-normal-INT64-range-proof-v114.json','Engram-history-factorization-v114.json','sample-hash-integer-validation-v114.json',
        'sample-hash-joint-work-and-capacity-v114.json','sample-hash-state-and-publication-v114.json','sample-hash-composition-v114.json']
    packet=read(O/names[-1]);clause=packet['clause'];group=packet['group']
    rel=deepcopy(read(O/'candidate-relation-clauses-v113.json'))
    assert all({r['a'],r['b']}!={'F24','F35'}for r in rel['clauses'])
    assert all(not {'F24','F31','F35'}<=set(g['members'])for g in rel['higher_order_groups'])
    rel['clauses'].append(deepcopy(clause));rel['higher_order_groups'].append(deepcopy(group))
    rel.update(parent_relation_index='candidate-relation-clauses-v113.json',clause_count=134,selected_pair_count=134,new_pair_clauses=1,new_higher_order_groups=1,overall_complete=False)
    write(O/'candidate-relation-clauses-v114.json',rel)
    cov=deepcopy(read(O/'candidate-relation-coverage-v113.json'));row=next(r for r in cov['pairs']if{r['a'],r['b']}=={'F24','F35'});assert row['clauses']==[]
    row.update(scope_state='SCOPED_OPERATOR_RELATION_REVIEWED_CONDITIONAL',clauses=[clause['id']],whole_pair_all_variants_closed=False,new_critical_path_gain_us=None)
    cov.update(parent='candidate-relation-coverage-v113.json',parent_relation_index='candidate-relation-clauses-v114.json',scoped_conditional_pairs=134,remaining_pair_relations=461,higher_order_groups=26,
        scope='134scopedconditionalpairs/26groups/35modelcandidates;461centralpairreconciliations andallvariantgates remain. NewF24/F35selected-token/historyhash relation requires actualnextcall/epoch/readerbinding.',
        sample_hash_review_v114=dict(evidence=names,new_pairs=1,new_groups=1),overall_complete=False)
    write(O/'candidate-relation-coverage-v114.json',cov)
    fps=deepcopy(read(O/'candidate-operator-footprints-v113.json'));ix={r['id']:r for r in read(O/'candidate-record-index-v113.json')['model_candidates']}
    for r in fps['candidates']:
        h=hashlib.sha256((R/'candidates'/(r['id']+'.json')).read_bytes()).hexdigest();assert h==ix[r['id']]['master_sha256'];r.update(captured_master_sha256=h,captured_master_version=113)
        if r['id']in ['F24','F31','F35']:r['sample_hash_review_v114']=dict(evidence=names,new_native_atoms=0,
            exactselected_history_packet_not_actualbinding=True,legacyCPUlabel_not_selectedhash_execution_resource=r['id']=='F24')
    fps.update(parent='candidate-operator-footprints-v113.json',overall_complete=False);write(O/'candidate-operator-footprints-v114.json',fps)
    nav=deepcopy(read(O/'candidate-pair-navigation-v113.json'));row=next(r for r in nav['pairs']if{r['a'],r['b']}=={'F24','F35'});assert row['conditional_clause_ids']==[]
    row.update(conditional_clause_ids=[clause['id']],semantic_pair_reconciliation_done_in_stated_scope=True,
        sample_hash_review_v114=dict(evidence=names,cross_iteration_owner_history_boundary=True))
    nav.update(parent='candidate-pair-navigation-v113.json',parent_footprints='candidate-operator-footprints-v114.json',parent_relation_index='candidate-relation-clauses-v114.json',scoped_conditional_pairs=134,remaining_central_pair_reconciliation=461,overall_complete=False)
    write(O/'candidate-pair-navigation-v114.json',nav)
    native=deepcopy(read(O/'operator-analysis-coverage-ledger-v113.json'));native['sample_hash_review_v114']=dict(evidence=names,
        existing_req_cache_hash_triples=88,existing_UVAlookups=176,new_native_regimes=0,new_native_executions=0,new_native_attribution_or_phase_promotions=0,new_conditional_relation='REL114-01')
    write(O/'operator-analysis-coverage-ledger-v114.json',native)
    obligations=deepcopy(read(O/'coverage-current-obligations-v113.json'))
    for g in obligations['obligations']:
        if g['id']=='G10':g['remaining']=g['remaining'].replace('133 conditional','134 conditional').replace('25 joint groups','26 joint groups').replace('462 pairs','461 pairs')
        if g['id']in ['G01','G05','G06','G07','G08','G10']:
            g['sample_hash_review_v114']=dict(evidence=names,
                normal_constructor_INT64nonnegative_range_andexactH_factor_proved_in_domain=True,
                actual_parameters_history_epochs_nextcall_andcostcapacities_bound=False,whole_requirement_closed=False)
        if g['id']=='G13':g['sample_hash_review_v114']=dict(evidence=names,
            old_sharedhistory_modspecialization_cachewriteorder_answers_reused=True,
            constructor_numeric_range_subdomain_answered=True,
            do_not_repeat_genericoverflow_unknown_for_validnormal_constructor_domain=True,
            actualregistered_buffers_overrides_andhistory_lifetime_remain_questions=True)
    obligations.update(parent='coverage-current-obligations-v113.json',new_native_or_whole_closures=0,overall_complete=False)
    write(O/'coverage-current-obligations-v114.json',obligations)
    audit=deepcopy(read(O/'whole-scope-requirement-audit-v113.json'));audit.update(previous_goal_turn='PROGRESS:v113 rejected-slotWAW/backwardsmain-demand andREL113-01;postcheck PASS765',input_checkpoint=113,followup_checkpoint=114)
    for g in audit['requirements']:
        if g['id']=='CROSS_OPERATOR_LAYER_CARD_ITERATION':g.update(conditional_pair_relations=134,higher_order_groups=26,central_pairs_not_reconciled=461)
        if g['id']in ['MODEL_VERSION_HARDWARE_LOAD','THEORY_AND_MEASURED_COMPARISON']:g.setdefault('evidence',[]).extend(names)
    audit['sample_hash_review_v114']=dict(evidence=names,normal_numeric_domain_andconditional_crossiteration_composition=True,
        actual_fixedload_nextcall_or_capacity_evidence_missing=True,whole_requirement_closed=False)
    write(O/'whole-scope-requirement-audit-v114.json',audit)
    write(O/'PROFILE-CODE-QUESTION-GATE-v114.json',dict(status='EXISTING_88HASH_176UVA_PROFILE_THEN_ONLY_HASH_BUILDER_HISTORY_AND_COMMIT_CLAUSES',
        questions=read(O/names[0])['operator_questions'],
        new_remote_or_public_source_files=0,source_files='ThreeexistingSHA-boundoperatorfiles;priorF35commit/contracts reused',
        no_frameworkwide_research=True,no_Torch_NPU_model_capture_replay_or_target_implementation=True,
        original_Pad_probe='StillPREPARED_NOT_EXECUTED;PENDING_REPLY',overall_complete=False))
    write(O/'sample-hash-validation-history-v114.json',dict(events=[
        'v113predecessor revalidated PASS765. v30/v62alreadyestablish88hashtriples/176UVAlookups,sharedall-layerhash,oncepertokenreqsearch,historypriority,backendmergeworkaround andlookupformatcosts;reusednotnew.',
        'Threeexistingoperatorfiles pinned. Builderrange wasexamined onlybecause itchanges signed/overflow/remainder proof domain;no tokenizer reconstruction,frameworkaudit orserver/devicework.',
        'Normal1<=V<=2^31 constructor proof bounds mappedcode*oddmultiplier andallXORs to0..2^63-1. Runtimebufferoverride/unboundactualvalues stayoutsideproof;thisdoesnotprove integerunitcapacity orINT32tablelayout.',
        'ExactH packet is6INT64/48B perpositionhistoryepoch;currentDEADbranch usesconstantP. Source-reference12288cases/589824IDchecks andindependent12288factorcases/4001rangefixtures pass.',
        'Deadproposal XOR-out,candidatecommit,windowunknown/fallback,currentcachewriteepoch,rawsentinel andone-longest-accumulatorcounterexamples retained. No claimall48hashIDs areuninvertible.',
        'Beforepersistence two premises were madeexplicit:earlyHrequires historyindependentofunselectedID/cacheeffects;192Bhashpublication isadditionalwhenrawIDfanoutremainsneeded. PrefixHwork/48Bstores/mod48/lookupwaste/epochlifetimearepriced.',
        'FirstfinalizerinvocationfailedatPythonparse onaninvalidmetadata keyword;no scriptbody/shared/checkpointmutation ran. Keywordcorrectedbeforetheactualpersistence.',
        'Logicalcommittedsuffix optimization mustnotdelete originalphysicalslotcacheproposalwrites/rollbackprotocol. Allprefix/H/prefetchalternatives stayprivate untilactualID/owner/position choice andnextcallmatching.',
        'REL114-01 F24/F35plusHG114(F24/F31/F35) addonepair/onegroup:134/26/461. Native/semantic/attribution/whole/MFU counts unchanged;Padprobe remainsunexecutedpendingreply.'],overall_complete=False))
    gap=('Normalconstructor nonnegativeINT64range andexacthistoryfactorization areprovedonlyin declaredvalidmapped/cache/parameter domain. '
        'Actualregisteredarrays/overrides,resolvedlast3source/dead/cachewriteepoch,accepted/recovered/bonus/validoutput owner/nextposition plan,allold/future readers andcanonicalhashcacheprotocol remain unbound. '
        'An earlyH mustbeindependentoftheunselectedcurrentID/dead/cacheeffects. Same address orhashIDs do notprovidevalidH;deadproposalxor-out iswrong. '
        'PriceonegraphHcompute/retention/selection,48mods,rawIDsretained,192Badditionalhashtransport ifIDfanoutstillneeded,UVAunique/wastedlookupwork andallDP/TPpublication. '
        'Integer/UVA/communicationcapacity andcriticalpathunknown;noBF16capacitysubstitution,no nativepromotions/MFU/speedup/wholeclosure.')
    updates={cid:dict(review_v114=dict(evidence=names,existing_hash_triples=88,existing_UVAlookups=176,
        normal_constructor_INT64range_proof=True,history_packet_bytes=48,full_hash_ID_packet_bytes=192,
        source_factor_cases=12288,hash_ID_checks=589824,independent_factor_cases=12288,independent_range_cases=4001,
        new_relation='REL114-01',new_group='HG114-SELECT-HISTORY-HASH',actualruntime_parameters_history_andpublication_bound=False,
        selected_resource_scope='Selectednativehash/UVAlookup=NPUtasks;UVAinputtier=hostmapped;legacyprovisionalCPUtemplateisnotdenominator'if cid=='F24'else'Preserveactualowner/readers',
        new_device_work=False,implementation=False),conclusion_changing_gap=gap)for cid in ['F24','F31','F35']}
    modules={'M14':dict(candidates=['F24'],finding='Defaultconstructor validmapped-code domain provesnonnegativeINT64hashaccumulators.48Bhistorysummary canseparatecurrenttokenwork underexactresolvedhistoryepoch;actualparameters/UVA/nextcallbindingsstillunknown.',evidence=names,gap=gap),
        'M17':dict(candidates=['F35'],finding='OnlyactualvalidemittedID/prefix/owner selection canfeednextEngramhash. RawIDreaders/RNG/recoveryoutputs andcanonicalcachecommit remain;prefetch/Halternatives areprivateandextra-workpriced.',evidence=names,gap=gap)}
    remaining=[g['id']+': '+g['title']+'. '+g['remaining']for g in obligations['obligations']]
    remaining.append('Currentv114:REL114-01F24/F35+HG114selectedhistoryhash,134pairs/26groups/461pending. NormalconstructorINT64nonnegativeproof and48BH/192Bhashfactor reviewed;actualparameter/history/nextcallbinding/capacities unbound. OriginalPad44Bnativeprobe stillawaits explicitreply;notexecuted. All14wholeobligations open.')
    report='''本轮检查目标采样F35到下一轮Engram hash/lookup F24的跨迭代边界。复用旧profile中88组req-index/cache-write/hash和176次UVA lookup，未新增运行或执行归属。一次hash已覆盖两个Engram层、request搜索已每token一次，均是当前基线。旧F24模板的CPU资源标签不能用于所选native任务的资源计数：这些hash/lookup是加速器任务，lookup输入来自host-mapped存储；CPU、NPU、UVA和通信容量仍须分开。

定点核对默认构造器得到新的整数范围证明。令M=2^63-1，压缩词表大小为V，且1<=V<=2^31；默认multiplier由B=floor(floor(M/V)/2)、u∈[0,B)、m=2u+1产生。映射值和pad均在[0,V-1]，所以乘积不超过M；DEAD在乘前被替换为pad，非负63位乘积的XOR仍在[0,M]。正常cache初始为0，指定writer只写有效mapped code或DEAD，维持该不变量。因此，这个明确域内不需要额外的INT64溢出回绕假设，正除数的有符号/无符号取余一致。实际注册buffer、外部覆盖、无效raw ID和越界状态不在证明内，INT32最终表索引布局与整数硬件吞吐也未由此证明。

对一个确定位置、request、参数和已解析历史，可预备6个INT64历史前缀H，共48B：每层对最近1/2/3项，按源码的DEAD/负位置阻断规则作乘积XOR。当前token有效时，各层各ngram的R=(mapped_current*m0) XOR H；当前token被阻断时，R是全pad的参数常量。之后保持原来的prime取余、offset和INT32写入。最终48个hash ID共192B。12,288组源码整数参考、589,824个ID比较，以及独立12,288组因式分离参考和4,001个范围用例通过；这些均是CPU参考，没有原生编译或设备验证。

提前计算有严格条件：H必须与最终hash调用的in_batch/known-window/fallback解析结果相同，且在选择前不能依赖尚未确定的current ID、dead predicate或当前cache写入。否则只能准备逐候选的H，或保留后算。已有hash ID是取余后的INT32值，不能直接当H；若保留取余前R/H，需要新的输出、bank和生命周期接口。dead proposal把整段都替换为pad，不能通过简单XOR移除proposal项恢复alive recovery的历史。未知window里的-1会走fallback，含义也不同于已解析的DEAD。还保留了单个最长accumulator不能通用移位更新的反例。

采样后只应为真正有效的accepted/recovered/bonus输出选择历史与owner，物理输出中的-1占位和未接受proposal不能直接索引token_map。候选前缀H、hash或只读lookup预备均保持私有，不能提前提交canonical历史。原有slot-cache程序可能先写proposal、再按rollback覆盖；新的逻辑committed suffix接口不能据此删掉原物理状态协议。token-map/multiplier/prime/offset/pad、request/位置、窗口/slot-cache快照、DP/TP owner、COW页和lookup表/scale epoch均要匹配，原始ID的embedding、用户输出、penalty、logprob和其他读者继续保留。

理论工作量没有按选择后阶段单独报收益。已有H时，终态只需2次当前项乘法、6次XOR和48次取余，但H的构建、写入、筛选和保活都要计费。若独立准备D+1个接受前缀H，再选择一个，参考总量相对一次晚算完整hash增加6D次乘法及4D+4次非平凡XOR；48次取余仍在。D=5时H载荷288B/request，另有标签/映射和生命周期开销。已有当前alive hash可研究保留取余前状态，但要新增6次XOR及48B/保留位置的写入；dead/current与最后bonus位置需单独处理。没有可用slack或资源容量证据时，不能声称这些额外工作能被隐藏。

跨卡也存在选择：sampler owner计算后发布192B hash包，或保留实际4/8B raw ID fanout后由接收端计算。若原始ID仍被其他读者需要，192B是额外通信，不能替代掉它。48倍/24倍只是包载荷参考比。TP8旧布局中，一个token两层各3个local head的冷读逻辑载荷为BF16 3072B、INT8_GROUP32 1728B、MXFP8_GROUP32 1584B；应按真正不同的(layer,head,hashID,table/scaleepoch)冷key及预取浪费计费，而不是套HBM峰值或假设预取命中。整数、UVA、通信的处理能力上界和关键路径证据均未补齐。

新增REL114-01（F24/F35）与三方组HG114（F24/F31/F35）。当前134条条件关系、26个联合组、461对待核对；35项模型候选、353条语义记录/26暂定模块及全部原生/归属缺口不变。9类/1352次原生执行仍待补证，0个整模块关闭，全模块MFU与最低模块仍未确立。原生Pad小探针等待单独明确回复，本轮没有远程操作、模型、NPU计算、捕获、replay或目标实现修改。整体继续进行。
'''
    action=('继续原Goal，v114PROGRESS，先读sample-hash-summary/report/profile/source/rangeproof/Hfactor/validation/work/state/composition/current-obligations-v114；verify_sample_hash_v114须PASS。'
        'v113PASS765；新F24/F35跨迭代关系REL114-01及HG114(F24/F31/F35)，134/26/461。旧88hashtriples176lookup/shared2layer/reqonce/backendmergefailure/UVAformat是prioranswer。'
        '源码defaultmultiplier builder B=floor(floor(M/V)/2),odd m≤floor(M/V)-1，1≤V≤2^31有效code/pad+正常cachewriter域乘积/XOR在0..M，无溢出/负取余歧义；actualregisteredbuffers/override未bind，INT32layout与integercapacity不由此证明。'
        '6INT64 H48B/position-historyepoch，alive current R=x*m0 xorH，blockedcurrent用P常量，原48mods/192BID保留。12288sourcecases589824IDchecks+独立12288factor/4001rangecases。'
        'EarlyH必须resolvedhistory独立于未知currentID/dead/currentcachewrite且匹配真正baseline fallbackepoch。deadproposal不能简单xor-out，windowunknown-1不等DEAD，原slotcacheproposalwrites/rollback不得删。'
        'D+1独立H增加6Dmul/4D+4xor；D5载荷288B+tags。192Bhashpublication当rawIDfanout仍必需时是额外，不是替代；UVA按uniquehead/tableepoch和unusedprefetchwork计，不是HBM。LegacyF24CPUtemplate标签不代表selectedNPUhash/lookup资源。'
        '继续有证据未分析算子关系/分支，不重复normalrange或泛泛historyreuse；需要actualaccepted/validIDs→nextposition/owner/historybank绑定才能提升此case，不能把条件替代为当前固定负载已成立。'
        'v111新Pad44BBOOL20→24 padTrue async仍待明确reply，禁止run；v109最小probe已完成授权消耗。若新批准用v115独立记录，按v111idle/container/SHAguards一capture/noReplay/noModel/noopaqueGPUderef，不改完成v114。'
        '35/353/26、813/1297/46、2073/285598conditional、9/1352pending849PMU16phase3136ambiguity0whole/noallMFU保持。GoalACTIVE TaskACTIVE Loop004EVALUATING；不重跑已完成finalizer。')
    inputs=[Path(r['path'])for r in read(O/names[0])['inputs']]
    persist_refinement(version=114,previous=113,title='Selected-token/history Engram hash and lookup composition',
        summary=dict(filename='sample-hash-summary-v114.json',previous_turn_classification='PROGRESS:v113 rejected-slotdemand andREL113-01;postcheck PASS765',
            reused_hash_triples=88,reused_UVAlookups=176,normal_constructor_range_proof=True,history_packet_bytes=48,hash_ID_packet_bytes=192,
            source_reference_factor_cases=12288,source_hash_ID_checks=589824,independent_factor_cases=12288,independent_range_cases=4001,
            refined_candidates=3,refined_modules=2,branch_records=353,semantic_modules=26,new_pairs=1,new_groups=1,
            relation_clauses=134,higher_order_groups=26,pending_central_pair_reconciliation=461,
            partial_graph_node_rows=813,partial_graph_edge_rows=1297,structural_pair_hints=46,
            scoped_reviewed_regimes=2073,scoped_reviewed_executions=285598,pending_regime_reconciliation=9,pending_execution_reconciliation=1352,new_native_promotions=0,
            new_device_work=False,implementation=False,native_test=False,targeted_Pad_probe='PREPARED_NOT_EXECUTED;PENDING_ASYNC_REPLY',
            actual_MFU_numeric_physical_ceiling_or_speedup_proved=False,remote_job_status='AlllocalCPUsource/integer/referenceworkfinished;no remotejob ornewmodel/NPU/capture/replay. Padprobe unexecuted.'),
        changes=['Prove validdefaultconstructor Engramhash products/XOR nonnegativeINT64 domain;actualregisteredparameter andgenericoverridecases remain unbound.',
            'Derive exact48Bhistorysummary/current-ID factor andblockedconstant branch;retain sourcepriority/cachewriteepochs,invalidrewrite witnesses andall speculativework/publication costs.',
            'Add REL114-01F24/F35andHG114F24/F31/F35 forvalidnextcall ID/history/owner publication;no rawID/cacheprotocol deletion orUVA/physicalspeedup inference.'],
        candidate_updates=updates,module_updates=modules,remaining=remaining,action=action,
        jobs=[dict(kind='profile',status='pass',description='Reuse88exacthashtriples176UVAlookups andthreeSHA-boundoperatorsources;noactualnextiterationoperandbinding ornewnativepromotions.',artifacts=[O/n for n in names]+inputs),
            dict(kind='design-check',status='pass',description='DefaultINT64range,H/currentfactor12288cases589824IDchecks plusindependent12288factor/4001rangechecks,sixinvalidrewrites andfullprefix/transport/UVAworkcontracts.',artifacts=[O/'sample-hash-preflight-v114.json',O/'sample-hash-validation-history-v114.json',O/'whole-scope-requirement-audit-v114.json',O/'coverage-current-obligations-v114.json'])],
        report=report,read_first=names+['coverage-current-obligations-v114.json'],publication_rows={},gate_updates=dict(overall_complete=False),
        scripts=[N/'analyze_sample_hash_v114.py',N/'verify_sample_hash_v114.py',N/'persist_profile_refinement_v90.py'],
        executor_prefix='sample-hash',main_script=Path(__file__),source_read=True)

if __name__=='__main__':main()
