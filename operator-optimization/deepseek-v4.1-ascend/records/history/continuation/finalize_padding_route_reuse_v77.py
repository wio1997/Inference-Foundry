"""Persist a bounded operator question without converting it into goal completion."""
from copy import deepcopy
from pathlib import Path
import hashlib
from finalize_origin_coverage_v9 import N,O,read,write
from persist_profile_refinement_v5 import persist_refinement


def main():
    assert not(O/'REVIEW-CHECKPOINT-v77.json').exists()
    names=['padding-route-domain-v77.json','padding-route-semantic-proof-v77.json',
           'padding-route-cost-models-v77.json','padding-route-variants-v77.json',
           'padding-route-source-index-v77.json','padding-route-validation-history-v77.json',
           'candidate-relation-clauses-v77.json','candidate-relation-coverage-v77.json',
           'operator-analysis-coverage-ledger-v77.json','padding-route-branch-inventory-v77.json',
           'next-operator-proof-questions-v77.json']
    domain=read(O/names[0]);cost=read(O/names[2]);assert domain['known_MC2_router_pad_calls']==320
    bad=N/'analyze_padding_route_reuse_v77.before-pad-format-fix.txt'
    write(O/names[5],dict(errors=[dict(state='RESOLVED',phase='PREWRITE_SOURCE_ASSERTION',
        cause='Extractor asserted spelling F.pad whereas exact source uses nn.functional.pad. Source already showed the same intended operator; corrected exact whitespace-normalized clause,not semantics or authorization.',
        saved_pre_fix_script=str(bad),sha256=hashlib.sha256(bad.read_bytes()).hexdigest(),
        partial_artifact_writes=0,checkpoint_or_shared_state_mutations=0)],overall_complete=False))
    rel=deepcopy(read(O/'candidate-relation-clauses-v76.json'))
    r=next(r for r in rel['clauses']if r['id']=='REL73-07')
    assert(r['a'],r['b'])==('F20','F21')
    r['refinement_v77']=dict(operator_claim='Full raw P×k result preservation allows constant-padding route reuse without proving that dispatch ignores every padding output. Keep unchanged postmapping/shared/EPLB/RNG and all MC2 receiver/protocol work. This is distinct from pruning externally visible outputs or EP receiver work.',
        gates=['Native result/precision/tie and per-layer/rank generation','originalN/P provenance','full output publication before unchanged postprocessing','no double count with existing direct producer padding budget'],
        evidence=names[:5],critical_path_gain_us=None,all_runtime_gates_proved=False)
    r['evidence']+=names[:5]
    write(O/names[6],rel)
    cg=deepcopy(read(O/'candidate-relation-coverage-v76.json'))
    # Existing relation only: no artificial new edge or all-variant closure.
    cg.update(parent_coverage='candidate-relation-coverage-v76.json',padding_route_refinement_v77='REL73-07',overall_complete=False)
    write(O/names[7],cg)
    cov=deepcopy(read(O/'operator-analysis-coverage-ledger-v76.json'))
    cov.update(parent_coverage_ledger='operator-analysis-coverage-ledger-v76.json',
        new_pending_regimes_reconciled=0,new_pending_executions_reconciled=0,
        padding_route_refinement_v77=dict(explicit_pad_calls=320,complete_root_source_order_correspondences=320,native_pointer_bindings=0,
            profiles_reassigned=False,constant_output_variants=5),overall_complete=False)
    write(O/names[8],cov)
    old={m['id']:m for m in read(O/'module-phase-branch-matrix-v76.json')['modules']}
    branches={'M12':deepcopy(old['M12']['branches'])}
    details={
        'M12-OP-03':'Zero-logit row still follows FP32 Exp/+1/Ln/Sqrt,rankingbias,unbiasedgather,selectedsum+eps/Div/scaling. Explicit scalarbinary32 witnesses refute unconditional1/k or bias-onlysort shortcuts;not device numerical attestation.',
        'M12-OP-05':'Exact native-output template reuse avoids repeated nonlinear/fullsort on explicitpadding only. CurrentSortdoc stable input tie reference is not pinned CANN9.2 binaryproof;retain loaded tie/NaN/precision/generation gates.',
        'M12-OP-09':'Joint no-logitsPad+prefixconsumer+fulltailoutput reference saves2506832B against logitsPad+TopK;2457600B overlaps existing directproducer validprefixcopy budget,increment49232B only. Hidden/mask/collectiveP unchanged;no routerGEMM padding FLOP saving.',
        'M12-OP-10':'Known320routerpadowners match320M816TopK calls by complete same-thread execute_model root cardinality40 and interleaved sourceorder in8windows;not pointerdefuse,not all640M816. Per-use16*384 rowfunction elements,48Brawtemplate/layer. Source constantdomain supports fulloutputreuse,not outputdeletion.',
        'M12-OP-13':'RawP×k fulloutput must be materialized before logicalmapping/sharedappend/EPLB/fullP RNG. in_profile_run lifecycle flag doesnot mean Torchprofileron. Fullsenderoutputs preserved,EP receiver expert/collective worknotremoved.',
        'M12-OP-15':'Separatecold/warm domains:8rankwindows*40layers not repeatedlayercachehits. NativeM1build needs crossshapeequivalence;conservativesameP816build16J>816⇒J>51beforeoverhead. Firstpadsnapshot96Blogicalcopy acceleratesno firstcall. Persistenttail all-writerepochmustrestoreafterN816realwrites.'}
    for b in branches['M12']:
        if b['id']in details:
            b['evidence']+=names[:8];b['padding_route_review_v77']=details[b['id']]
        if b['id']=='M12-OP-10':
            b['branch']='Explicit zero-logit/ID0 MC2 alignment tail: existing masked-output pruning remains conditional; added exact native-result reuse with full raw outputs preserved before unchanged postprocessors.'
            b['state']='CONSTANT_DOMAIN_AND_FULL_OUTPUT_REUSE_ANALYZED_CONDITIONAL;NATIVE_GENERATION_GATES_OPEN'
    write(O/names[9],dict(modules=branches,retained_families={'M12':15},extended_families=sorted(details),protected_other_modules=25,whole_modules_closed=0,overall_complete=False))
    write(O/names[10],dict(completed_bounded_questions=[dict(id='TOPK-PAD-CONSTANT-ROUTE',state='CONDITIONAL_OPERATOR_ANALYSIS_REVIEWED',evidence=names[:5],implementation_required_for_conditional_analysis=False)],
        questions=[dict(id='ALL-MODULE-FIXED-CASE-GUARD-AUDIT',state='NEXT_ANALYSIS_NOT_COMPLETED',question='Reconcile existing26semanticmodule inventories against exact target/draft/phase/shape/optionalinput domains. Find genuinely unexamined operator branches,exclude inapplicable branches only with evidence,and prioritize conclusion-changing fusion/reorganization/typed-ceiling gaps rather than framework internals.',needed=['Existing module/operator ledgers first;do not reread broad framework repositories','Preserve legacyV41/CP distinction and graph16phaseunknown','For each open branch state whether new source question,native argument evidence,capacity upper bound,or alreadyconditionalanalysis—not just reportstatus']),
                   dict(id='PAIR-AND-HIGHER-ORDER-RECONCILIATION',state='OPEN_EXISTING_EVIDENCE_GAP',question='550 central pair slots may have older scoped evidence. Reconcile existing clauses and actual composition/alias/resource domains;blank index is neither no-fusion proof nor proof of no earlier analysis.')],
        unchanged_pending='7MemSet1272 +2BOOL80;missingPMU849/ambiguous3136/phase-unbound16;no anonymous-field reclassification.',overall_complete=False))
    write(O/'PROFILE-CODE-QUESTION-GATE-v77.json',dict(status='OPERATOR_CONSTANT_PADDING_OUTPUT_REUSE_AND_COST_DOMAIN',
        previous_turn_classification='PROGRESS:v76typedTopKguards/nativeIDABI/predicateproof',
        questions=['Can literalzero/ID0tail reuse exact raw nativeTopK results?','Where must fulloutputs be assembled to preserve mapping/RNG/EP consumers?','Does16-rowwork elimination lower the current48program longestrow loop?','What are cold/warm,copy/fill and existingpadding-budget interactions?'],
        allowed_source='Saved4472TopK descriptors/320explicitpadowners,exactruntimeAPIconnectionsets/rootannotations,and6alreadyfrozenoperatorclauses;onepublicSortAPIreferencewithversionlimit.',
        new_remote_source_read=False,new_source_files_obtained=0,local_operator_clause_read=True,overall_complete=False))
    gap='Conditional source-level padding-output reuse is analyzed. Native bit/result/tie/determinism,perlayer/rank textbias/scalar-generation and shape-equivalence conditions remain;exactloadedpayload not present. No measuredspeedup or physical Vector/Sort/nonlinear/store capacity upper bound. All othermodule/branch/candidate-composition and9/1352operandgaps remain open;implementation is outside scope.'
    updates={cid:dict(review_v77=dict(domain=str(O/names[0]),semantics=str(O/names[1]),costs=str(O/names[2]),variants=str(O/names[3]),
        source_index=str(O/names[4]),scope='Constant padding rawTopK result reuse,mandatoryoutputconsumerboundary and typed-ceiling conditions only.',conditional_operator_analysis_deliverable=True,implementation=False,native_test=False,new_performance_result=False))for cid in ['F20','F21']}
    updates['F20'].update(profile_review_state='CONSTANT_PADDING_FULL_OUTPUT_REUSE_REVIEWED_CONDITIONAL',conclusion_changing_gap=gap,
        next_profile_or_contract_action='This bounded source question is conditionallyreviewed;retain nativebit/generation/capacity gates. Advance26module fixedcaseguard audit and existingrelation/higherorder reconciliation rather than repeatedlydeepening this single tail.')
    updates['F21'].update(next_profile_or_contract_action='Keep fullrawPTopKoutputs before mapping/shared/EPLB/fullP RNG andunchangedMC2protocol. Continue actual receiver/resource/composition analysis;constant localsendertail is not receiverempty proof.')
    mods={'M12':dict(finding='Explicitzero/ID0tail supports conditionalnative-resultreusewithfulloutputs.48program prefixN800andP816bothmax17rowloops. Integratedreference1302144→1277488B,butnativeprefix+assembly1354288B. Jointpadding2506832Boverlapdedup,cold/warmcostandfullRNG/EPconsumerconditions retained.',
        evidence=names,gap=gap,candidates=['F20','F21'],profile_scope='Only pinned TopK norm/sort/weights,rowpartition,padding andrawoutputpostprocessing/receiverfrontier.')}
    remaining=list(read(O/'module-coverage-gaps-v76.json')['remaining'])
    remaining += ['v77 boundedTOPK-PAD-CONSTANT-ROUTE conditionalanalysis reviewed:320explicitpadcalls/320sourceordercorrespondences in8complete40layerroots,notpointerdefuseorcachehits;notall640M816. Rawfulloutputreuse avoidsneedtoignorepaddingoutputs,butnativeprecision/tie/parametergeneration+shape andallreader/capturepublicationremainconditions.',
        'v77 logicalTopKbytes1302144→integrated1277488(saving24656),separateassembly1354288(extra52144).48programmax17rowsunchangedafter816→800;no proportional2%latencyclaim. Jointno-logitsPadbudget2506832includesold2457600directproducerbudget,increment49232only. ActualtypedVector/Sort/nonlinear/storecapacities andrequestnumericboundsunknown.',
        'v77 conditionaloperatorquestiondone doesnotcloseF20/M12orGoal. Nextauditall26module fixedcasebranchapplicabilityandremainingfusion/reorganization/typedcapacityconditions;45pairclauses10groups/550centralreconciliationsremain.35model+7legacyframework/2073regimes285598executionsreviewed9/1352pending/global84916/3136/0wholeclosuresunchanged.']
    action=('继续原Goal，v77 PROGRESS。用户要求围绕算子融合/理论上限，profile优先，代码只回答具体算子语义/依赖/精度问题，不研究框架。'
        '先读padding-route-domain/semantic-proof/cost-models/variants/source-index-v77和next-operator-proof-questions-v77。当前paddingconstant问题已完成条件分析，不以未实施/未跑NPU为未分析。'
        '320MC2routerpadding800→816按8个同线程完整execute_model窗口各40pad/40TopK顺序对应，非pointerdefuse，640M816不能都算pad；没有同层多次cachehit证据。'
        'ID0textbias，norm2phi原生FP32排序和unbiasedweight顺序保留，不能替1/k或bias-onlysort；用exactnative结果模板48B/layer，40层1920B/rank仅payload。'
        '缓存位于rawinnerrouter输出边界，先写全P×k，再原样logicalmap/sharedappend/forceEPLB/全P RNG；profile_run不是profiler开关。保留所有MC2hidden/mask/跨卡receiver计算；N/Pguard和参数/模式/native/shape/输出所有writer代际必须明确。'
        '48核rowpartition816/800均max17，删16行不自动2%latency。TopK逻辑ref1302144→integrated1277488省24656B；existingnativeprefix+separateassemble1354288反多52144B，fixedrowmodel不优但不全域否定。jointno-logitsPad省2506832B含旧directproducer2457600，不能叠加，routerGEMM本来算800不省paddingFLOPs。'
        '模板首次/后续分开：M1nativebuild需crossshape精确性；sameP816build需J>51才正rowwork未含fill；firstnativepadresult复制96B不省firstcall；N816real会改persistenttail，下次N800必须allwriterepoch或恢复。'
        '下轮按nextquestion审计全部26模块固定case的target/draft/phase/shape/optional条件，把已有条件分析和真正未分析分支分开；优先影响结论的融合重组与typed物理capacity。并继续旧595pair中央汇总/多候选约束，未填不等于未研究或不能融合。'
        '本轮45clauses10groups/550centralpending不增边，M12保留15families富化6，25其他modules完全保留；2073/285598conditionalreview9/1352pending，849PMU/16phase/3136ambiguity，35model7legacyframework0wholemoduleclosures。全部模型/框架/设备/远程服务保持未执行，只本地savedprofile和source/maths。'
        '保留1resolvedprewriteERROR：source拼写F.pad应为nn.functional.pad，存pre-fixscriptsha；没有partial产物。3PASS。verify_padding_route_reuse_v77应PASS。Goalactive，报告成稿不complete。')
    report='''已完成对“零logits、ID 0的补齐行能否复用TopK结果”的条件分析。结论是可登记为保留完整输出的复用机会，仍受原生精度、参数代际和缓存发布条件约束；没有实施或测得提速。

320次明确的800→816 router padding，落在8个完整执行窗口，每窗口40次padding和40次TopK，按同线程源码顺序交替。这个对应不是指针级def-use证明，也不表示640次M816 TopK都有16条补齐行。8个窗口的40层调用不能证明同一层后续缓存命中。

ID0选择text bias。固定源码对零行执行FP32 Exp、加1、Ln、Sqrt，然后加bias排名，取未加bias的值进行选中项求和、加eps、除法和缩放。离线binary32反例说明，不能直接把结果替成1/k，或只排序bias。本候选复用原生输出位模式，不能用CPU公式生成的近似值冒充原生结果。当前[Ascend Sort文档](https://asc.gitcode.com/api/SIMD-API/adv_api/sort_operations/Sort.html)描述相同分数按输入索引先后排序，但页面未绑定已安装CANN9.2版本，因此原生tie行为仍是条件。

缓存应位于原始TopK输出边界：先完整写出816×6的ID和权重，再保持逻辑到物理映射、shared append、EPLB、完整形状的随机数消耗与MC2协议。这样不依赖“后续忽略所有padding输出”的证明，也不能借此删除其他卡接收的专家工作。生命周期标记in_profile_run不等于开启Torch profiler。

| 同一TopK边界的逻辑读写参考 | 字节/调用 |
|---|---:|
| 原816行TopK | 1,302,144 |
| 集成800有效行计算＋缓存尾部完整写回 | 1,277,488 |
| 现有800行原生调用＋单独拼接输出 | 1,354,288 |

后一种路线比原路径多52,144B。集成路线的24,656B预算、额外拼接及缓存成本均未换算为物理HBM或请求延迟。若连同logits padding一起消除，合计参考预算2,506,832B，其中2,457,600B已属于已有direct producer方案，不能重复累计。Router GEMM原本只算800行，没有16条padding GEMM FLOP可再省。

按固定源码48程序分工，816行和800行的最长程序都处理17行。减少16行总工作，不自动使TopK延迟下界降低16/816。真实收益还依赖资源竞争、调度与新增写回/发布成本；Vector、Sort、非线性及store的数值容量上限仍未知。

每层缓存6个FLOAT权重与6个INT32 ID的有效载荷为48B，40层为1,920B/卡。冷启动另计：保守地用同形状816行建立模板，仅行工作摊销就需超过51次复用；从首次补齐输出复制模板不会加速首次调用。单行原生建立模板需证明跨shape一致。持久输出银行还必须追踪全部写者：一次真实816行会覆盖旧尾部，随后800行不能凭地址相同直接复用。

保留35项模型候选和7项历史框架候选。当前2,073类、285,598次执行已有条件分析，9类、1,352次关键操作数证据待补；整模块闭环仍为0。下一步回到全部26模块的固定负载分支适用性、融合重组组合及理论容量证据审计。Goal继续进行中。
'''
    src=read(O/names[4])
    jobs=[dict(kind='review',status='error',description='Resolvedprewriteextractorassertion:F.padspellingversusactualnn.functional.pad. Exactsourceclausecorrected;noanalysisartifact/sharedstatewritesbeforefailure.',artifacts=[O/names[5],bad]),
        dict(kind='profile',status='pass',description='320explicitMC2routerpadowners/320TopKnativeexactAPIowners grouped8same-threadcomplete40layerrootsandinterleavedsourceorder;notpointerdefuse/cachehits,notall640M816. No functionalassignmentchanges.',artifacts=[O/names[0],N/'analyze_padding_route_reuse_v77.py']),
        dict(kind='design-check',status='pass',description='Sixpinnedoperatorclauses,nativeconstant-outputreuseboundary,FP32counterwitnesses,48program17→17rowmodel,fivevariants,coldwarmpublicationandnonadditivebytebudgets. No native/device/implementation work.',artifacts=[O/names[i]for i in[1,2,3,4]]+[Path(r['path'])for r in src['source_files']]),
        dict(kind='review',status='pass',description='Retain45relations10groups/550centralpending;M12families15enrich6and25othermodulesidentical. Profile2073/285598review9/1352pending/global84916/3136/35model7framework0wholeclosures. Nextallmoduleguardquestionexplicit.',artifacts=[O/names[i]for i in[6,7,8,9,10]])]
    persist_refinement(version=77,previous=76,title='Constant padding TopK native-result reuse, full outputs and cost ceilings',
        summary=dict(filename='padding-route-ceiling-summary-v77.json',previous_turn_classification='PROGRESS:v76typedTopKguards/nativeIDABI/predicateproof',
            explicit_padding_calls=320,source_order_correspondences=320,pointer_bindings=0,native_result_values_obtained=False,
            original_TopK_reference_bytes=cost['original_TopK_reference_bytes'],integrated_reference_budget_bytes=24656,separate_assembly_extra_bytes=52144,
            fixed_program_longest_row_count_before_after=[17,17],joint_reference_budget_bytes=2506832,prior_direct_producer_budget_overlap=2457600,
            operator_variants_or_exclusions=5,conditional_question_reviewed=True,source_files_reused=6,new_source_files_obtained=0,new_remote_source_read=False,
            retained_semantic_families=15,enriched_semantic_families=6,protected_other_modules=25,relation_clauses=45,higher_order_groups=10,pending_central_pair_reconciliation=550,
            scoped_reviewed_regimes=2073,scoped_reviewed_executions=285598,pending_regime_reconciliation=9,pending_execution_reconciliation=1352,
            implementation=False,native_test=False,new_performance_result=False,actual_physical_numeric_ceilings_proved=False,
            remote_job_status='No remote/server/model/Torch/NPU/capture/service/process work;local saved-profile/source/math only.'),
        changes=['Conditionallyanalyzedexplicitzero/ID0TopKrawresultreusewithfullPoutput beforeallmapping/RNG/EPconsumers;320sourceorderpairsnotpointerbindings/cachehits.',
            'FP32counterwitnesses,cold/warmbankgenerationand48program17→17rowmodel;integrated24656Breferencebudgetversusseparateassembly52144Bextra;deduplicatejointpaddingbudget.',
            'M12retain15familiesenrich6/25othermodulespreserved;45relations10groups/profile2073/285598+9/1352/globaluncertainty/0moduleclosuresunchanged. Nextwholemodelbranchguardaudit.'],
        candidate_updates=updates,framework_updates={},module_updates=mods,semantic_branches=branches,remaining=remaining,action=action,jobs=jobs,report=report,
        read_first=names[:5]+names[6:],publication_rows={},gate_updates=dict(overall_complete=False),scripts=[N/'analyze_padding_route_reuse_v77.py',N/'verify_padding_route_reuse_v77.py'],
        executor_prefix='padding-route-reuse',main_script=Path(__file__),source_read=False)


if __name__=='__main__':main()
