"""Persist operator-specific analysis; remaining scope is explicitly open."""
from copy import deepcopy
from pathlib import Path
from finalize_origin_coverage_v9 import N,O,read,write
from persist_profile_refinement_v5 import persist_refinement

def main():
    assert not (O/'REVIEW-CHECKPOINT-v69.json').exists()
    p=read(O/'remaining-field-regime-models-v69.json');assert(p['regime_count'],p['captured_execution_count'])==(161,6680)
    cov=read(O/'operator-analysis-coverage-ledger-v69.json');assert(cov['scoped_conditional_review_regimes'],cov['remaining_regime_reconciliation_count'])==(1992,90)
    inv=read(O/'M21-remaining-field-branch-inventory-v69.json');assert inv['extended_family_count']==9
    assert read(O/'remaining-field-fusion-variants-v69.json')['variant_count']==20
    e=['remaining-field-regime-models-v69.json','remaining-field-native-chains-v69.json','remaining-field-operator-contract-v69.json','remaining-field-fusion-costs-v69.json','remaining-field-fusion-variants-v69.json','remaining-field-math-validation-v69.json','opaque-plan-TASK-type-audit-v69.json','MC2-mask-typed-valuation-v69.json','remaining-field-source-binding-v69.json','operator-analysis-coverage-ledger-v69.json','operator-regime-refinement-v69.json','M21-remaining-field-branch-inventory-v69.json']
    old={m['id']:m for m in read(O/'module-phase-branch-matrix-v68.json')['modules']};assert len(old)==26
    write(O/'remaining-field-protected-module-state-v69.json',dict(modules={mid:m for mid,m in old.items()if mid!='M21'},previous_M21_branches=deepcopy(old['M21']['branches']),extended_M21_family_ids=inv['extended_families'],scope='25othermodulesidentical;18M21familiesretained9extended,noothersemanticclausesremoved.',overall_complete=False))
    e.append('remaining-field-protected-module-state-v69.json')
    write(O/'PROFILE-CODE-QUESTION-GATE-v69.json',dict(status='SPECIFIC_SLOT_SWA_NEXT_ID_ROPE_MASK_OPERATOR_WORK_AND_FUSION_CONTRACTS',previous_turn_classification='PROGRESS:v68typed_metadata_fields',questions=['Whatdoactualslot/copy-expand/SWAkernelsalreadyfuseandwhichoutputrangesareunavoidable?','Canfull-bankPAD/BOOLthresholdinvariantspermitdirtyupdateswithoutchanginganyreader?','Whatdoesnext-IDcount-indexselectactuallymeanforholes/negativeIDsandhowcanitjointlypublishreject/selectors?','HowdocachedgatherandadvancedindexRoPElookupdifferinwork/error/precision?','Whichresourceexecutesopaqueplansandwhatinternalalgorithm/capacityevidenceisstillmissing?'],source_scope='Onlyexactoperatorcaller/kernel/allocationclausesfrom10localfrozenfiles;9priorhistoricalSHApinsand1forwardcontextfrozen/profilecallsite-onlyconditionalbinding. No framework/API lifecycle research.',source_read=False,local_operator_clause_review=True,new_remote_read=False,no_model_Torch_NPU_capture_health_process_kill_implementation=True,overall_complete=False))
    u=dict(review_v69=dict(regime_models=str(O/e[0]),native_chains=str(O/e[1]),operator_contract=str(O/e[2]),fusion_costs=str(O/e[3]),variants=str(O/e[4]),source_binding=str(O/e[8]),conditional_operator_analysis_deliverable=True,implementation=False,native_test=False,new_performance_result=False,candidate_specific_scope='161regimes6680TASK;88next13TASK/288cachedcos+288sin/96advancedcos+96sin/88slot/88SWA/72prepare;MC2directmask/dirtysuffix/copyexpand/936tailfills.20conditionalchoices/exclusions.'),profile_review_state='SLOT_DRAFT_SWA_NEXT_ID_ROPE_BOOL_TYPED_OPERATOR_WORK_REVIEWED',conclusion_changing_gap='81remainingM21regimes2632execincluding18opaqueAICPUplans529+3planzeros79;actualbankcapacity/alias/generations/values/allreaders/nativeVector-index-AICPUcapacityremainconditions. Forwardcontextindependent166fileSHAunbound. Implementationabsencealonedoesnotinvalidateconditionalanalysis.',next_profile_or_contract_action='Review60remainingnonplanregimes2024TASKselectedindex/composedaccessor/fills,then18AICPUplannerinternalworkand3planzeros;7MemSet1272+2BOOL80remainwithoutoperands. Profilefirst;onlyspecificoperatorcodegaps.')
    mods={'M21':dict(finding='161regimes6680TASKexacttypedoperatorworkand20alternatives. Existing13group/adaptivetiling/copyexpand/SWA/reject+selectorfusionbaseline.88next-IDcount-indexcores38BK+94B;cached4112N/advanced1056N rowlookup1032N;symboliccapacityslotdirty52*(N+max(oldN-N,0));SWArequestcompactnewABI;MC2logical1461368→161984BOOLbytes.529opaqueplansAICPUinternalworkstillpending.',evidence=e[:12],candidates=['F31'],gap=u['conclusion_changing_gap'],profile_scope='Onlyoperatorfusion/necessarywork/IO/consumers/math;161actualregimes,25othermodulesunchanged. No wholemodule closure.')}
    remaining=[]
    for s in read(O/'module-coverage-gaps-v68.json')['remaining']:
        if s.startswith('v68new'):continue
        if s.startswith('Observedoperatorfrontier'):s='Observedoperatorfrontier2082regimes286950ComputeInfoexec83types:1992regimes282966reviewed,90regimes3984pending=81M21/2632+7MemSet/1272+2BOOL/80. Notallmodelbranches/physicalcapacityorwholegoalcompletion.'
        if s.startswith('M21remaininginput/')or s.startswith('M21remaining242'):s='M21remaining81regimes2632exec:60nonplanregimes2024TASKinput/accessor/composedgather/fills,18opaqueAICPUplanregimes529+3planzero79. Slot/SWA/draft/next-ID/RoPE/mask161regimes6680nowconditionalreview;529AICPUresourcebindingdoesnotcloseinternalwork. Onlyspecificoperatorcodegaps.'
        remaining.append(s)
    remaining.append('v69new161regimes6680TASK/20operatorchoices. Exactnextcount-indexnotrightmostvalid;[0,-1,3]counterexample,negative<-1valid.13TASK38BK+94B→4BK+16B;optional72preparejoint38BK+122B-4→4BK+36B-4. CachedRoPE4112N→1032N,advanced1056N→1032Nwithnegativewrap/checks;FLOATbanks512Nkept. G13slotoutput52*C,Cunbound,dirtysuffixonlyfullbankinvariant;SWAalreadyjointindices+rawlengths,N5Brequestcompactoutput5xneedsnewconsumerABI. MC2masktypedlogical1461368→161984BOOLvalues,notmeasuredHBM/requestgain.10localfiles9priorSHApins;forwardcontextsame166revisioncondition.90regimes3984pending/0wholemodules/849PMU/16graphs/3136ambiguityremain.')
    action=('继续原Goal最新v69 PROGRESS；v68也是PROGRESS。用户要求只围绕算子融合/拆分重组、必要计算/访存/通信/依赖及理论上限；profile优先，只有具体契约缺口才定点查代码，不研究框架。'
      '先读operator-analysis-coverage-ledger-v69、remaining-field-regime-models/native-chains/operator-contract/fusion-costs/fusion-variants-v69、MC2-mask-typed-valuation-v69和remaining-field-source-binding-v69。'
      '本轮161regimes6680TASK/20方案：88next13TASK38BK+94B→4BK+16B、可接72prepare双输出；count-1不能换最右有效，[0,-1,3]取-1，<-1仍valid。288+288cachedRoPE3TASK4112N→1032N，96+96advanced2TASK1056N→1032N保留负索引/检查/FLOATbanks。'
      'target13组slot已融合/adaptive为基线，UINT64[13]是地址数组，输出52*C而C未绑定；dirty suffix需全bank已PAD/历史活跃末端/全部writer-reader/reset/capture-generation。SWA88已单kernel输出索引+raw长度；N5B请求重排5倍输出节省需要新SparseFlashMla/CP读者ABI；同layout按请求tile仍写满N*C。'
      'copy-expand88现有两loop融合，context20T实际活跃行，query输入28+4R每行、输出12BQ+4BD，不能按8192capacity算写量；Gdraft不能套target13。MC2mask42regimes1448TASK已定点契约/typedlogic1461368→161984BOOL值；未知80BOOL不归因。'
      '10本地源码9旧历史SHA一致，forwardcontext只冻结SHA+profile同位置，无独立166文件SHA，保留版本相同条件。529opaqueplans全KERNEL_AICPU，不套Cube/Vector，也不据此清零849missingPMU或关闭内部工作。'
      '下一步先其余60非plan M21regimes2024TASK：worker1836/1837复合gather、2069/2086/2089concat-index、_get_positions1031、proposer1061/1212/1216/1220/1246/1343/1534/1584；复用v38/v65/v67已知契约核算选中元素/compose/error/alias，不研究框架。然后18AICPUplanregimes529+3planzeros79内部算法/纯函数地址依赖证据。'
      '当前1992regimes282966reviewed、90/3984pending=81M21/2632+7MemSet1272+2BOOL80，0geometry；2082/286950总数守恒。35model+7legacyframeworkcandidates/0wholemodules。'
      '实现缺失不阻碍条件设计交付，但真实未分析模块/分支/物理容量不关闭Goal。3136ambiguity/849missingPMU/16phase-unbound保护。全本地saved-profile/冻结operator代码/纯数学；无remote/model/Torch/NPU/capture/健康/进程/kill/实现。')
    report='''本轮继续围绕算子是否可以融合及理论工作量，分析161类、6,680次执行。只使用保存profile和定点算子代码，没有连接服务器或执行模型。记录20个条件方案、已有机制及排除结论，未把已有融合算作新增收益。

| 算子范围 | 条件工作量或读写模型 | 关键约束 |
|---|---|---|
| 下一token选择，88组13任务 | 38BK+94B → 4BK+16B | 保留原有效条件、count-1选择、INT64计数、INT32 token、discard及backup |
| 与已有reject/selector kernel联合，K=6 | 38BK+122B-4 → 4BK+36B-4 | 72次接口范围；同一请求与代际，cu/QSL已就绪，计数其他读者保留 |
| 缓存RoPE cos/sin查询，288对 | 4112N → 1032N | 64维扩展索引可改按行；原INT32转换、越界语义和FLOAT输出保留 |
| 非缓存RoPE高级索引，96对 | 1056N → 1032N | 保留负索引回绕和验证；此路径原本没有64维索引扩展 |
| 13组slot mapping，88次 | 原输出52C；条件dirty方案52×(N+max(oldN-N,0)) | 已有分组融合与自适应tile；C仍未绑定，不能从13个指针推断写入容量 |
| SWA索引，88次 | 原必写4N(C+1)；请求紧凑格式4B(C+1) | 已有单kernel生成索引与raw长度；N=5B时输出存储可缩小5倍，但需要新消费者接口 |

B表示请求数，K表示每请求采样列数，N表示查询行数，slot表容量C与SWA索引宽度C是不同上下文的变量。上述均为逻辑值读写模型，不是实测HBM流量或请求加速比例。

下一token的源码虽然注释写“最右有效”，实际算法先计数，再读取count-1位置。[0,-1,3]有两个有效项，实际选中-1，并非3；[-2]在原条件(x!=-1)&(x<V)下仍有效。融合可以消除中间张量，但必须保持这些实际语义。已核对88组原生13任务成本，并以39,060个有限用例检查条件公式；这不构成设备实现测试。

slot mapping已经把13组放在同一启动，并按请求长度选择ordinary/adaptive策略。源码还把INT32 block号转为FLOAT做gather再转回；16777217会变为16777216，直接整数查表要保留该行为或证明值域。跨迭代只清理上次活跃而本次不活跃的后缀，能在完整PAD后缀不变式成立时保持整个bank一致；需要逐bank跟踪所有写者、读者、重置和回放代际。不能据此宣称936次worker尾部fill都可删除，也不能跳过真实请求转dummy时的失效处理。

SWA当前每个query扫描请求结束位置并重算同一请求窗口。按请求组织tile可以减少这些读和算术，但当前N×C格式仍要写满输出。改成B×C需要SparseFlashMla、CP切片及其他读者共同支持请求映射；不能默认stride为0的expand已被现有算子接受。raw visible length也必须保留，不能直接改成截断后的有效索引计数。

草稿copy-expand已经融合context复制和query展开，并使用向量核数量限制grid。实际context复制为每活跃行20B逻辑值读写；8192是输出bank容量。query每行参考输入为28B，带reject再加4B，输出位置/ID/slot共12B，另有采样索引。query的RoPE位置来自接受后context的位置，而物理KV slot来自权威序列长度和DCP归属，两者在多模态时不能混同。目标的13个cache组也不能直接当成草稿实际组数。

MC2布尔mask已按源码确认是[0:A)为True、[A:P)为False。42类、1,448次任务的已知逻辑值读写合计1,461,368B，其中最终BOOL输出161,984B；直接生成最终mask可研究避免INT32广播与转换。跨迭代只改阈值变动区间，需要先建立完整bank不变式，再约束全部写者和回放参数。这项分析没有给尚未明确字段的80次BOOL写入强行归因。10个本地算子文件中，9个复用历史SHA绑定；forward_context目前仅有冻结文件SHA和匹配profile位置，应用于166的结论保留同版本条件。

529次SMLA/QLI计划生成全部是KERNEL_AICPU。blockNum为0、输出为4096B不能说明没有计算，也不能套Cube或Vector峰值。内部算法、输入值依赖、地址绑定和可复用性仍缺证据；18类计划任务及3类计划清零继续待分析，原849次缺PMU总数保留。

当前1992类、282,966次执行有显式条件分析；90类、3,984次仍待核对，其中81类属于M21，另有7类MemSet及2类BOOL字段。全部适用分支、物理上限与整模块闭环仍未完成。整体Goal继续保持进行中。
'''
    jobs=[
      dict(kind='review',status='error',description='ResolvedinitiallocalImportError:Snotexportedbyhelper;definedW/frozen-source-snapshotandreransuccessfullybeforeregimeoutput/sharedauthoritywrites. Notdevicefailure.',artifacts=[O/e[5],N/'analyze_remaining_fields_v69.py']),
      dict(kind='profile',status='pass',description='161regimes6680uniqueTASKs/0selectedmissingPMU;88next13TASK/288+288cached/96+96advancedgroupsand529KERNEL_AICPUplansresourcecensus.',artifacts=[O/e[0],O/e[1],O/e[6],N/'analyze_remaining_fields_v69.py']),
      dict(kind='design-check',status='pass',description='20specificoperatorchoices/exclusions;all88next/384RoPEpaircostsexacttyped.39060count-indexcases+5524fullsuffix+5524BOOLthresholdcases;holes/negative/FP32blockcounterexamples.',artifacts=[O/e[3],O/e[4],O/e[5],N/'model_remaining_fields_v69.py']),
      dict(kind='review',status='pass',description='10localoperatorfiles9priorhistoricalSHApins;forwardcontextfrozenlocal/profilecallsiteonlysameversioncondition. Existinggroup/copyexpand/SWA/reject-selectorfusionbaseline,notnewgains. MC2logicalvaluesaudited,no80BOOLidentityclaim.',artifacts=[O/e[2],O/e[7],O/e[8]]+[Path(r['path'])for r in read(O/e[2])['source_files']]),
      dict(kind='review',status='pass',description='1992/282966review+90/3984pending=2082/286950;81M21/2632includes18AICPU529+3zero79;7MemSet1272+2BOOL80and849/16/3136uncertaintykept.18familiesretained9extended;25othermodulesidentical.',artifacts=[O/e[9],O/e[10],O/e[11],O/e[12],N/'consolidate_remaining_fields_v69.py'])]
    persist_refinement(version=69,previous=68,title='Slot, draft, next-ID, SWA, RoPE and mask operator work and fusion bounds',
      summary=dict(filename='remaining-field-ceiling-summary-v69.json',previous_turn_classification='PROGRESS:v68typed_metadata_fields',reviewed_regimes=161,reviewed_executions=6680,operator_decisions=20,observed_regimes=2082,scoped_reviewed_regimes=1992,scoped_reviewed_executions=282966,pending_regime_reconciliation=90,pending_execution_reconciliation=3984,remaining_M21_regimes=81,remaining_M21_executions=2632,opaque_metadata_regimes_pending=18,opaque_metadata_executions_pending=529,opaque_TASK_resource='KERNEL_AICPU',plan_zero_regimes_pending=3,plan_zero_executions_pending=79,MemSet_explicit_pending_regimes=7,MemSet_explicit_pending_executions=1272,BOOL_regimes_pending=2,BOOL_executions_pending=80,M21_retained_semantic_families=18,M21_extended_families=9,protected_other_modules=25,source_files=10,prior_source_SHA_bindings=9,forward_context_same166version_condition=True,implementation=False,native_test=False,new_performance_result=False,all_actual_physical_numeric_ceilings_proved=False,new_remote_source_read=False,local_operator_clause_review=True,remote_job_status='No newremoteconnection/job;alllocal savedprofile/frozenoperatorclauses/math. No model/Torch/NPU/capture/health/process/kill/implementation/backgroundanalysis.'),
      changes=['161regimes6680TASKspecificwork/IO/consumerreview;20operatorchoices/exclusionsandnativecountsconserved. Existingfusionexplicitbaseline.',
        'Nextcount-indexexactsemantics/typedcosts,RoPEcached-vsadvancedrowlookup,fullbankdirtysuffix/maskproof,SWArequestcompactnewABI,copyexpandactualextentsanddummy/tailconditionsquantified.',
        'Opaque529TASKsallAICPUwithinternalworkpending;1992/282966review90/3984pending,18M21familiesretained9extendedand25othermodulesidentical.9priorSHApins+forwardcontextversioncondition;globaluncertainty/0moduleclosurekept.'],
      candidate_updates={'F31':u},framework_updates={},module_updates=mods,semantic_branches=inv['modules'],remaining=remaining,action=action,jobs=jobs,report=report,read_first=[e[9],e[0],e[1],e[3],e[4],e[2],e[8],e[7],e[6],e[11]],publication_rows={},gate_updates=dict(overall_complete=False),scripts=[N/x for x in ['analyze_remaining_fields_v69.py','model_remaining_fields_v69.py','consolidate_remaining_fields_v69.py','verify_remaining_fields_v69.py']],executor_prefix='zcode-remaining-fields',main_script=Path(__file__),source_read=False)

if __name__=='__main__':main()
