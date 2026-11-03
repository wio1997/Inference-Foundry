"""Reconcile60 concrete operator regimes, protect full scope and persist v70."""
from collections import Counter
from copy import deepcopy
from pathlib import Path
from finalize_origin_coverage_v9 import N,O,read,write
from persist_profile_refinement_v5 import persist_refinement

def main():
    assert not(O/'REVIEW-CHECKPOINT-v70.json').exists()
    p=read(O/'selected-input-regime-models-v70.json');assert(p['regime_count'],p['captured_execution_count'])==(60,2024)
    assert read(O/'selected-input-fusion-variants-v70.json')['variant_count']==16
    g=read(O/'selected-input-native-chains-v70.json')['chains'];assert [len(g[x])for x in ['COMPOSED_DRAFT_ID_GATHER','AUX_GATHER_CONCAT','DRAFT_FIXED_BANK_TAILS','MARKOV_SEED_COLUMN']]==[72,72,88,104]
    costs=read(O/'selected-input-fusion-costs-v70.json');assert sum(x['source_4Fill_write_bytes']for x in costs['draft_tail_models'])==11531424
    c=deepcopy(read(O/'operator-analysis-coverage-ledger-v69.json'));rows={r['regime_id']:r for r in c['regimes']}
    e=['selected-input-regime-models-v70.json','selected-input-native-chains-v70.json','selected-input-operator-contract-v70.json','selected-input-fusion-costs-v70.json','selected-input-fusion-variants-v70.json','selected-input-def-use-and-guard-proof-v70.json','selected-input-math-validation-v70.json','operator-analysis-coverage-ledger-v70.json','operator-regime-refinement-v70.json','M21-selected-input-branch-inventory-v70.json','selected-input-protected-module-state-v70.json']
    for r in p['regimes']:
        x=rows[r['regime_id']];assert x['state'].startswith('REGIME_')and x['executions']==r['executions']and r['consumer_fusion_review_complete_in_this_turn']
        x['state']='SCOPED_OPERATOR_CONDITIONAL_FUSION_AND_WORK_IO_REVIEWED';x['scoped_conditional_review_evidence']+=e[:7]
    rev=[r for r in rows.values()if r['state'].startswith('SCOPED_')];pending=[r for r in rows.values()if r['state'].startswith('REGIME_')]
    assert(len(rev),sum(r['executions']for r in rev))==(2052,284990)and(len(pending),sum(r['executions']for r in pending))==(30,1960)
    assert Counter('|'.join(r['module_candidates'])for r in pending)=={'M21':21,'M19':4,'M12':3,'M05|M21':2}
    assert len(rows)==2082 and sum(r['executions']for r in rows.values())==286950
    assert c['unchanged_global_uncertainty']=={'ambiguous_module_assignment_executions':3136,'missing_PMU':849,'phase_unbound_graphs':16}
    assert(c['explicit_profile_insufficient_regimes'],c['explicit_profile_insufficient_executions'])==(7,1272)
    assert(c['explicit_unresolved_BOOL_field_regimes'],c['explicit_unresolved_BOOL_field_executions'])==(2,80)
    rem=[r for r in pending if r['module_candidates']==['M21']];assert sum(r['executions']for r in rem)==608
    assert Counter(r['operator']for r in rem)=={'SparseFlashMlaMetadata':12,'VllmAscendQuantLightningIndexerV2Metadata':6,'ZerosLike':3}
    c.update(parent_coverage_ledger='operator-analysis-coverage-ledger-v69.json',scoped_conditional_review_regimes=2052,scoped_conditional_review_executions=284990,remaining_regime_reconciliation_count=30,remaining_execution_reconciliation_count=1960,newly_reconciled_regimes=60,newly_reconciled_executions=2024,new_pending_regimes_reconciled=60,new_pending_executions_reconciled=2024,prior_geometry_promoted_to_role_conditional_review=0,prior_geometry_executions_promoted=0,remaining_regimes_by_operator=dict(Counter(r['operator']for r in pending)),remaining_executions_by_operator={op:sum(r['executions']for r in pending if r['operator']==op)for op in sorted(set(r['operator']for r in pending))})
    old={m['id']:m for m in read(O/'module-phase-branch-matrix-v69.json')['modules']};assert len(old)==26 and len(old['M21']['branches'])==18
    bs=deepcopy(old['M21']['branches']);details={
      'M21-S03':'72composedID5TASKgroups24S+24D→20Dlegaldomainor8S+12DfullLvalidation/localreuse. PreserveINT32J+1wrap/nestednegative/indexerrorsincludingunselectedL;originalLmetadataremainsneeded.',
      'M21-S10':'16initialauxconcat+72indexed3gather/concatnativegroupsquantifyexistingF27indirectmechanism. Indexed24T+24TH→4T+12THjointorprojection-direct;T6668jointbudget409815280logicalB.16prefixconcatT6400/6528;single15360Dcontraction/norm/TPreadyretained,no3roundedGEMMs.72paddedinput/positionprefixgathersandsample104(88real16dummy)separated.',
      'M21-S11':'88four-FillgroupsdraftCslot8208/seq8200/QSL8200/sample8192derivedserve8192/4/K5andallshapes.11,531,424Bnativewrites;steadyBdirty0,B4→2clear96Bunderfullsuffix/allwriterresetproof;first3beforemetadatareader,4thlater. NOTtarget13groupunknowncapacitybinding.',
      'M21-S12':'16fallbackselectorSubresultoverwrittenwithoutincomingreadinDSparkoverride.88positionsgathervaluesonly2loadsitesunder!parallel;config/constructorparallelTruelocalSHAsame166revisionconditionpluslegalindices/errorguardsneeded. Samplehidden104=88draft+16dummy,shapesnotidentityproof.104seedViewCopyonlyBINT64stride6,fullbufferdynamic_specobserverretained.',
      'M21-S15':'Inputprefixviews/indirectprojectionmustpreserveoldaux/sourceepochs,lastreaders,capturedaddressesandTPmapping. Dirtytailscannotcrossuntrackedreal/dummy/writer/replayreset. Source-onlycopiesnotinventednativeidentities;aliasvaluesnotcaptured.',
      'M21-S16':'2052regimes284990execconditionalreviewed,30regimes1960pending=18AICPUplans529+3planzero79+7MemSet1272+2BOOL80. All60nonplanM21regimes2024typedworknowbound. Unknownphysicalindex/Vector/AICPUcapacity/allapplicablebranchesstayopen;849PMU/16graphs/3136ambiguityunchanged.'}
    for b in bs:
        if b['id']in details:b['evidence']+=e[:9];b['typed_operator_review_v70']=details[b['id']]
        if b['id']=='M21-S16':b['state']='TYPED_OPERATOR_BOUNDS_PARTIAL;21_PLANNER_REGIMES_AND_PHYSICAL_CAPACITY_OPEN'
    write(O/e[7],c)
    write(O/e[8],dict(refined_regimes=[dict(regime_id=r['regime_id'],family=r['family'],executions=r['executions'],scope='ExactselectedrowIO,consumerdef-use,conditionalfusion/error/precisioncontracts.')for r in p['regimes']],reviewed=[60,2024],remaining_M21=[21,608],remaining_opaque_metadata=[18,529],remaining_opaque_family_clear=[3,79],selected_missing_PMU=0,sample_hidden_native_callers={'CALLER_DRAFT_OR_DRAFT_METADATA':88,'CALLER_DUMMY':16},whole_modules_closed=0,overall_complete=False))
    write(O/e[9],dict(modules={'M21':bs},retained_semantic_families=18,extended_families=sorted(details),extended_family_count=6,scope='18M21familiesretained6extended.25othermodulematrixrecordsunchanged,F27existingmechanismwithnewM21inputcostevidence.',whole_modules_closed=0,overall_complete=False))
    write(O/e[10],dict(modules={mid:m for mid,m in old.items()if mid!='M21'},previous_M21_branches=old['M21']['branches'],extended_M21_family_ids=sorted(details),whole_modules_closed=0,overall_complete=False))
    write(O/'PROFILE-CODE-QUESTION-GATE-v70.json',dict(status='SELECTED_INPUT_GATHER_CONCAT_TAIL_SEED_OPERATOR_WORK_AND_DEFUSE',previous_turn_classification='PROGRESS:v69remaining_slot_draft_nextID_RoPE_mask_fields',questions=['CansequentialIDgathersbecomposedwithoutlosingfullfirst-indexvalidation?','Whatistheactual3auxselectedcopy/concatworkversusexistingF27indirectsinglecontraction?','Arepaddedinputselectorsidentityandwhichdummy/headselectorsarenot?','Whichinitialselector/nextsteppositionvaluesareunusedunderDSparkguards?','Whatcapacities/writeintervalsdothefourpersistentdrafttailFillshave,andwhichreadinessfrontierspreventblindlaunchcoalescing?','Doescolumn0returnexclusionactuallyproveMarkovseeddeadness?'],allowed_source='Exact8localproducer/consumer/allocation/guardfiles:6oldhistoricalSHApins+2localconfig/constructorfileswithsame166revisioncondition. Notframework/APIresearch.',new_remote_source_read=False,source_read=False,local_operator_clause_review=True,no_model_Torch_NPU_capture_health_process_kill_implementation=True,overall_complete=False))
    updates={}
    for cid,scope in [('F31','ComposedID/selectedpositions/DSparkdeadtemps/fourbanktails/Markovseedconsumerpreciseoperatorconditions.60M21regimes2024TASK16choices.'),('F27','16auxprefixconcatand72threegather+concatnativeM21groupspricedforexistingF27-v34-INDIRECT/v64single15360Dcontraction.Newmaterializationcostevidence,notnewmissingfusion.')]:
        updates[cid]=dict(review_v70=dict(regime_models=str(O/e[0]),native_chains=str(O/e[1]),operator_contract=str(O/e[2]),fusion_costs=str(O/e[3]),variants=str(O/e[4]),defuse_guard_proof=str(O/e[5]),candidate_specific_scope=scope,conditional_operator_analysis_deliverable=True,implementation=False,native_test=False,new_performance_result=False),profile_review_state='SELECTED_INPUT_OPERATOR_WORK_AND_CONSUMER_CONDITIONS_RECONCILED',conclusion_changing_gap='Opaque18AICPUplanregimes529+3planzeros79remainunreviewed;7MemSet1272/2BOOL80operandsunbound. Actualalias/value/bankgenerations/allreaders/nativeindex-Vector-AICPUcapacityandallapplicablemodelbranchesremainconditions. 2supportconfig/constructorfileslocalSHAonlyretain166sameversioncondition. ExistingF27projection/communication/normprecisionconstraintsretained.',next_profile_or_contract_action='NextM21remaining21planregimes608:internalSMLA/QLIalgorithm/1024INT32publication/purity-value-addressdependenceandactualzero-caseconsumer. Readonlysavedprofilefirst,onlyexactnativeoperatorAPI/sourceclauseifneeded;noimplementation/serverjobs.')
    mods={'M21':dict(finding='60regimes2024TASKexactselectedworkand16choices.72composedID5TASK,72aux7TASK,16prefixconcat,88fourtailFill/11.531424MB,104seedBINT64copies,72ID+72position+88usedposition+104samplehiddengroups. ExistingF27indirectmechanismnewnativecostbinding;2deadvalueclauses/seedfullbufferconsumerguarded.',evidence=e[:10],candidates=['F31','F27'],gap=updates['F31']['conclusion_changing_gap'],profile_scope='Onlyoperatorfusion/necessaryselectedwork/consumer/error/precision/tail-frontiercontracts;25othermodulesunchanged.')}
    remaining=[]
    for s in read(O/'module-coverage-gaps-v69.json')['remaining']:
        if s.startswith('v69new'):continue
        if s.startswith('Observedoperatorfrontier'):s='Observedoperatorfrontier2082regimes286950ComputeInfoexec83types:2052regimes284990conditionalreviewed,30regimes1960pending=21M21/608+7MemSet/1272+2BOOL/80. Profilefrontiernotwholemodel/allbranches/physicalupperlimitclosure.'
        if s.startswith('M21remaining81'):s='M21remaining21regimes608exec:18SMLA/QLIopaqueAICPUplannerregimes529+3planbankzero79. AllotherobservedM21regimesnowtypedconditionalanalysis;internalalgorithm/input-key-purity-addressdependence/planreader-zerocasephysicalworkstillunbound. No generic4096Boutputformulaasanall-workanswer.'
        remaining.append(s)
    remaining.append('v70new60regimes2024TASK/16choices.72IDnestedgathers24S+24D→20Dor8S+12DwithfullLvalidation,unselectedinvalidcounterexample/INT32wrapkept.72auxgatherconcat24T+24TH→4T+12TH;existingF27indirectsingle15360Dprojectioncanavoidprivateplanes,not3BF16roundedGEMMs.16prefixconcatT6400logical393216000Bpercall.88fourtailFill11531424B,C8208/8200/8200/8192boundonlyDRAFT;allwriter/resetdirtyproofrequired,first3beforemetadatareader4thlater.16initialselectorand88usedpositionvaluesdeadonlyspecifiedoverride/parallelguardandlegalindex/errorconditions. Samplehidden88draft+16dummy/no-shape-identity.104seedstride6copy16Btrafficnot192Bbase,fullbufferdynamic_specobserverretained.2052/284990review30/1960pending/0wholemodules;3136ambiguity/849PMU/16graphsprotected.')
    action=('继续原Goal最新v70 PROGRESS；v69也是PROGRESS。用户只要算子是否可融合/拆分重组及理论上限，profile优先，代码只补具体读写/依赖/精度/消费者，不研究框架。'
      '先读operator-analysis-coverage-ledger-v70、selected-input-regime-models/native-chains/operator-contract/fusion-costs/fusion-variants/def-use-and-guard-proof-v70及M21-selected-input-branch-inventory-v70。'
      '本轮60regimes2024TASK/16choices：72nestedID5TASK24S+24D→20Dlegaldomain或8S+12D保留全L检查与局部复用；不能只检selectedL，A[10,20]L[7,0]J[0]反例。72aux7TASK24T+24TH→4T+12TH；F27-v34/v64直接三源单15360D收缩已存在候选，新增原生M21成本不是重发现。16prefixConcatT6400/6528，6400预算393216000B逻辑。'
      '88four-tailFills原生11531424B，草稿bankCslot8208/seq8200/QSL8200/sample8192由serve8192/4/K5及每组形状绑定；不是target13slotC。dirty后缀steadyB0、B4→2清96B条件allwriter/reset/fence，前三字段在metadata builder前/第四later不能随便延后合启动。'
      '16fallbackSub incomingselector被DSpark首遍无读覆盖；88usedposition值只有2个!parallelguard读取，config设parallelTrue/构造赋值两support文件只localSHA，同166版本条件保留；删IndexCheck要合法索引证明。104samplehidden=88real+16dummy，dummy零索引不identity。104seedViewCopy仅B个INT64 stride6复制16B流量、24base非copy，dynamic_spec收到fullbuffer是额外observer，不得因返回去col0就宣称seed死。'
      '当前2052regimes284990conditionalreviewed/30regimes1960pending=18AICPUplans529+3planzero79+7MemSet1272+2BOOL80。geometry0，2082/286950总数守恒。35model+7legacyframework候选，0wholemodules。'
      '下一步优先18AICPUplanner实际算法/ABI证据与3planzeros：先opaque-plan-TASK-type-audit-v69和v37SMLA/QLI调用契约，明确1024INT32输出内容、输入值/属性/地址/设备依赖、纯函数cachekey/消费者blob契约及实际0分支写入；只查exactnative算子实现/接口，不能以4096B输出或AICPU无PMU当内部work完成。随后7MemSet1272与2BOOL80operand/capacity缺口。'
      '保留模型全部适用未观测分支/跨层卡迭代布局和关键路径/硬件物理容量缺口，不能仅因profile覆盖即关Goal。849missingPMU/3136functionalambiguity/16phase-unbound不变。全本地savedprofile/源码operatorclauses/数学，无remote/model/Torch/NPU/capture/健康/进程/kill/实现；必要时只读已有容器指定算子文件无需启动。')
    report='''本轮按算子的选中行、实际读写和消费者检查，补齐60类、2,024次执行的条件分析，记录16个方案或排除结论。没有连接服务器、执行模型或修改推理实现。

| 已对齐范围 | 已知逻辑读写模型 | 融合或省略条件 |
|---|---|---|
| 72组连续token gather，共5任务/组 | 24S+24D → 20D；保留完整首层检查及局部索引复用时为8S+12D | 保留INT32的J+1、逐级负索引、对未选中L的验证，以及原L元数据 |
| 72组三份辅助hidden gather后concat，共7任务/组 | 24T+24TH → 4T+12TH | 三份输入的顺序、重复索引、边界、代际及最终6TH字节输出都保留 |
| 16次辅助hidden前缀concat | 原12TH | 已有F27直接三源投影候选可避免该私有拼接；原投影计算、通信和完整特征norm仍需要 |
| 88组四个草稿bank尾部fill | 原共11,531,424B写入 | 维护完整常量后缀后，稳态相同B可不重复写；必须跟踪所有写者与reset |
| 104次Markov seed列复制 | 每次16B，其中B=2或4 | 24个INT64是底层容量；full-buffer消费者和seed代际仍需处理 |

S是首次logits索引数，D是第二次选中的draft token数；T是选中行数，H=5120，B是请求数。以上是原生描述符和源语义对应的逻辑值访问，不是实测HBM流量或请求加速。

连续gather的合并不能只检查最终选中的地址。例如A=[10,20]、L=[7,0]、J=[0]时，直接读取A[L[J+1]]会得到10，但原先A[L]已经因未选中的7越界而失败。保留整个L的检查，或证明原输入全合法，才允许省去该检查。固定profile的S仅12、14或24，可研究把全部INT64索引放在局部存储后复用；原返回给其他消费者的L元数据仍要发布。

三份辅助hidden的准备开销较大。在T=6668的7任务组中，联合gather与concat的逻辑接口预算为409,815,280B；在T=6400的直接前缀concat中，拼接本身读写393,216,000B。这些预算补充到既有F27-v34-INDIRECT及v64方案：让一个15360维收缩直接读取三份有序输入，保持BF16投影边界和后续norm。它不减少原投影的2×T×15360×640次常规计算，也不能换成三次分别舍入的BF16 GEMM再相加。联合搬运、前缀view和投影间接读取是不同粒度的替代方案，收益不能叠加。

prepare_inputs_padded返回的是arange[:T]，会保留拒绝token作为padding；该路径的输入ID和INT64 position gather可研究改为前缀复制，或在生命周期允许时直接供消费者读取。unpadded路径仍可能有任意有序索引。sample_hidden共有88次真实草稿和16次dummy调用；dummy可以重复读取第0行，所以输入输出shape相同不证明索引为identity。源版本、索引生产者和别名代际比shape相等更关键。

定点检查还发现两类未消费的值：DSpark首遍会在读取之前覆盖传入selector，因此16次初始QSL-1临时结果可研究省去；used_update_positions只有两处读取，均在非parallel的下一步更新分支下，DSpark parallel drafting时其值没有消费者。后者删除索引检查仍须证明合法索引或保持错误行为。用于确认parallel=True和构造赋值的两份支持文件目前只有本地冻结SHA，本轮保留与166版本一致的条件；其他六份算子文件复用历史SHA。

四个草稿bank容量已由固定启动参数和全部88组fill形状相互验证：slot为8208，seq_lens与QSL为8200，采样索引为8192。它们不绑定上一轮13组目标slot指针所指向的未知容量。完整后缀不变式成立后，B=4降至2只需把24个INT32恢复为常量，共96B；稳定B时尾部不必反复清理。但前三个字段在attention metadata构造前就有读者，最后的采样索引fill更晚，不能把四者无条件延后到一起执行。前缀数据仍要更新，真实请求转dummy、buffer重绑和其他写者均需重置规则。

Markov seed的col0虽然不在最终[:,1:]返回值中，dynamic_spec.update仍接收完整draft buffer。首步直接读取seed bank是一个带条件的新接口选择，不能仅因返回切片排除了col0就删除复制。

本轮核对72组连续gather、72组辅助gather/concat、88组四bank fill及104组seed复制成本，并执行8,132个负索引/重复索引组合和780个三源选择拼接等价检查。这是纯整数与布局分析，不是NPU实现正确性或性能测试。

已采集范围累计2052类、284,990次执行有显式条件分析；仍有30类、1,960次待分析：18类AICPU计划生成、3类计划清零、7类MemSet和2类BOOL字段。所有适用模型分支、物理上限和整模块闭环尚未完成，Goal保持进行中。
'''
    jobs=[
      dict(kind='profile',status='pass',description='60regimes2024uniqueTASK/0missingPMU;72composed5TASK/72aux7TASK/88tail4TASK/104seed/16prefixconcatand336simplegathergroupsconserved.',artifacts=[O/e[0],O/e[1],N/'analyze_selected_inputs_v70.py']),
      dict(kind='design-check',status='pass',description='Alltypedgroupcostsexact;8132nestednegative/repeatcases780threeauxcases;fullLinvalid/dummyidentity/INT32wrapcounterexamples;fourbankdirtysequencechecked.16conditionaloperatorchoices/noNPUtest.',artifacts=[O/e[3],O/e[4],O/e[6],N/'model_selected_inputs_v70.py']),
      dict(kind='review',status='pass',description='Exactdef-useDSparkincomingselectoroverwrite/2positionconsumerloadsunder!parallel/fullbufferdynamic_specobserver;draftcapacityserveSHA/all88groups.6historicalsourceSHApins+2localfrozenconfig/constructorconditions;existingF27indirectmechanismreused.',artifacts=[O/e[2],O/e[5]]+[Path(r['path'])for r in read(O/e[2])['source_files']]+[N/'original-profile-cache-v53/runtime/scripts/serve.sh']),
      dict(kind='review',status='pass',description='2052/284990review30/1960pending=2082/286950;all60nonplanM21scopedreview,18AICPU529+3zero79+7MemSet1272+2BOOL80pending.18M21familiesretained6extended25othermodulesidentical;35model7framework0wholemodules/84916/3136uncertaintykept.',artifacts=[O/e[7],O/e[8],O/e[9],O/e[10]])]
    persist_refinement(version=70,previous=69,title='Selected input gather, concat, draft tails and seed operator work',summary=dict(filename='selected-input-ceiling-summary-v70.json',previous_turn_classification='PROGRESS:v69remaining_slot_draft_nextID_RoPE_mask_fields',reviewed_regimes=60,reviewed_executions=2024,operator_decisions=16,observed_regimes=2082,scoped_reviewed_regimes=2052,scoped_reviewed_executions=284990,pending_regime_reconciliation=30,pending_execution_reconciliation=1960,remaining_M21_regimes=21,remaining_M21_executions=608,opaque_metadata_regimes_pending=18,opaque_metadata_executions_pending=529,plan_zero_regimes_pending=3,plan_zero_executions_pending=79,MemSet_explicit_pending_regimes=7,MemSet_explicit_pending_executions=1272,BOOL_regimes_pending=2,BOOL_executions_pending=80,M21_retained_semantic_families=18,M21_extended_families=6,protected_other_modules=25,source_files=8,prior_historical_source_SHA_matches=6,two_support_files_same166revision_condition=True,new_remote_source_read=False,local_operator_clause_review=True,implementation=False,native_test=False,new_performance_result=False,all_actual_physical_numeric_ceilings_proved=False,remote_job_status='No newremoteconnection/job;alllocalsavedprofile/operatorclauses/puremath. No model/Torch/NPU/capture/health/process/kill/implementation/backgroundanalysis.'),changes=['60regimes2024TASKtypedselectedwork/consumercontractsand16operatorchoices;allnativegroupcounts/costsconserved.',
      'ExistingF27indirectprojectionnewM21auxgather/concatbudget;composedIDfullLvalidation/INT32wrap,source-specificunusedvalues,actualdrafttailcapacities/readiness/dirtysuffixandseedfullbufferconsumerconditionsbound.',
      '2052/284990review30/1960pending,18M21familiesretained6extended25othermodulesidentical.6historicalpins+2frozenconditionalfiles/globaluncertaintyand0moduleclosurepreserved.'],candidate_updates=updates,framework_updates={},module_updates=mods,semantic_branches={'M21':bs},remaining=remaining,action=action,jobs=jobs,report=report,read_first=[e[7],e[0],e[1],e[3],e[4],e[2],e[5],e[9]],publication_rows={},gate_updates=dict(overall_complete=False),scripts=[N/x for x in ['analyze_selected_inputs_v70.py','model_selected_inputs_v70.py','verify_selected_inputs_v70.py']],executor_prefix='zcode-selected-inputs',main_script=Path(__file__),source_read=False)

if __name__=='__main__':main()
