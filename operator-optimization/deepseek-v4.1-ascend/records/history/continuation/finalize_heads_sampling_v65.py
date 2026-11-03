"""Persist operator head/sampling analysis with explicit pending full scope."""
from pathlib import Path
from copy import deepcopy
import hashlib
from finalize_origin_coverage_v9 import N,O,read,write
from persist_profile_refinement_v5 import persist_refinement

def main():
    assert not (O/'REVIEW-CHECKPOINT-v65.json').exists()
    p=read(O/'heads-sampling-regime-models-v65.json');g=read(O/'heads-sampling-native-chains-v65.json')
    cost=read(O/'heads-sampling-fusion-costs-v65.json');v=read(O/'heads-sampling-fusion-variants-v65.json')
    cov=read(O/'operator-analysis-coverage-ledger-v65.json')
    inv=read(O/'M15-M16-M17-M20-M24-operator-branch-inventory-v65.json')
    contract=read(O/'heads-sampling-operator-contract-v65.json')
    assert p['regime_count']==166 and p['captured_execution_count']==9792
    assert [len(g[k])for k in ['target_tails','base_heads','Markov_steps','greedy_rejections','state_corrections']]==[104,192,520,72,72]
    assert len(v['variants'])==28 and inv['total_families']==49
    assert cov['scoped_conditional_review_regimes']==998 and cov['scoped_conditional_review_executions']==189903
    assert cov['remaining_regime_reconciliation_count']==1084 and cov['remaining_execution_reconciliation_count']==97047
    assert cov['explicit_profile_insufficient_regimes']==3 and cov['geometry_only_other_role_regimes']==0
    assert cost['Markov_step_models'][0]['private_interface_budget_per_step_bytes']==6205488
    assert sum(x['groups']for x in cost['separate_draft_helper_collapse_models'])==104
    assert set(inv['modules'])=={'M15','M16','M17','M20','M24'}
    old=read(O/'module-phase-branch-matrix-v64.json')
    assert all(m.get('branch_schema') is None for m in old['modules'] if m['id']in inv['modules'])
    # Verify previous authority before ANY shared-state mutation, then protect every other module.
    for a in read(O/'checkpoint-manifest-v64.json')['artifacts']:
        q=Path(a['path']);b=q.read_bytes()
        assert len(b)==a['bytes'] and hashlib.sha256(b).hexdigest()==a['sha256'],q
    protected={m['id']:deepcopy(m)for m in old['modules']if m['id']not in inv['modules']}
    assert len(protected)==21
    write(O/'heads-sampling-protected-module-state-v65.json',dict(modules=protected,
        scope='All21othermodulematrixrecords,includingM06v64correction,M09/M18andframework/inputrows,mustremainidenticalafterv65.',overall_complete=False))
    e=['heads-sampling-regime-models-v65.json','heads-sampling-native-chains-v65.json',
       'heads-sampling-operator-contract-v65.json','heads-sampling-fusion-costs-v65.json',
       'heads-sampling-fusion-variants-v65.json','operator-regime-refinement-v65.json',
       'operator-analysis-coverage-ledger-v65.json','M15-M16-M17-M20-M24-operator-branch-inventory-v65.json',
       'heads-sampling-local-validation-history-v65.json','heads-sampling-design-refinement-history-v65.json',
       'heads-sampling-protected-module-state-v65.json']
    write(O/'PROFILE-CODE-QUESTION-GATE-v65.json',dict(
        status='OPERATOR_ONLY_HEAD_MARKOV_SAMPLING_STATE_FUSION_WORK_AND_DEPENDENCY_REVIEW',
        previous_turn_classification='PROGRESS:v64Index_context_typed_partition_and_quant_precision',
        questions=[
            'Whichcollapse/norm/headscoreinterfacesareprivateafterallnormalized/raw/prompt/confidencereaders?',
            'CanMarkovW2→BF16bias/Add→argmaxcompose,shard,moveownerormemoizewith5serialIDdependencies?',
            'Whatprecision/tie/NaN/collective/RNG/fullscoreconditionsboundgreedyorprobabilisticprotocols?',
            'Doeslazyheadreduceworkwithoutrepeatedweighttraffic,andsnapshot-safe17stagecorrectionpreservealiassemantics?'],
        allowed_source='11previously166SHA-boundlocaloperator/callerclausesonly. No newremote source. Noframework/API/HTTP/SSE/lifecycleaudit.',
        operator_goal='Canoperatorsfuse/split/reorder/reuseandwhatthetypedwork/IO/communication/dependencyboundsare. Frameworkcodeonlywhenneededforthespecificoperatorquestion.',
        local_pinned_operator_clause_review=True,new_remote_source_read=False,
        no_new_model_Torch_NPU_capture_health_process_kill_or_implementation=True,
        current_9p2_INT64_collective_mode_support_proved=False,overall_complete=False))
    scopes={
        'F26':'104target6TASKtailsplus104draft/helper4TASKcollapses;76NH/72NHprivateinterfacebudgets,rawsnapshot16NH,localSPnorm/live-snapshotcommtradeoff/all-readerrowunion/alreadyselectedheadrows.',
        'F30':'520Markovsteps/BF16barriers/24BV+24Bprivatebudget,owner-vocabshard/cold-firstcachelambda/warm0,confidenceinputseedembeddings/globalbudget,6xnaivecombinedwork/.75shardeddualaccumulator,versionedbiasmemo/certifiedblockbounds.',
        'F35':'192baseheadsand72greedychains,processedroundedglobalties/49bitfinitekeywith9.2INT64supportunboundandINT32AGfallback;distributedprobability/entropy/recoveryRNG,rawregenerationnet2TV/lazyheadweightwavepenalty/fullwrite-rejectionFillconditions.',
        'F31':'72statecorrection17TASKcalls215N→40NconditionalvaluebudgetwithwholeNoldvaluepreload. Explicitpermutationaliascounterexample,intwidth/error/outputprefix/validcountpredicateandasyncgenerationretained.'}
    updates={cid:dict(review_v65=dict(regime_models=str(O/e[0]),native_chains=str(O/e[1]),
        operator_contract=str(O/e[2]),work_costs=str(O/e[3]),variants=str(O/e[4]),coverage=str(O/e[6]),
        candidate_specific_scope=s,conditional_operator_analysis_deliverable=True,implementation=False,native_test=False,new_performance_result=False),
        profile_review_state='OPERATOR_WORK_IO_ROUNDING_RECURRENCE_SNAPSHOT_AND_PARTITION_CONDITIONS_REVIEWED',
        conclusion_changing_gap='Actualrow/flag/index/alias/value/RNG/weightandstate-generation,allactiveconsumers/changednativeprecision/loadedcollectivemodeandtypedphysicalcapacityremainconditions. Missingimplementationaloneisnotunfinishedconditionalanalysis.',
        next_profile_or_contract_action='NextremainingM23communicationandM21modelinput/index/metadataoperatorregimes. Codeonlyforconcretefusion/necessarywork/traffic/dependencyquestion;donotre-auditframeworkorpreviouslyansweredhead/samplerconsumercontracts.')for cid,s in scopes.items()}
    findings={
        'M15':('42regimes1040TASKs:104targettailsand104separatedraft/helpercollapses;collapse/RMS/BF16round/allreaderrowunion/SPcomm/rawsnapshotmodels. Notpairedbyshape.', ['F26','F30']),
        'M16':('17strictregimes904TASKsplus9ambiguousM04|M16regimes176TASKs:192baseheads520MarkovGEMMs,typedcoldweight/processedselection/globalbest/fallbackprobabilityandlayout/owner-shardmodels. Ambiguitypreserved.', ['F26','F30','F35']),
        'M17':('33regimes1144TASKs:72greedychains,partialwriterFill-1contract/rawclone/ID-onlyconditions/lazyweightwaves/distributedprobability-recoveryRNGandfullwritealternative.', ['F35','F31']),
        'M20':('30regimes5304TASKs:520source-ordered10TASKMarkovstepspluspreembeddingSliceand104draftRMS. BF16epilogue/recurrence/confidence/shard-owner/memo/certifiedboundanddynamicprefixmodels.', ['F30','F31']),
        'M24':('35regimes1224TASKs:72calls17stages,215Nvalue-reference→40NwithwholeNoldvaluespreloadedbeforestores;aliasracecounterexample. Crossmodelstate/cache/rollback/generationbranchesretainedin11semanticfamilies.', ['F31','F35'])}
    mods={mid:dict(finding=f,evidence=e[:8],candidates=cids,
        gap='Actualruntimeguards/validrows/maps/aliases/RNG/parameterandstategeneration/allreaders/nativeprecision/physicalcapacityremainexplicit;unobservedapplicableoperatorbranchesopen.',
        profile_scope='FixedsavedcasefunctionalTASKpopulationandoperator-onlyconditionalanalysis;newsemanticinventoryarchivesoldchronologicalrecords,notwholemoduleclosure.')for mid,(f,cids)in findings.items()}
    remaining=[]
    for s in read(O/'module-coverage-gaps-v64.json')['remaining']:
        if s.startswith('v64Index/Context'):continue
        if s.startswith('Finalnorm/head/sampling/DSparkoperatorchains:'):
            s='Finalnorm/head/sampling/DSparkv65:all166selectedregimes9792TASKshavenamedconditionalwork/valueIO/consumer/rounding/recurrence/statecontractsand28variants. Actualvalidrows/flags/NaN-ties/RNG/aliases/loadedcollectivemode/generations/physicalcapacityandallunobservedapplicablebranchesremainopen;implementationabsencealoneisnotanunfinishedconditionaldecision.'
        remaining.append(s)
    remaining.append('v65fullobservedcoverage998regimes189903TASKreviewed,1084regimes97047pending,0geometry-only. ThreeMemSet960stillprofileinsufficient;849missingPMU/16phase-unboundgraphs/3136ambiguousassignmentsincludingM04|M16nine176retained. 28head/Markov/sampling/stateoperatorchoicesand49semanticfamiliesin5modules;allother21modulematrixrecordsincludingcorrectedM06protected. NextM23collectiveoperatorpayload/phase/resourcesandM21modelinput/metadata/indexoperators;do notinferframeworkresearchfromlargecallerfiles. No wholemodule/wholegoalcompletion.')
    action=(
        '继续原Goal最新v65 PROGRESS。用户最新约束：代码只围绕算子能否融合、必要工作/访存/通信/依赖和理论上限；充分profile，缺证据才定点补代码，不研究框架。'
        '先读operator-analysis-coverage-ledger-v65、heads-sampling-regime-models-v65、heads-sampling-native-chains-v65、heads-sampling-operator-contract-v65、heads-sampling-fusion-costs-v65、heads-sampling-fusion-variants-v65及49familysemanticinventory。'
        '本轮166regimes9792exec=42M15/1040+17M16/904+33M17/1144+30M20/5304+35M24/1224+9M04|M16/176。全2082/286950现在998/189903显式条件review、1084/97047pending、0geometry，不是Goal完成百分比。'
        '104targettail6TASK与104draft/helpercollapse4TASK分开；76NH和72NH私有interfacebudget，rawHCcopy16NH/allreaderunion/promptguard/SPNone已有localcollapse，别重复发明。Headselection本就beforeGEMM。192baseGEMM+Transpose、520Markovsteps、72greedy和72statecalls已TASK绑定，不推actualrequest/RNG/flags。'
        '每Markovstep有4Slice：embedding前1+score后3；Gather至IDstore9M20+1M16共10TASK。Initialexpected2scoreSlice被assert拒绝/诊断同根因，修3后520全conserve，ERROR留证。W2BF16→postprocess→AddBF16→argmax保留；private24BV+24B预算，B2=6205488B，不适用fullscore分支。'
        'BaseW每卡165478400B，MarkovW2每卡66191360B；5step冷首轮cachelambda1..5，暖HBM可0，L2另计。Naivefullcombined(H+R)每卡F6倍，shardeddualaccumulator.75源自Markovshard，BF16分支舍入不能合并。Memo33.4266368GBreplica/4.1783296GBshard、8.557Tbuild，hit/capacity/nativeB1vsB2/B4precision未知；certifiedblockprune需native-error-outwardbound，baseTopKalone反例。'
        'GlobalfiniteFP32score/col可49bitkey，±0归一/NaNfallback/processedties/remap保留。不要假设INT64MAX合法：CANN9.0通用API支持但A3mode限制相冲，fixed9.2loadedmode未绑定。2INT32wordAG+localunsignedlexmax fallback56M ring发送vsfullVAG226240M，仅payload4040x不是speedup；2-stageFLOATMAX/INT32MIN两latencies也14M。Probability/epsentropy/recovery/RNG是额外协议。'
        '当前greedyTritonpartialwriter必须Fill-1 suffix/non-greedy；新fullwritekernel可折叠，不能直接删Fill。ID-only20TV预算要求所有raw/processor/prob/entropy/logprob观察者inactive或精确替代；fullraw laterregen6TV vsclone8TV仅净2TV。Lazyhead首拒F/6仍一整W，allaccepted6waves可能重载6W，weightcache/tiling不可省。'
        'State72×17TASK，两个dest都是INT32，validsummaryINT64；215N常规value→40N wholeNpreload条件。prevmap[1,0],old[10,20],valid[1,1]应[21,11]，逐行inplace会[21,22]；必须单program全读后写/同步或immutableoldbuffer。validID原谓词(x!=-1)&(x<V)，不是自动0<=x<V。'
        '11old166SHA-boundlocalsourcefiles复用，未连服务器/无newremote、无模型/Torch/NPU/capture/健康/进程/kill/实现。下一步M23通信算子和M21输入/metadata/index算子剩余regime，跨卡/跨迭代价差和上界继续做。35模型候选+7legacy框架候选，0整模块关闭；3MemSet960待定、849PMU、16unboundgraph、3136ambiguity保留；原Goal active不得因报告标记完成。')
    report='''本轮按算子目标分析输出头、Markov、采样和状态更新，共166组、9,792次原生执行。代码只补充舍入、消费者、循环依赖、写入范围和别名约束；本轮使用11份已绑定原166版本的本地源码，没有连接服务器或执行模型。

已有104条目标末端链、104条独立的草稿/辅助HC收缩、192次base词表投影、520个Markov步骤、72次贪心拒绝链和72次状态修正的算子身份对照。它们是profile中的调用实例，不代表实际请求、有效行、标志或RNG值已知。

| 方案 | 条件成本或上界 | 必须保留的边界 |
|---|---|---|
| HC收缩与RMS融合 | 保留原始HC快照时，目标尾部可减少76NH字节逻辑接口读写；单独辅助收缩为72NH | FP32加权与求和、收缩BF16舍入、完整H维RMS、最终BF16及所有内容读者 |
| Markov投影epilogue融合加法与argmax | 私有分数分支每步减少24BV+24B；B=2时约6.205MB | bias与base各自BF16、处理顺序、Add BF16、并列/NaN/映射及下一步ID依赖 |
| 词表分片后只交换最优项 | 两个INT32字组成的key AllGather，ring模型每卡发送56M字节，对比全词表226240M | 所有分数读者、处理后舍入值与并列规则；此4040倍仅是payload比 |
| 按需恢复raw FLOAT | 全raw随后仍要使用时，再生成6TV对比clone8TV，净预算为2TV | 未修改的BF16源、索引和代际；所需raw、熵、概率与logprob读者 |
| 状态修正17阶段融合 | 声明的常规值读写模型215N→40N，预算175N | 全部旧值读完后才能写，INT32计算/输出和INT64有效计数、错误与异步读者 |

N/M/B/T分别使用各段明确的物理行、头部行、请求批量和采样容量，不混用；H=5120，V=129280。表中的值是逻辑接口预算，不能直接当作HBM流量或端到端收益。计算项按每次MAC两次操作建模；376TF BF16、1.6TB/s只作为公开每die参考，实际时钟、缓存、Vector/INT64能力和通信时延仍未绑定。

原始HC快照仍有读者时，不能用本地收缩替代全部HC通信；它可能需要保留原HC gather并新增归一化结果gather。buffer为None时，本地收缩本来就已存在。目标LM head也已经在投影前选行；新增裁剪需要把选择推进到HC/norm之前，并包含全部内容读者的行集合。

Markov有五步串行ID依赖。W2按词表分片可减少每卡工作与权重，但每步要解决全局ID；单owner方案则仍承担完整串行计算并要交付其他卡需要的结果。把完整base与Markov拼成每步一个复制GEMM，会让当前每卡合计矩阵工作增加到六倍。保留双累加器及各自BF16边界的分片方案，计算比为0.75，这部分来自Markov分片，需要与保留批量base投影的方案比较。

Markov bias可以研究按token和权重/处理器/原生精度代际缓存。完整BF16转移表约33.43GB，按词表列分片约每卡4.178GB，预计算约8.557T操作；当前命中率、容量和摊销都未知，B=1的缓存生成也不能默认等于原B=2/4的原生舍入。仅按base TopK截断不能保证原结果；精确分块裁剪必须建立包含原生累加与舍入误差的保守上界。

有限FP32分数和列号可以构造49位排序key，保留首列并列规则。INT64 MAX的当前可用性不能只由API枚举确认：官方通用接口与A3展开模式存在不同约束，固定9.2的已加载模式尚未绑定。因此保留两个INT32字AllGather后本地比较，以及分两轮交换FLOAT最大值和INT32最小列号的替代方案；概率、熵和随机恢复还需要各自的全局协议。[HCCL通用接口](https://www.hiascend.com/doc_center/source/zh/canncommercial/900/API/hcclug/hcclcpp_07_0021.html)、[展开模式约束](https://www.hiascend.com/doc_center/source/zh/canncommercial/900/API/hcclug/hcclenvref_07_0009.html)。

贪心拒绝kernel当前只写有效前缀、首次拒绝位置及满足条件的bonus。原Fill为后缀及非贪心行提供-1；可以设计完整写入的新kernel来折叠初始化，不能直接删掉现有Fill。延迟head也有取舍：B=2、五个草稿的首次全拒绝示例可少算六分之五的行，但仍可能读取完整165.48MB权重；全接受则分成六轮串行head，最坏重复读六次权重并损失批量tiling。暖缓存与冷首轮应分别建模。

状态更新存在确定的别名反例：旧值[10,20]、计数[1,1]、重排[1,0]应得到[21,11]，逐行原地处理却会得到[21,22]。小批量单program要先加载全部旧值并同步后写；更大批量可用不可变旧缓冲区或双缓冲，并计入其成本。有效ID计数原谓词是(x!=-1)&(x<V)，不能未经有效域证明改成0<=x<V。

本轮形成28项条件方案或排除结论，并把5个模块的历史记录整理为49个语义分支。全观察台账现有998组、189,903次执行已做显式条件分析，另有1,084组、97,047次待核对。3个MemSet待定项、849条缺PMU、16个未绑定阶段图、3,136次模块归属歧义均保留。条件分析无需等待实现才成立，但这些观察和分支台账仍不证明全部模型路径、实际物理上限或整个Goal完成。
'''
    jobs=[
        dict(kind='profile',status='error',description='Retainresolvedlocalassertions:2Markovattemptsexpected2scoreSlicesratherthan3plus1preembedding;preflightexpected20othermodulesbutM00-M25total26leaves21. Allstoppedbeforecurrentcheckpoint/sharedmutations;correctedandverified. No targetfailure.',artifacts=[O/e[8]]),
        dict(kind='profile',status='pass',description='166functionalregimes9792uniqueTASKs/PMUcountsconserved;104targettails192baseheads520Markovsteps72greedy72statecalls. 9M04|M16regimes176ambiguityretained.',artifacts=[O/e[0],O/e[1],N/'analyze_heads_sampling_v65.py']),
        dict(kind='review',status='pass',description='11previous166SHA-boundlocaloperatorclausesreusedforrounding/recurrence/consumer/fullwrite/snapshotconstraints;primaryHCCLmodeevidencequalifiedbyversion,notfixed9.2supportproof.',artifacts=[O/e[2]]+[Path(x['path'])for x in contract['source_files']]),
        dict(kind='design-check',status='pass',description='28operatorchoiceswith104separatedhelpermodels,typedGEMM/cold-warmweight/communication/Markovmemo/lazyhead/statebudgets. 243finitepacked-ordercomparisons,processedtieandaliasracewitnessesPASS;no native/devicevalidation.',artifacts=[O/e[3],O/e[4],O/e[9],N/'model_heads_sampling_v65.py']),
        dict(kind='review',status='pass',description='998review189903exec+1084pending97047=2082/286950. 49semanticfamilies5legacyinventoriesarchived;all21othermodulematrixrecordsprotected. MemSet/globalambiguitiesretained;wholemodules0.',artifacts=[O/e[5],O/e[6],O/e[7],O/e[10],N/'consolidate_heads_sampling_v65.py'])]
    persist_refinement(version=65,previous=64,title='Operator-only head, Markov, sampling and snapshot-safe state fusion bounds',
        summary=dict(filename='heads-sampling-ceiling-summary-v65.json',
            previous_turn_classification='PROGRESS:v64Index_context_typed_partition_and_quant_precision',
            reviewed_regimes=166,reviewed_executions=9792,operator_decisions=28,semantic_families=49,
            target_tails=104,separate_draft_helper_collapses=104,base_heads=192,Markov_steps=520,greedy_rejections=72,state_correction_calls=72,
            observed_regimes=2082,scoped_reviewed_regimes=998,scoped_reviewed_executions=189903,
            geometry_only_regimes=0,geometry_only_executions=0,pending_regime_reconciliation=1084,pending_execution_reconciliation=97047,
            MemSet_explicit_pending_regimes=3,MemSet_explicit_pending_executions=960,
            ambiguous_M04_M16_regimes=9,ambiguous_M04_M16_executions=176,
            new_remote_source_read=False,local_pinned_operator_clause_review=True,implementation=False,native_test=False,new_performance_result=False,
            all_actual_physical_numeric_ceilings_proved=False,
            remote_job_status='No new remote job/connection. All work local saved-profile/pinnedsource/math;no model/Torch/NPU/newcapture/health/process/kill/implementation/backgroundanalysis.'),
        changes=['166regimes9792nativeexecutionsscopedconditionaloperatorreview;104targettailsand104separatedhelpercollapses,192baseheads,520Markovsteps,72greedyand72statecallsbound.',
            '28operatorchoicespriceHCnorm/Markovepilogue/globalbestprotocols/shard-owner/memo/precision/lazyheadweightwaves/rawregeneration/fullwriteandsnapshot-safe17stagecorrection.',
            '998regimes189903execreviewed,1084pending97047;49semanticfamilies5moduleswithall21otherrecordsprotected. Globaluncertaintiesand0wholemoduleclosureunchanged.'],
        candidate_updates=updates,framework_updates={},module_updates=mods,semantic_branches=inv['modules'],remaining=remaining,action=action,
        jobs=jobs,report=report,read_first=[e[6]]+e[:5]+[e[7]],publication_rows={},gate_updates=dict(overall_complete=False),
        scripts=[N/x for x in ['analyze_heads_sampling_v65.py','model_heads_sampling_v65.py','consolidate_heads_sampling_v65.py','verify_heads_sampling_v65.py']],
        executor_prefix='zcode-heads-sampling',main_script=Path(__file__),source_read=False)

if __name__=='__main__':main()
