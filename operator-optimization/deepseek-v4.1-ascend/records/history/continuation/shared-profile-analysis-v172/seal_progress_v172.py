"""Seal the fresh grouped module diagnosis; keep original fusion scope active."""
import json
import sys
from copy import deepcopy
from pathlib import Path
sys.path.insert(0,'/Users/wio/work/166-vllm-analysis-continuation-20261030')
from finalize_origin_coverage_v9 import N,O,read,write
from analyze_evidence_batches_v155 import pin
from persist_delta_checkpoint_v163 import persist_delta

H=Path(__file__).resolve().parent
old=read(O/'REVIEW-CHECKPOINT-v171.json')
prefix='grouped-module-diagnosis'
checks=read(H/'module-diagnosis-independent-checks.json')['checks']
receipt=read(H/'final-health-and-remote-artifacts.json')
checks.update(service_healthy_and_unrestarted=receipt['health_http']==200 and receipt['running'] and receipt['started_at']=='2026-11-02T19:01:54.649699535Z',all_analysis_handles_terminal=all(not x['same_process_live'] for x in receipt['analysis_handles']),no_new_device_work=receipt['new_application_kernel_model_replay_or_model_reload_calls']==0)
assert all(checks.values()),checks
write(O/(prefix+'-preflight-v172.json'),{'status':'PASS','preflight':True,'checks':checks})
write(O/(prefix+'-inputs-v172.json'),{'inputs':[pin(p) for p in sorted(H.rglob('*')) if p.is_file() and '__pycache__' not in p.parts],'overall_complete':False})

changes=[
'All32 v164 CPU+PMU inputs hashed and factorized once:288846 COMPUTE executions,285336 exact-containment PMU groups,28 metric fields,no duplicates or invalid values;183242 unique CPU connections and105604 absent creators retained.',
'Reused295/492 used frame symbols;indexed only197 new frames from34files by AST,zero unresolved/hash/name mismatches. Six sampling contract source hashes match exactly; no body rereads.',
'Exclusive functional ownership accounts201351 executions with87495 ambiguous retained;984 source/semantic differences are explicit communication-vs-caller partition,not data conflicts. ParentM19 children not double counted.',
'Two formal16-profile windows ranked separately. M12 lowest among12 measured MAC modules pooled(2.41549%),firstwindow(2.55250%),secondwindow(1.17269%);recorded-positive-cycle ambiguous membership bounds preserve both minima.',
'M23 is lowest reported Vector arithmetic activity among20 measured modules(0.01791% pooled);communication is useful data transport/polling,not zero utilization. No credible single global minimum across different resource metrics.',
'F21 accounts98.76493%/99.19250% of M12 cumulative native time and99.72133%/99.92787% of its reported AIC-cycle denominator. Causes separated into supported accounting and hypotheses;no speedup/physicalMFU claim.',
'All26 modules have measured/conditional/symbolic/host P5 routes. Ordinary shape GEMM work has guarded direct-dense/transpose/mode cases;FLOAT unresolved,clock1800MHz onlyreference,no crosscohort receiver/time join.',
'All32 raw blockDim keys decodeN/A;zero COMPUTE metadata duplicate IDs. Corrected diagnostics merge by decodedunknown,never used IDs as corecounts. Construction/parse errors preserved without re-running model.',
'Current frozen8requests/2seeds input hashes and exact client execution pin revalidated;explicittemperature0 source-domain branch contract reused while pre-return observers and real per-call flags remain distinct.',
'All CPU extraction/analysis jobs terminal;81remote outputs pinned. Original171 service health200/StartedAtunchanged;newkernel/model/replay/reload0.'
]
write(O/(prefix+'-summary-v172.json'),{'status':'PROGRESS','previous_goal_turn':'PROGRESS','overall_complete':False,'new_native_operand_or_phase_promotions':0,'new_whole_module_closures':0,'complete_module_metric_routes':26,'measured_MAC_modules':12,'measured_Vector_modules':20,'native_executions':288846,'PMU_groups':285336,'certain_owner_executions':201351,'ambiguous_owner_executions':87495,'measured_MAC_minimum':'M12','global_physical_MFU_minimum':None,'protocol_does_not_require_global_physical_MFU':True,'new_application_kernel_model_replay_or_model_reload_calls':0})

action=(
'继续完整目标，checkpoint172。直接读shared-profile-analysis-v172/module-diagnosis-question-review、all26-module-metric-and-P5-routes、lowest-module-cause-review、fixed-case-branch-contract-reuse和coverage-work-queue-v172。不要重读已索引source或重复32DB全量导出。'
'统一模块诊断在已声明的32个v164 CPU+PMU窗口中已有明确结果：12个MAC-bearing模块M12最低，pooled2.41549%，两个16rank窗口2.55250%/1.17269%；201351certain、87495ambiguous的全部执行保留，记录的positive-cycle归属边界下最低不变。358语义行的原始融合扫描完整性是另一项任务，不能把当前模块指标结果冒充整图完成。'
'协议MODULE_DIAGNOSIS允许分资源组/已测子集排名并说不存在可信全体最低值，不要求全26模块物理MFU。因此不要把fmax、全局MFU或私有MemSet参数包设为所有分析统一前置。20个Vector模块中M23算术活动比例最低是通信角色，不能说无用/零工作。26模块均已有P5数值条件/符号/host路由。'
'M12内F21占98.76%/99.19%累计计算时长、99.72%/99.93%reportedAICcycles；F21现有路径已融合dispatch/quant/GMM1/SwiGLU/GMM2/return，不把已有融合再算机会。重点审计跨卡到达、receiver不均、合法拆分重组及跨边界控制，而非逐个周边小算子探针。不能从低ratio量化等待/可达提速；CPU+PMU扰动和缺57个laterF21PMU保留。'
'下一步一次性把原26模块358语义行按算子问题归为已分析条件域、固定Case适用性未知、真正未分析三类，沿profile角色/既有合同聚合同一guard问题，输出最小决定性问题集。条件答案本身是分析成果；缺优化实现、无可达提速或没展开每条私有ABI不是自动未分析。也不能以空白推出独立。'
'复用当前输入/client哈希和六份完全相同的采样source合同，显式temperature0不是truthiness默认；非空有效当前metadata和封闭Case请求集合下的all-greedy分支结论条件成立。entropy/synthetic/raw/logprob等前置reader不能由temperature单独消除。实际profile的Markov/greedy调用是正证，不等于所有flags。若具体候选确需flags/receiver/readers，再用已授权容器做一次合并小探针，不再逐项问许可或为记账重采完整Case。'
'随后审计一个覆盖跨算子/层/卡/迭代的非重复组合图，区分互斥、共享、叠加与等待/存活期条件，并核对必要工作/资源max和生命周期，禁止笛卡尔配对。原35+7候选/26模块/358语义行全部保留；别把全局callordinal或物理MFU推广成所有候选必须条件。'
'本轮全为CPU SQL/AST/hash及健康查询，主服务StartedAt仍2026-11-02T19:01:54.649699535Z、health200。整体ACTIVE/PROGRESS，仍有实际范围/分支/组合审计未完成。')

gate={'version':172,'parent':pin(Path(old['source_gate'])),'policy':'PROFILE_FIRST_AND_CACHED_CONTRACT_REUSE','current_questions_review':str(H/'module-diagnosis-question-review.json'),'new_source_scope':'197 new exact used frames indexed structurally; only merged-draft classification-changing source lines read. Six existing sampling sources hashed only, bodies not reread.','same_case_no_new_device_work':True,'next_source_gate':'Only a named operator/guard question that changes a selected fusion, decomposition, necessity or bound result; inspect profile and cached contracts first.','no_cartesian_pairs':True,'no_universal_private_ABI_or_global_MFU_gate':True,'overall_complete':False}
overlay={'version':172,'parent_checkpoint':pin(O/'REVIEW-CHECKPOINT-v171.json'),'changes':changes,'module_metrics':str(H/'all26-module-metric-and-P5-routes.json'),'measured_lowest_and_causes':str(H/'lowest-module-cause-review.json'),'exact_case_contract':str(H/'fixed-case-branch-contract-reuse.json'),'independent_checks':str(H/'module-diagnosis-independent-checks.json'),'protocol_scope':'Grouped reported-counter minima satisfy the measured-population diagnosis question; unmeasured/ambiguous/missing pieces explicit. A universal physical MFU is not a hidden acceptance requirement. Fusion coverage remains separate.','source_conditional_answers_are_analysis_results_not_unfinished_implementations':True,'new_whole_module_fusion_scan_closures':0,'overall_complete':False}
queue=deepcopy(read(O/'coverage-work-queue-v171.json'))
queue.update(version=172,parent=pin(O/'coverage-work-queue-v171.json'),new_evidence=changes)
queue['work']=[
{'id':'Q172-FIXEDCASE-BRANCH-AND-COMPOSITION-AUDIT','state':'NEXT_DECISION_CHANGING_SCOPE_AUDIT','action':'Classify all358 semantic rows once by reviewed conditional answer,unknown fixedCase applicability,and genuinely unanalysed operator scope. Aggregate shared guard questions from existing evidence; no fullsource reread or Cartesian pairs. Audit one nonduplicate joint candidate graph across op/layer/card/iteration and only then choose one combined argument probe if needed.','evidence':[str(H/'module-diagnosis-question-review.json'),str(H/'all26-module-metric-and-P5-routes.json'),str(H/'fixed-case-branch-contract-reuse.json')],'remaining':'Actual branch/variant scope and composition dependencies not exhausted; conditional source answers already count as analysis where their guards are stated.'},
{'id':'Q172-MODULE-DIAGNOSIS','state':'RECORDED_GROUPED_METRIC_QUESTION_ANSWERED_WITH_EXPLICIT_COVERAGE','action':'Reuse current12MAC/20Vector ranks and26P5 routes. M12 lowestinbothwindows robustunderrecordedambiguity;F21dominatesM12. Do not use global physicalMFU as a new completion gate. Only obtain additional PMU/module data if it changes the declared analysis question.','evidence':[str(H/'module-activity-review.json'),str(H/'module-window-diagnosis.json'),str(H/'lowest-module-cause-review.json')],'remaining':'No unrecorded/missing-PMU/production-MFU claim or physicalcausal time split. Fusion scope is distinct.'}
]+[x for x in queue['work'] if x['id']!='Q171-UNIFORM-MODULE-DIAGNOSIS']

base=read(Path(old['coverage_obligations']))['obligations'];recs={}
for x in base:
    recs[x['id']]={'current_evidence_facets':['E0','E1','E8'],'current_dispatch_policy':'Named operator semantic/guard/resource question; reuse profile and identical cached source contracts. NoCartesianpair or globalphysicalMFU/privateABI gates.','current_v172_review':str(H/'module-diagnosis-question-review.json'),'whole_domain_closed_by_this_increment':False,'original_broad_scope_and_history_preserved':True}
recs['G12'].update(protocol_question_answered_for_declared_recorded_population=True,protocol_does_not_require_universal_physical_MFU=True,measured_MAC_modules=12,measured_Vector_modules=20,all26_metric_and_P5_routes=str(H/'all26-module-metric-and-P5-routes.json'),remaining='Missing and unrecorded work remain explicitly outside measured-group ranks; no global physical MFU inference. Broader fusion/branch coverage is not a module-metric requirement.')
for k in ['G02','G06','G13']:
    recs[k]['new_case_branch_contract']=str(H/'fixed-case-branch-contract-reuse.json')
for k in ['G07','G08']:
    recs[k]['new_all26_P5_comparison_routes']=str(H/'all26-module-metric-and-P5-routes.json')
recs['G11']['new_supported_vs_hypothesis_diagnosis']=str(H/'lowest-module-cause-review.json')

report='''当前整体为 PROGRESS，尚未完成原目标。此次完成的是同一批已存 profile 的模块诊断、理论比较路由和决定性分支证据复用，没有把成稿当作整图完成。

32 份 v164 CPU＋PMU 数据库只读汇总了 288,846 条计算执行和 285,336 组 PMU；按相同任务 ID 与唯一包含时间区间关联，未发现重复或非法指标值。183,242 条有唯一 CPU creator，105,604 条没有；没有连接并不等于独立或无工作。最终按功能叶子归属确定 201,351 条，87,495 条保持歧义。M19 的已知 HC、MoE、attention 子模块不在父项重复计数；通信统一计入 M23，984 处调用者与通信功能归属差异均已核对。

主指标为 Σ(报告的 engine total cycles × pipe ratio)/Σ(engine total cycles)。分子分母、单位、有效样本、每个设备/导出窗口和歧义范围均保存。它是 CANN 报告周期加权的管线活动比例，不是 MFU、整机 busy 或剩余提速空间。协议允许按可比资源组给出“已测模块中最低”，并说明不存在可信的全体最低值；不额外要求所有模块都获得物理 MFU。

| 已存正式窗口 | 计算执行 | PMU 组 | 已测 MAC 模块最低 | MAC 活动比例 |
| --- | ---: | ---: | --- | ---: |
| 20261102152817，16 个 profile | 153,354 | 152,449 | M12 路由 MoE | 2.55250% |
| 20261102152835，16 个 profile | 135,492 | 132,887 | M12 路由 MoE | 1.17269% |
| 合并同口径 | 288,846 | 285,336 | M12，12 个已测 MAC 模块中最低 | 2.41549% |

两个窗口分别把所有已记录、正周期、归属可选的执行纳入保守成员边界后，最低者仍是 M12。此边界不包含缺失 PMU 或未记录工作，不能扩大成全部真实运行的排序。后窗另有 57 个 F21 执行缺 PMU。两个导出是同一次正式负载的窗口，不是启动模型加载窗口；导出时间簇也不保证其中只有一种语义阶段。前窗还包含运行时 execute_dummy_batch 的正向调用证据，这类工作可能接收远端专家行，不能按空 sender/no-op 删除。

F21 在前后窗分别占 M12 累计 native 时长的 98.76493%／99.19250%，占报告 AIC 周期分母的 99.72133%／99.92787%。前窗 2,408 个 F21、累计 26.4620s；后窗 2,064 个、累计 2.89553s。这是跨设备执行时长相加，不是请求墙钟时间。低活动主要位于已经融合的 dispatch＋量化＋GMM1＋SwiGLU＋GMM2＋return 内部，而非仅由周围小算子造成。不能把这组现有融合再次登记为新收益。

小 receiver tile、专家/卡间不均、到达与协议等待、host 延迟和 CPU＋PMU 采集扰动是竞争解释。现有源合同及另一计数 cohort 支持其机制，但不把另一 cohort 的 receiver work 与当前 PMU 时间硬配，也不分配未经测量的因果时间。F21 全区间算术及 MTE 活动均偏低，不支持“全区间 HBM 饱和”这一单一解释；短阶段带宽限制仍可能。低 ratio 不能推出有多少时间可移除、哪次融合可达多少倍。

20 个已测 Vector 模块中，M23 的 Vector 算术活动比例最低，合并值 0.01791%。该模块承担通信、分区和轮询，标量/MTE 活动明显；少做 Vector 算术不是没有有效数据传输。不能将通信、Cube、Vector 与 host 模块拼成一个可信全体利用率最低值，也不能把 MTE ratio 当 HBM/链路带宽。

26 个模块均已生成 P5 对比行：有确定归属的普通 GEMM 用同一执行群的 shape 条件 MAC、周期下界和匹配累计时间；其余保留必要工作/容量/存储初态/网络 cut 的符号表达式或 host 适用范围。未把缺证据写零。普通 shape 推导明确限制 A 的转置/批维解释、direct-dense 算法和真实模式；FLOAT 保留 HF32-off/on 情况。24 Cube 核、arch2201 可复用，1800MHz 仅额定参考，非物理 fmax。不同资源取 max，不能把独立算子的下界直接相加当融合子图下界。receiver-domain 的 F21 工作仍保留 Σm_e×35389440 的条件式；当前 PMU 没有该同调用计数，不编造数值 MFU。

492 个实际使用的项目调用帧中，295 个直接复用已有索引，仅新增 197 个帧、34 份文件的 AST 结构索引，未发现未解析、符号或旧 hash 冲突。只有会改变 Markov 归属的少量新调用点检查了源码行。初版诊断使用了 blockDim 原始 STRING_IDS 键作分组标签；随后 32 库全部解码为 N/A，已合并修正，没有把这些键当 core 数或用于容量推断。COMPUTE 元数据 ID 无重复。构造错误和一次执行前 SyntaxError 保留，未因此重跑模型。

冻结的 8 个请求和 2 个 seed 文件 hash、实际 client 执行 pin 已复核，client 显式 temperature=0、ignore_eos=True。6 份参数/请求集/拒绝采样相关源码 hash 与已有合同完全相同，直接复用原分支推导。合法、非空、当前 metadata 且全部请求来自该 Case 的条件下可用 all-greedy 不变量；不能由温度推断 entropy/synthetic/raw/logprob 等前置观察者均关闭，真实 per-call flag 也未冒称已观测。独立公开 argmax 替换可保留其他 score 观察者，不必为更强的全 score 死数据论证卡住它。

服务 health200，StartedAt 未变；本轮仅 CPU SQL/AST/文件 hash 与只读健康检查，全部分析进程终止，未新建应用 NPU 负载或重载模型。81 份远端派生资料已 pin。

下一步集中审计原 26 模块 358 语义行的条件域、固定 Case 适用性和真正未分析范围，再审一个覆盖跨算子、层、卡、迭代的非重复组合图。原 35＋7 候选全部保留；条件分析本身是结果，缺实现或无法给出可达提速不是额外的分析失败门槛。仍会改变所选候选结论的 flags、receiver/readers/state 才进入一次合并探针。整体 ACTIVE，未宣告全部完成。
'''
jobs=[
{'kind':'review','status':'pass','description':'Factorized all32samecohort DBs:288846native executions,285336exactPMUgroups,28fields;201351certain/87495ambiguous retained;only197newusedASTframesindexed.','artifacts':[H/'extraction-observation.json',H/'profile-frame-symbol-delta.json',H/'module-activity-review.json']},
{'kind':'review','status':'pass','description':'Grouped module ranking:12MAC/20Vector;M12lowestboth16rankwindows robustrecordedownershipambiguity. All26P5routes saved withsymbolic/conditional/hostdomains;no universalphysicalMFUgate.','artifacts':[H/'module-window-diagnosis.json',H/'all26-module-metric-and-P5-routes.json',H/'lowest-module-cause-review.json',H/'module-diagnosis-independent-checks.json']},
{'kind':'review','status':'error','description':'Preserve local submissionformatting and correctionSyntaxError;rawblockDimlabelgrouping correctedtoN/A afterall32FKdecode. Primaryweightedmetricsunchanged;norepeatedmodelrun.','artifacts':[H/'block-dimension-decode.json',H/'dimension-correction-observation.json',H/'dimension-correction-final-observation.json',H/'module-window-diagnosis-raw-dimension-first.json']},
{'kind':'review','status':'pass','description':'CurrentfrozenCase/clientpinsandall6existing sampling sourcecontracthashes match;reuse conditionalgreedy-domainanalysiswithoutsourcebodyrereads ornewinference.','artifacts':[H/'fixed-case-request-semantics.json',H/'fixed-case-client-request-contract.json',H/'sampling-source-reuse-fingerprint.json',H/'fixed-case-branch-contract-reuse.json']},
{'kind':'review','status':'pass','description':'Original171servicehealthy/unchangedStartedAt,allCPUanalysisprocessesterminal,81remoteartifacts pinned;newapplicationdevicework/modelreload0.','artifacts':[H/'final-health-and-remote-artifacts.json']}
]
persist_delta(version=172,previous=171,prefix=prefix,changes=changes,report=report,action=action,
remaining_delta='G172:Grouped module diagnosis answered forrecordedpopulation,all26P5routesreviewed. No globalphysicalMFUrequirement. Full358-row actualbranch/variant andnonduplicatecrosscandidatecompositionaudit stillneeded;existingconditionaloperatoranswerscountwithoutimplementation/privateABIgate.',
source_gate=gate,operator_overlay=overlay,queue=queue,jobs=jobs,scripts=list(H.glob('*.py')),obligation_reconciliations=recs,
remote_status='AllCPUextraction/classification/windowjobs terminal;81remoteoutputs pinned. Servicehealth200/StartedAtunchanged;newapplicationNPUwork/reload0. Fullgoalactive.')
manifest=read(O/'checkpoint-manifest-v172.json')
assert all(pin(x['path'])==x for x in manifest['artifacts'])
post={'status':'PASS','checkpoint':172,'checks':checks,'manifest_artifacts':len(manifest['artifacts']),'all_manifest_pins_verified_after_seal':True,'predecessor_pins_verified_before_shared_writes':True,'complete_original_scope_preserved':True,'overall_complete':False}
write(O/(prefix+'-postcheck-result-v172.json'),post);write(N/'CURRENT-POSTCHECK.json',post)
write(N/'LIVE-SHARED-COHORT.json',{'effective_checkpoint':172,'status':'CPU_ANALYSIS_AND_CURRENT_HEALTH_TERMINAL','receipt':str(H/'final-health-and-remote-artifacts.json'),'no_live_device_or_analysis_job':True,'overall_complete':False})
write(N/'NEXT-EVIDENCE-PENDING.json',{'effective_checkpoint':172,'status':'FIXEDCASE_BRANCH_AND_JOINT_COMPOSITION_AUDIT_NEXT','next_action':action,'work_queue':str(O/'coverage-work-queue-v172.json'),'overall_complete':False})
print(json.dumps(post,ensure_ascii=False))
