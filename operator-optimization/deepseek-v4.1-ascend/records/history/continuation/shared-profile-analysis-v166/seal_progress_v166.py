"""One evidence delta: actual NPU-only capture,graph bridges and typed neighborhoods."""
import json,sys
from copy import deepcopy
from pathlib import Path
sys.path.insert(0,'/Users/wio/work/166-vllm-analysis-continuation-20261030')
from finalize_origin_coverage_v9 import N,O,read,write
from analyze_evidence_batches_v155 import pin
from persist_delta_checkpoint_v163 import persist_delta
H=Path(__file__).resolve().parent;old=read(O/'REVIEW-CHECKPOINT-v165.json')
capture=read(H/'device-only-evidence-review.json');runtime=read(H/'new-profile-runtime-review.json')
rank=read(H/'ALL32-NPU-ONLY-RANKING.json');bridge=read(H/'replay-bridge-summary.json')
env=read(H/'native-envelope-summary.json');comm=read(H/'typed-communication-neighborhoods.json')
last=read(H/'final-runtime-and-derivation-receipt.json');work=read(H/'decode-MoE-conditional-work-polytope.json')
checks={
 'same_full_case_valid':capture['case_valid'] and capture['exact_response_text_matches']==8,
 'original_service_restored':runtime['controller']['state']=='FAILED_ORIGINAL_SERVICE_RESTORED' and last['current_health_http']==200,
 'original_script_restored':last['current_serve_sha256']=='c297201185b5676254af4d7693279f89bfa60f9c76eb340058a78ecf730b7bba',
 'constructor_reuse_verified':capture['constructor_reuse']=={'worker_count':16,'constructors_per_worker':[1],'profiles_per_worker':[2],'same_worker_set':True},
 'all32_actual_NPU_only_configs':len(runtime['profile_configurations'])==32 and all(x['info']['config']['common_config']['activities']==['ProfilerActivity.NPU'] and x['info']['config']['experimental_config']['_aic_metrics']=='ACL_AICORE_NONE' for x in runtime['profile_configurations']),
 'actual_CPU_and_PMU_absent':capture['CPU_op_range_total_bytes']==0 and capture['CPU_and_PMU_DB_rows']=={'PYTORCH_API':0,'TASK_PMU_INFO':0},
 'all32_vendor_DBs_ranked':len(rank['profiles'])==32 and capture['new_compute_executions']==966674,
 'raw_copy_verified':read(H/'device-only-INPUT-MANIFEST.json')['original_and_copy_verified_equal'],
 'old_cohort_actual_replay_bridge':bridge['bridges']==64 and bridge['unique_processes']==16 and bridge['scope_end_statuses']=={'returned':64} and bridge['native_captured_compute_instances']=={'unique':105600},
 'new_native_envelopes_complete_in_scope':env['envelope_count']==288 and env['assignment']=={'unique':475200} and env['issues']==0 and env['F21_count_per_envelope']=={'40':288},
 'typed_communication_anchors':comm['anchors']==424,
 'conditional_work_bound_scope_explicit':work['condition_application_status'].startswith('Mathematical conditional') and work['consequences_under_premises']['empty_experts_min']==96,
 'no_history_or_whole_closure':rank['overall_complete'] is False and work['overall_complete'] is False,
}
assert all(checks.values()),checks
write(O/'device-only-shared-preflight-v166.json',{'status':'PASS','preflight':True,'checks':checks,'basis':'Actual raw/configuration/DB/correlation evidence,not synthetic device tests.'})
inputs=[p for p in H.iterdir() if p.is_file()]
write(O/'device-only-shared-inputs-v166.json',{'inputs':[pin(p) for p in sorted(inputs)],'overall_complete':False})
summary={'status':'PROGRESS','previous_goal_turn':'PROGRESS','overall_complete':False,
 'new_native_operand_or_phase_promotions':0,'new_whole_module_closures':0,
 'NPU_only_capture':capture,'same_case_counters':read(H/'new-native-schema-and-case.json')['case']['counter_deltas'],
 'new171_compute_executions':966674,'new171_native_envelopes':288,
 'new171_envelope_compute_assignment':{'unique':475200},'old_shared171_replay_native_model_bridges':64,
 'old_shared171_graph_compute_assignment':{'unique':105600},
 'native_name_ranking':rank['native_name_ranking'],
 'communication_dtype_correction':'COMMUNICATION_OP.dataType uses ENUM_HCCL_DATA_TYPE;v165 STRING_IDS projection invalid. Frozen v165 data retained;no old physical byte claim depended on it.',
 'typed_communication_CPU_anchors':424,'conditional_MoE_work_polytope':str(H/'decode-MoE-conditional-work-polytope.json'),
 'failed_controller_preserved':True,'live_reusable_profile_capability':False,
 'historical_native_or_phase_promotions':0,'new_independence_proofs':0,'whole_modules_closed':0}
write(O/'device-only-shared-summary-v166.json',summary)
action=('继续全部模块/条件分支和跨算子/层/卡/迭代原目标。上一轮v165真实共享profile与恢复是PROGRESS，本轮真实NPU-only采集、32DB/966674执行、64条Python-replay到native模型的关系和288个新native完成范围也是PROGRESS，整体未完成。'
 '直接读shared-profile-analysis-v166的device-only-evidence-review、ALL32-NPU-ONLY-RANKING、NATIVE-ENVELOPES、REPLAY-MODEL-BRIDGE与typed-communication-neighborhoods，不重复采集本批数据或重复旧CPU测试。'
 '原采集完整成功且8输出相同；控制器误将16个复用两次的profiler对象要求为32个，后置检查触发额外恢复。错误保留，原脚本与服务已恢复，当前没有保留可复用live profiler能力。不要因为该后置错误重跑成功Case。'
 'F21在新窗口累计任务工作时长占42.44%，仍为第一；旧68.28%和新窗口不能作生产加速比。优先复用现有F21原生协议/条件MAC合同，细分实算/通信/等待及小行数/专家负载策略；现有profile不能定案时只定点补算子代码或必要元数据。'
 '旧171 cohort已有64个同线程replay范围严格包含RuntimeAPI，再经官方connectionID绑定MODEL_EXECUTE/MODEL_WAIT_COMPLETE，105600图内计算执行唯一归属；只是本域graph身份/执行范围，不是每个捕获算子的源位置、参数或全部reader，也不回填169/166历史。'
 '新NPU-only cohort已有288个原生模型完成范围、475200图内计算执行，每范围40个DispatchFFNCombine；没有CPU/source绑定，不按model49同号跨进程或跨cohort串接。'
 '通信424个实际CPU栈邻域区分SP输入/最终HC、Engram和DSpark，按原F26/F32/F24等既有条件变体复用，避免重查已分析代码或新增笛卡尔配对。dtype已修正为HCCL专用enum，group_name_N仅本地别名，count非wire字节。'
 '针对本地M3/24expert/K6的形状，条件工作多面体给出同一EP16交换且所有rank<=3行时N<=48、assignment<=288、至少96空expert；不能把单rank形状套到混合prefill/decode交换，不能由空expert删除同步协议，不能当25%时间收益。'
 '保持35+7候选、26模块358行及G00-G14全范围；原9/1352、849PMU、16phase、3136归属、copy/control、数值/读者/物理容量与最低MFU仍open。当前授权覆盖自有容器推理/框架/探针操作，不重复询问。')
gate={'version':166,'parent':pin(Path(old['source_gate'])),'policy':'Profile-first,operator-directed,shared-neighborhood evidence;no Cartesian pair sweep or repeated framework reads.',
 'actual_answers':[
 {'question':'Can device timing be obtained without the previous CPU stack/memory/PMU collectors?','answer':'Yes:all32NPU-only profiles,zeroCPU/PMUrows,966674compute executions,8matchingoutputs;controller accounting error afterward separately preserved.'},
 {'question':'Must every captured task have its own CPU connection before graph membership can be known?','answer':'No:exact same-thread CPU nesting plus vendor launch/wait connections bind64replays and105600child executions in the old171cohort. No nearest-time or operand inference.'},
 {'question':'Can local exported group and datatype labels be used directly across traces?','answer':'No:HCCLdatatype has its own enum;group_name_N is local.424actual source-neighborhood anchors are retained without global communicator claims.'}],
 'next_targeted_questions':[
 {'id':'Q166-F21-INTERNAL','question':'Which internal MoE readiness,group-row schedule or protocol boundary limits this small-row captured domain?','allowed':'Reuse existing operator/native contract first;only targeted operator code/metadata if profiles cannot decide.'},
 {'id':'Q166-COMM-REORDER','question':'Which already-modeled F26/F32/F24/DSpark reorderings remove actually demanded materialization/communication?','allowed':'Reuse the typed CPU source neighborhoods and old candidate conditions;supplement only the missing consumer/readiness/value premise.'}],
 'current_container_process_authorization':'Explicit broad user authorization;no repeated per-probe approval within owned container.',
 'all_missing_fields_remain_UNKNOWN':True,'no_new_independence_or_whole_closure':True,'overall_complete':False}
overlay={'version':166,'parent':pin(Path(old['evidence_overlays'][-1])),
 'affected_candidates':['F21','F24','F26','F27','F31','F32','F33'],'affected_modules':['M07','M11','M12','M14','M15','M18','M21','M22','M23','M24'],
 'actual_NPU_timing':pin(H/'ALL32-NPU-ONLY-RANKING.json'),'actual_replay_bridge':pin(H/'REPLAY-MODEL-BRIDGE-v166.json'),
 'native_envelopes':pin(H/'NATIVE-ENVELOPES-v166.json'),'typed_CPU_neighborhoods':pin(H/'typed-communication-neighborhoods.json'),
 'conditional_work_polytope':pin(H/'decode-MoE-conditional-work-polytope.json'),
 'new_graph_identity_relations_in_new171_only':64,'new_native_completion_envelopes_in_NPU_only_cohort':288,
 'new_pair_clauses':0,'new_independence_proofs':0,'new_native_operand_or_phase_promotions':0,'historical_native_or_phase_promotions':0,
 'new_whole_module_closures':0,'whole_modules_closed':0,'overall_complete':False}
queue=deepcopy(read(Path(old['work_queue'])));queue.update(version=166,parent=pin(Path(old['work_queue'])))
for row in queue['work']:
 if row['id']=='Q165-MOE-TIMING':
  row.update(state='NPU_ONLY_TIMING_AND_EXPLICIT_NATIVE_ENVELOPES_AVAILABLE',action='Reuse966674NPU-only executions and288native envelopes plus existing F21 internal protocol/MAC contracts. Focus small-row receiver/work/readiness mechanisms;no rerun because of the post-capture accounting assertion.',remaining='Actual receiver rows/stage times/physical capacities and legal reordering proof open;recorded42.44%work is not a production bound.')
  row['evidence'] += [str(H/'NATIVE-ENVELOPES-v166.json'),str(H/'decode-MoE-conditional-work-polytope.json')]
 elif row['id']=='Q155-GRAPH-CUTS':
  row.update(state='ACTUAL64_REPLAY_TO_NATIVE_MODEL_RELATIONS_BOUND_NEW171_ONLY',action='Use exact marker/supplement equality,same-thread API nesting and vendor launch/wait identity to group105600recorded graph executions;unknown per-op creator/operand/reader premises remain open.',remaining='Graph membership is not per-operator source/all-reader/capture-update operand identity;no original169phase backfill.')
  row['evidence'] += [str(H/'REPLAY-MODEL-BRIDGE-v166.json')]
queue['work'].append({'id':'Q166-COMM-NEIGHBORHOODS','state':'424_EXACT_TYPED_SOURCE_NEIGHBORHOODS',
 'action':'Apply already-modeled F26HC-local-collapse/F32project-before-gather/F24Engram/DSpark conditions to actual recorded source neighborhoods. Reuse old proofs;new code only for a conclusion-changing missing premise.',
 'remaining':'All demanded consumers/row ownership/numerics/readiness and global communicator incarnations remain open;no wire-byte inference from counts.',
 'evidence':[str(H/'typed-communication-neighborhoods.json')]})
queue['new_evidence']=['Actual32NPU-only profiles,966674compute tasks,zeroCPU/PMUrows','64old171replay/native model relations and105600unique child memberships',
 '288newnative completion envelopes with40MoE kernels each','424typed communication source neighborhoods','Conditional small-row MoE work polytope;scope/error/service restoration explicit']
recs=deepcopy(read(O/'current-obligation-reconciliations-v165.json')['reconciliations'])
for key,row in recs.items():row.update(v166_evidence_delta={'operator_overlay':str(O/'device-only-shared-operator-overlay-v166.json'),
 'scope':'Actual new171 timing/replay identity/typed communication neighborhoods refine routing only;all original requirements and historical gaps retained.'},whole_domain_closed=False,original_broad_remaining_and_history_preserved=True)
write(O/'current-obligation-reconciliations-v166.json',{'parent':pin(O/'current-obligation-reconciliations-v165.json'),'reconciliations':recs,'does_not_delete_or_narrow_original_requirements':True,'overall_complete':False})
report='''# v166：设备时间、replay身份与算子工作边界

本轮实际完成同一C8、32768输入/1024输出、P90负载的32份NPU-only窗口；8个请求文本与原参照逐字一致，实际前缀计数235904/262144，8192输出token，无抢占。原服务健康检查200，启动脚本恢复原SHA。全部32份元数据为NPU活动、Level0、无CPU栈/内存/PMU；原始torch.op_range总字节0，数据库CPU与PMU行数0。官方离线导出全部返回，966674条计算执行已按实际native name统计。

采集成功后的控制器检查写错：16个profiler对象各复用两次，产生32份profile，检查却要求32个构造对象，触发多余恢复。原失败记录保留，独立复核按每worker1构造/2profile确认数据有效。实际发生一次计划重载和一次自动回退重载，当前没有留下live可复用profile入口。没有重采成功Case，也不把控制器错误改写成PASS。

DispatchFFNCombine仍为累计任务工作时长第一，占新窗口42.44%。Level0缺失opType而保留native name，因此按记录的native name排序；AivKernel不臆拆成多个通信family。这个比例与旧68.28%来自不同采集方式、窗口长度和动态工作状态，不能当生产延迟占比、优化收益或物理MFU。

现有上一批171 profile自身补出64条强关联：同线程RuntimeAPI完整嵌入唯一replay marker；marker名称与补充process UUID/graph token/capture epoch精确一致；官方connectionID再连到同model/device/pid/stream的MODEL_EXECUTE及唯一MODEL_WAIT_COMPLETE。64个scope均正常返回，105600条图内计算执行唯一落入相应原生完成范围，没有最近时间补配。这里只新增本域图身份/执行范围，不是每个captured算子的源位置、参数或全部读者，也不回填旧169/166历史。

新NPU-only域另有288个原生模型完成范围、475200条图内计算执行，每范围40个DispatchFFNCombine。范围时长中位数37.03ms，最小30.44ms、最大50.45ms；这是本域原生记录，不是整个请求关键路径。此域没有Python图对象记录，不以相同model49跨cohort串接。

修正了一个通信投影错误：COMMUNICATION_OP.dataType必须通过ENUM_HCCL_DATA_TYPE解释，旧STRING_IDS投影无效。新结果有14704条typed通信记录和424个唯一CPU连接/调用栈邻域，分开SP输入、最终HC、Engram及DSpark通信。group_name_N只是本地导出别名，不能凭同号当全局communicator；API count也不是wire字节。现有F26已经包含本地collapse/norm后再gather等条件变体，F32已包含投影前移，直接复用这些分析，不新增重复候选或笛卡尔配对。

对记录的3行/24本地expert/K6形状建立了条件工作多面体：仅在同一语义EP16交换的所有参与者均不超过3行、24expert分区和不重复TopK6合同成立时，全球有效行不超过48、assignment不超过288、384个expert中至少96个为空。单expert最多48接收行；直接两GMM每assignment35389440MAC。一般混合prefill/decode必须用N_valid<=sum_r M_r，不能把单rank的M3套到全组。本条件未绑定到任何具体跨卡迭代；空expert不授权删除Set/Wait/SyncAll，逻辑权重域不等于HBM流量，更不是25%时间收益。条件下界仍为max_r(n_r*35389440/P_MAC_upper_r)，实际n_r与匹配物理上界继续缺失。

动态attach旧尝试的前置条件另核实于启动脚本、配置和容器环境：未启用动态模式，/root及实际脚本home均无socket。/proc环境快照的空字段本身不作为当前Python环境证明。官方9.2beta1文档还说明dynamic不能与duration共用；该文档不替代本机9.2.0二进制合同。本轮已经采用实际成功的内置NPU路线，不继续变换PID重复试错。

下一步沿F21内部工作/等待和实际通信邻域处理缺少的消费者、row ownership、数值与ready协议；现有profile不能定案时只定点查算子代码。全部35+7候选、26模块358行及G00—G14保留；原9/1352、849PMU、16phase、3136归属、copy/control、all-reader/物理容量与最低MFU继续open。整体为PROGRESS，未完成。
'''
jobs=[
 {'kind':'profile','status':'pass','correctness':'pass','description':'Actual same full171case32NPU-only profiles valid;all8response texts equal;raw/DBCPU andPMUzero;966674compute executions;independent capture validation after controller error.',
  'artifacts':[H/'device-only-evidence-review.json',H/'new-profile-runtime-review.json',H/'ALL32-NPU-ONLY-RANKING.json',H/'final-runtime-and-derivation-receipt.json']},
 {'kind':'review','status':'error','description':'Post-capture controller incorrectly expected32constructors instead of16reused objects,causing extra automatic restoration;error/source preserved;original service and script restored.',
  'artifacts':[H/'controller.py',H/'device-only-submission.json',H/'device-only-evidence-review.json',H/'new-native-schema-and-case.json']},
 {'kind':'review','status':'pass','description':'Existing profile-only64replay/native model bridges and105600memberships;new288native envelopes/475200memberships;no historical operand/phase backfill.',
  'artifacts':[H/'REPLAY-MODEL-BRIDGE-v166.json',H/'NATIVE-ENVELOPES-v166.json',H/'replay-bridge-summary.json',H/'native-envelope-summary.json']},
 {'kind':'review','status':'pass','description':'HCCL enum projection corrected;424actual source neighborhoods;conditional small-row MoE work polytope explicitly excludes unbound mixed-cohort/physical claims.',
  'artifacts':[H/'TYPED-COMMUNICATION-v166.json',H/'typed-communication-neighborhoods.json',H/'decode-MoE-conditional-work-polytope.json']}]
persist_delta(version=166,previous=165,prefix='device-only-shared',changes=[
 'Actual samefullcase NPU-only cohort completed;32DBs/966674compute executions;8texts equal;service restored after accounting-error fallback.',
 'Profile-only exact CPU/runtime/vendor correlations bind64replays and105600captured executions in the old171domain.',
 'New native completion envelopes bind475200captured executions across288replays;no cross-run identity borrowing.',
 'Corrected HCCLdatatype enum;424source neighborhoods reuse existing candidate variants.',
 'Conditional M3 EP16 receiver-work polytope bounds arithmetic and occupancy assumptions without physical/speedup/independence promotion.'],
 report=report,action=action,remaining_delta='G166:Actual low-overhead timing and graph identity/envelopes available. Operator internal readiness/actual receiver work,global communication/reader identities,numerical legality andphysical resource uppers remain open. No history or whole-module closure.',
 source_gate=gate,operator_overlay=overlay,queue=queue,jobs=jobs,scripts=[Path(__file__),*H.glob('*.py')],obligation_reconciliations=recs,
 remote_status='All166acquisition/export/analysis handles terminal. Validcapture retained despite post-capture accounting assertion;original service/script restoredhealth200. No live reusable profiler capability or residual acquisition job.')
post={'status':'PASS','checkpoint':166,'checks':checks,'predecessor_pins_verified_before_shared_writes':True,
      'manifest_artifacts':len(read(O/'checkpoint-manifest-v166.json')['artifacts']),'complete_original_scope_preserved':True,'overall_complete':False}
write(O/'device-only-shared-postcheck-result-v166.json',post);write(N/'CURRENT-POSTCHECK.json',post)
write(N/'LIVE-SHARED-COHORT.json',{'effective_checkpoint':166,'status':'TERMINAL_VALID_CAPTURE_CONTROLLER_ERROR_RESTORED','receipt':str(H/'device-only-evidence-review.json'),'all_analysis_handles_terminal':True,'overall_complete':False})
write(N/'NEXT-EVIDENCE-PENDING.json',{'effective_checkpoint':166,'status':'OPERATOR_ANALYSIS_NEXT_NO_RECOLLECTION','work_queue':str(O/'coverage-work-queue-v166.json'),'next_action':action,'overall_complete':False})
print(json.dumps(post,ensure_ascii=False))
