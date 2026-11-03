"""Persist exact eager/captured evidence routes and rejected same-shape field matches."""
from copy import deepcopy
from pathlib import Path
from finalize_origin_coverage_v9 import N,O,read,write
from persist_profile_refinement_v90 import persist_refinement


def main():
    assert not(O/'REVIEW-CHECKPOINT-v125.json').exists()
    pre=read(O/'operand-correlation-preflight-v125.json');assert pre['status']=='PASS'and pre['checks']==5565
    names=['operand-correlation-inputs-v125.json','operand-correlation-native-records-v125.json',
        'operand-correlation-CPU-stacks-v125.json','operand-correlation-consequences-v125.json',
        'operand-correlation-source-and-case-witness-v125.json','operand-correlation-identity-witnesses-v125.json',
        'operand-correlation-current-method-gates-v125.json']
    detail=dict(evidence=names,hash_pinned_databases=32,original_pending_executions=1352,
        original_pending_regimes=9,eager_MemSet_executions=1272,captured_BOOL_ZerosLike_executions=80,
        eager_MemSet_exact_CPU_parent_links=1272,captured_BOOL_recorded_CPU_producer_links=0,
        same_shape_CPU_BOOL24_calls=16,same_shape_CPU_BOOL24_links_to_pending=0,
        actual_cross_process_modelId49_collisions=80,GetParams_device_argument_packets=5,
        new_native_promotions=0,new_pairs=0,new_groups=0,new_device_work=False,
        new_fusion_findings=0,whole_requirement_closed=False)
    write(O/names[-1],dict(
        status='CURRENT_EVIDENCE_METHODS;NOT_AUTHORIZATION_OR_SUBMISSION',
        replaces_stale_capability_or_pending_approval_reading_of='Historical v107 method states remain archived;v109/v120 approvals were granted and consumed.',
        methods=[
            dict(method='Exact recorded CPU/backend/runtime connection joins',established='All1272 pending MemSet tasks are eager and link to aclnnConstantPadNd;all80 pending BOOL tasks are captured templates without a recorded CPU producer link.',not_established='Native destination/count/provider-byte packet and all reader/storage epochs.'),
            dict(method='CPU BOOL24 zeros_like call stacks',established='All16 matching CPU calls are C2 complete-mask dummy initialization and exactly link to separate eager tasks,not pending graph tasks.',not_established='The semantic field of the80 original captured BOOL tasks;do not transfer identity by shape/name/time.'),
            dict(method='Installed aclmdlRITaskGetParams capability',established='Prior own native-Pad probe succeeded on5kernel tasks;all returned isHostArgs0. Graph getter exposes metadata but not a safe host pointer to opaque argument contents.',not_established='No authorization or justification for ctypes.string_at/device-argument dereference,actual historical operand extents or eager-task coverage.'),
            dict(method='Capture-creation dump/field observation',established='Needed for captured BOOL producer/value/update epoch;numeric modelId49 is actually reused across original processes and the later own probe.',not_established='Replay-only CPU profiling does not reconstruct missing capture operators;no newly authorized model capture here.'),
            dict(method='Same-call eager launch/argument observation',established='A separate necessary evidence route for all7 pending MemSet regimes;graph-only enumeration would miss them.',not_established='No ready/authorized native argument collection or proved all-case provider/write/overlap/reader identity yet.')],
        next_work='Prepare the smallest exact-provider eager-launch argument route and captured-BOOL field route using these concrete keys. Prefer existing indexed evidence;if a new device window is needed,prepare a reviewable bounded collector before requesting it.',
        no_model_load_service_restart_capture_replay_or_additional_probe_authorized=True,
        Pad_authorization='GRANTED_AND_CONSUMED',overall_complete=False))
    rel=deepcopy(read(O/'candidate-relation-clauses-v124.json'))
    rel.update(parent_relation_index='candidate-relation-clauses-v124.json',new_pair_clauses=0,
        new_higher_order_groups=0,operand_correlation_audit_v125=detail,overall_complete=False)
    write(O/'candidate-relation-clauses-v125.json',rel)
    cov=deepcopy(read(O/'candidate-relation-coverage-v124.json'))
    cov.update(parent='candidate-relation-coverage-v124.json',parent_relation_index='candidate-relation-clauses-v125.json',operand_correlation_audit_v125=detail,overall_complete=False)
    write(O/'candidate-relation-coverage-v125.json',cov)
    nav=deepcopy(read(O/'candidate-pair-navigation-v124.json'))
    nav.update(parent='candidate-pair-navigation-v124.json',parent_footprints='candidate-operator-footprints-v125.json',
        parent_relation_index='candidate-relation-clauses-v125.json',operand_correlation_audit_v125=detail)
    write(O/'candidate-pair-navigation-v125.json',nav)
    footprints=deepcopy(read(O/'candidate-operator-footprints-v124.json'))
    next(r for r in footprints['candidates']if r['id']=='F31')['operand_correlation_audit_v125']=dict(evidence=names,
        no_new_operator_atoms=True,same_shape_C2_mask_is_not_pending_graph_field=True)
    footprints.update(parent='candidate-operator-footprints-v124.json')
    write(O/'candidate-operator-footprints-v125.json',footprints)
    op=deepcopy(read(O/'operator-analysis-coverage-ledger-v124.json'));op['operand_correlation_audit_v125']=detail
    write(O/'operator-analysis-coverage-ledger-v125.json',op)

    obligations=deepcopy(read(O/'coverage-current-obligations-v124.json'))
    for r in obligations['obligations']:
        if r['id']=='G03':r['remaining']=(
            'All7 MemSet regimes/1272 executions are eager(modelId UINT32_MAX),with exact recorded CPU aclnnConstantPadNd parent links but no native pointer/count/overlap/workspace-role fields. '
            'Both BOOL24 regimes/80 executions are captured templates with no recorded CPU producer link. '
            'Sixteen recorded CPU BOOL24 zeros_like calls exactly identify separate eager C2 complete-mask initialization,not these80 graph fields. '
            'Need eager launch argument/provider/write scope and capture-creation/value/storage/update/all-reader evidence separately. '
            'Completed Pad probe proves a bounded own32B source span only;GetParams returned device argument pointers,not host-readable old packets.')
        if r['id']in ['G01','G03','G05','G09','G11','G13']:r['operand_correlation_audit_v125']=deepcopy(detail)
        if r['id']=='G13':r['remaining']=(
            'v125 has now exhausted the exact recorded CPU/backend join question for all1352 pending native tasks. '
            'Do not repeat ordinary stack/shape scans or claim graph-only collection covers the7 eager MemSet regimes. '
            'A new same-call eager argument/provider route must be prepared;captured BOOL needs original creation/value/storage/update/reader identity. '
            'The16 C2 BOOL24 eager operators are proven different tasks;modelId49 also actually collides across processes/runs. '
            'Prior GetParams capability is tested,not unavailable;all5 own Pad kernel argument packets were device-side,so do not host-dereference their addresses. '
            'No new device permission is pending;v120Pad grant consumed. All original9/1352,native precision/physical capacities/whole graph/branch and comparable MFU/lowest gaps remain.')
    obligations.update(parent='coverage-current-obligations-v124.json',new_native_or_whole_closures=0,overall_complete=False)
    write(O/'coverage-current-obligations-v125.json',obligations)
    audit=deepcopy(read(O/'whole-scope-requirement-audit-v124.json'))
    audit.update(previous_goal_turn='PROGRESS:v124 multiset/request-window semantics,2semantic rows andG9source41/49;corrected postcheck v124b PASS190,original sealed verifier variable-shadow failure retained',
        input_checkpoint=124,followup_checkpoint=125,operand_correlation_audit_v125=detail)
    for r in audit['requirements']:
        if r['id']in ['MODEL_VERSION_HARDWARE_LOAD','NATIVE_OPERATOR_REGIMES','CRITICAL_EVIDENCE_AND_FINAL_CLOSURE']:
            r.setdefault('evidence',[]).extend(names)
    write(O/'whole-scope-requirement-audit-v125.json',audit)
    write(O/'PROFILE-CODE-QUESTION-GATE-v125.json',dict(
        status='ALL32_EXISTING_PROFILES_EXACT_CONNECTION_AND_STACK_AUDIT;SOURCE_ONLY_FOR_C2_COUNTEREXAMPLE',
        questions=read(O/names[0])['questions'],
        source_scope='Existing frozen C2 complete-mask assignment lines only;no framework survey/source download.',
        CPU_stacks_not_operand_contents=True,graph_and_eager_evidence_routes_distinguished=True,
        no_new_remote_Torch_NPU_model_service_capture=True,Pad_authorization='GRANTED_AND_CONSUMED',overall_complete=False))
    write(O/'operand-correlation-validation-history-v125.json',dict(events=[
        'Revalidate predecessor using v124b verifier PASS190. Original v124 sealed verifier has local T-shadowing error;its original file/manifest and repair record retained. New fixed postcheck evidence is pinned as versioned input,not the mutable CURRENT-POSTCHECK pointer.',
        'First two local read-only attempts were stopped(exit143)after a duplicated slow per-task SQL join was noticed. Known owned local PIDs only;no server/device jobs affected. Retain both attempts,then replace repeated joins with one lookup per database.',
        'All32 cached raw databases re-hashed against v98 original source hashes,opened ro&immutable,temp_store MEMORY,stat unchanged. Exact1352 TASK keys match original pending list.',
        '1272 MemSet tasks all have modelId UINT32_MAX and exact framework group ->runtime connection ->TASK joins to aclnnConstantPadNd. CANN launch alone still has no destination/length/provider packet contents.',
        '80 BOOL ZerosLike tasks all have captured modelId and no matching recorded CPU producer connection. Empty join is not an empty operator;capture/replay template IDs are not eager operator creation events.',
        'All16 CPU BOOL24 zeros_like calls have dsa_v41.py1342 build_c2_metadata leaf stack. Nested aclnnInplaceZero group IDs join to16 distinct eager native tasks;zero overlaps with original80 pending graph TASK keys.',
        'Concrete database0 false match:CPU complete-mask zero links to eager TASK817/global811,not original captured TASK880/global5535. No nearest timestamp or shape/name binding used.',
        'Own later native-Pad debug dump modelId49 collides numerically with all80 original pending graph BOOL tasks from16 different original processes. Host/run/process/rank/device/capture/update epoch is mandatory;do not join by display modelId.',
        'Own Pad GetParams5kernel rows all isHostArgs0. No host dereference added. Prior public32Bsource-span/packet152 capability remains scoped to that own capture,not old eager MemSet operands.',
        'Independent exact-SQL join verifier preflight5565checks PASS;all pending pointer/length/field/readers stay null and all9regimes/1352executions remain open. No root fusion/centralpair/globalgraph/semantic-inventory expansion claimed.',
        'Current method gates replace stale v107 capability/approval readings without editing historical artifacts. Need a separately prepared eager-launch route plus captured-BOOL producer route,not another equivalent Pad probe or ordinary stack-only full profile.'],overall_complete=False))
    gap=(
        'Same-shape BOOL24 CPU C2 complete-mask calls are distinct eager tasks,not pending graph field evidence. '
        'Need capture-creation/value/storage/update epoch and all readers for80 graph BOOLs. '
        'All7 pending MemSet regimes are eager:recorded aclnnConstantPadNd parent stacks do not expose native destination/count/provider byte packet,overlap or workspace role;graph-only getters cannot cover them. '
        'Use complete process/run/rank/device/graph/update identities,not modelId49 alone. GetParams device argument pointers are not safe host packet addresses. '
        'No native fusion/speedup/physical upper/MFU or whole-module closure is proved.')
    remaining=[r['id']+': '+r['title']+'. '+r['remaining']for r in obligations['obligations']]
    remaining.append('v125 exact correlation audit:1272eager MemSet parentlinks/80captured BOOL withoutCPUproducer;16same-shape C2 eagerBOOL are different tasks. Need separate launch-argument andcapture-creation routes;all9/1352 remain. Inventory358/26,partialgraph854/1346/48,relations135/26/460 and0whole closures unchanged.')
    report='''本轮优先补原生实参缺口的取证边界，未再扩展假设性融合候选。上一轮v124的源码分支与G9图是实质进展；已用修正版v124b复核通过190项，保留原封存核验脚本的变量名冲突和修复记录。本轮不修改历史结论，也没有新的服务器、Torch、NPU、模型或服务操作。

对32份缓存原始profiler数据库重新核对SHA、只读打开并验证stat未变，逐一复查原9类/1352次待补任务。正确关联链是PYTORCH_API.connectionId→CONNECTION_IDS.id→CONNECTION_IDS.connectionId→TASK.connectionId；CANN_API使用runtime connection namespace。不能直接把framework group ID与runtime ID数值相等当作关联，也没有用最近时间、同名或同shape补齐缺失关联。

结果明确分成两条不同取证路径：1272次MemSet全部是eager任务，modelId为UINT32_MAX，每次都能精确关联到CPU backend aclnnConstantPadNd以及CANN launch。这个结果保留已有的Pad owner结论，但调用栈不包含原生destination、length、写重叠或workspace/payload身份，仍不能据此删除某个MemSet helper。只在NPUGraph完成后枚举graph tasks的方案，会漏掉全部七类待补MemSet，不能作为这部分原始证据的完整采集方案。

另80次BOOL[24] ZerosLike均为捕获图任务；原数据库找不到与它们runtime connection匹配的CPU producer。不存在的CPU关联并不表示没有算子工作，也不能用replay时附近的CPU初始化替代capture时的字段身份。这里真正需要的是原capture-creation/value/storage及后续参数update epoch与全读者信息，而不只是再打开with_stack或按replay任务名查询。

已核对所有16条CPU BOOL[24] aten::zeros_like：其叶栈都是dsa_v41.py第1342行build_c2_metadata。冻结源码对应skip_ring_update分支中的complete=torch.zeros_like(complete)，随后写入持久C2 complete-mask bank。沿嵌套aclnnInplaceZero的实际connection group继续关联，它们恰好指向16个不同的eager native任务，与80个待补graph任务的精确键交集为0。

数据库0给出一个具体误关联反例：该CPU complete-mask初始化关联eager TASK row817/globalTaskId811；原待补BOOL对应捕获图TASK row880/globalTaskId5535。两者同为BOOL[24]、相似ZerosLike名字，但不是同一算子值。这个CPU栈能够解释自己的调用来源，不能据此宣称原prefix BOOL字段已经定案，更不能删除原图中的初始化或忽略其读者。

另一个真实碰撞出现在display modelId：先前自己的Pad小探针debug dump也为modelId49，而80个原待补graph BOOL任务也都使用数值49，来自原16个进程。新probe的进程14525与原763/764等进程和采集时间均不同。modelId不是跨进程/跨run全局唯一ID，更不是modelRI handle；至少要同时保留host/run/process/rank/device、capture及update epoch、stream/task等身份，不能把新探针的32B写范围带入原图。

先前Pad探针还有一个实际接口事实：五个成功的kernel GetParams查询全部isHostArgs=0。返回的是设备侧参数地址与metadata，不能把opaque地址交给ctypes.string_at当主机包读取，argsSize也不是写长度。既有debug dump及有界32B source-span证明仍只属于自己的那次捕获。此处没有新增参数解引用、device-to-host复制或探针，也没有把已消耗的Pad授权当成新的授权。

本轮据此更新了当前方法表：eager MemSet需要同调用launch/argument/provider/分配与读写生命周期证据；captured BOOL需要capture-creation和具体字段/storage/update/读者证据。只重采普通profile/callstack或只采graph参数都不足。下一步先准备能覆盖这两类的定点取证入口；若必须新增设备窗口，应先形成可审查、有限范围的采集器，再请求相应授权。目前没有新的device permission等待回复，也没有启动模型采集。

前置5565项精确关联核验通过；初始两个本地慢查询尝试已中止并保留记录，改为每库一次批量lookup后完成。所有旧1352任务的destination/length/BOOL-field/readers仍为未知，9类pending没有被提升或关闭。35模型候选、7历史框架记录、358语义行/26模块、135条件关系/26联合组/460待核对pair、部分图854/1346/48、849缺PMU、16 phase未绑定图及3136含糊归属均保持，整模块关闭0。这里的进展是已用完整现存profile证据修正后续采集路线、排除了具体的错误字段绑定；整体目标仍未完成。
'''
    action=(
        'Continue original Goal;v125 PROGRESS from exact1352-task correlation audit and concrete false-field/modelId collision evidence. '
        'Read operand-correlation summary/native-records/CPU-stacks/consequences/identity-witnesses/current-method-gates/current-obligations-v125;verify_operand_correlation_v125 must PASS. '
        'Predecessorv124 sourceG9/2semanticrows progress;usev124b PASS190 because original sealedv124 verifier has localTshadow bug,repair retained. '
        '32rawDB SHA/stat/roimmutable verified;correctjoin PyTorch.connectionId->CONNECTION_IDS.id->runtimeCID->TASK.CID. '
        'All1272pendingMemSet eager modelIdUINT32MAX exactCPUaclnnConstantPadNd parent+CANNlaunch;no dst/count/provider bytes. Graph-onlycollector missesall7regimes. '
        'All80pendingBOOL24 captured templates have no recordedCPUproducerlink.16CPUBOOL24zeros_like calls dsa_v41.py1342 C2complete skipupdate linkto16separate eager tasks,0pendingintersection;do not relabelasoriginalfield. DB0eagerTASK817/global811 !=graphTASK880/global5535. '
        'NumericmodelId49collidesfor80originalgraphBOOLs across16oldprocessesandownPadpid14525;mustkeepfullrun/process/rank/device/capture/update/stream/task keys. '
        'OwnPadGetParams5kernelsall isHostArgs0 device packets;neverhostdereferenceopaqueaddresses oruseargsSizeaswriteextent. Prior32B source-span isownprobeonly. '
        'Prepare exact-provider same-call eagerlaunch argument route andcapturedBOOLcreation/storage/reader/update route;not another ordinaryprofile or equivalentPadprobe. No newcapture/model/devicepermissionpendingorgranted;Padv120consumed. '
        'Two slowduplicate localread attempts stopped143 andretained,batchedqueryreplacement completed;no remote/devicejobaffected. '
        'All9/1352remainpending,358/26 semantic,854/1346/48 graph,135/26/460 relations,2073/285598 conditional849PMU16phase3136ambiguity0whole unchanged;noallMFU/physicalceiling/speedup. '
        'All localjobs finished;do noteditsealedv125/rerunfinalizer. Current postcheck overlay is onlyverifierselection andmustbeusedonlyforitsmatchingcheckpoint. TaskACTIVE Loop004EVALUATING GoalACTIVE.')
    inputs=[Path(r['path'])for r in read(O/names[0])['inputs']]
    persist_refinement(version=125,previous=124,title='Exact profiler operand provenance and eager versus captured diagnostic coverage',
        summary=dict(filename='operand-correlation-summary-v125.json',
            previous_turn_classification='PROGRESS:v124 sparse multiset/request-window semantics andG9;corrected v124b postcheck PASS190',
            **detail,refined_candidates=1,refined_modules=4,branch_records=358,semantic_modules=26,
            relation_clauses=135,higher_order_groups=26,pending_central_pair_reconciliation=460,
            partial_graph_node_rows=854,partial_graph_edge_rows=1346,structural_pair_hints=48,
            scoped_reviewed_regimes=2073,scoped_reviewed_executions=285598,
            pending_regime_reconciliation=9,pending_execution_reconciliation=1352,
            actual_MFU_numeric_physical_ceiling_or_speedup_proved=False,implementation=False,native_test=False,
            Pad_authorization='GRANTED_AND_CONSUMED',
            remote_job_status='All local read-only SQLite/source/probe-metadata comparisons complete;no remote/Torch/NPU/model/replay/implementation. Two local query attempts stopped and retained;replacement passed.'),
        changes=[
            'Audit exact original1352task connection IDs across32hash-pinnedDBs:1272eagerMemSet parentlinks versus80capturedBOOL tasks without recordedCPUproducers.',
            'Reject16actualsame-shape C2BOOL eager calls as a substitute for pendinggraphfields;recordactualcross-process modelId49collision and device-sideGetParams pointer fact.',
            'Replace a graph-only/stack-only evidence route with explicit eagerlaunch versuscapturedcreation requirements;retainalloriginalnative/whole/numerical/capacity gaps.'],
        candidate_updates={'F31':dict(review_v125=deepcopy(detail),conclusion_changing_gap=gap)},
        module_updates={
            'M05':dict(candidates=[],finding='Unresolved prefix BOOL24 captured tasks cannot be identified by a separate same-shape C2 eager call;exact recorded producer linkage is absent,not zero work.',evidence=names,gap=gap),
            'M12':dict(candidates=[],finding='All original router MemSet helpers belong to eager launch scope;CPU ConstantPadNd parent does not supply native fill operands,so graph-only metadata collection is insufficient.',evidence=names,gap=gap),
            'M19':dict(candidates=[],finding='Original initialization MemSet helpers are eager and need launch-argument/provider/interval evidence;shape/stack/same model number cannot establish helper write extent.',evidence=names,gap=gap),
            'M21':dict(candidates=['F31'],finding='Sixteen CPU C2 complete-mask zeros_like calls have exact distinct eager-task provenance;they do not close the80prefix graphBOOL field/readers. Full capture/update/namespace identity remains necessary.',evidence=names,gap=gap)},
        remaining=remaining,action=action,
        jobs=[dict(kind='profile',status='error',description='First local read-only SQLite attempt stopped143 before completion;repeated per-task joins were too slow,all source/device processes untouched.',artifacts=[O/'operand-correlation-local-attempts-v125.json']),
            dict(kind='profile',status='error',description='Duplicate local audit attempt stopped143 before completion;known owned local process only,then one batched lookup implementation used.',artifacts=[O/'operand-correlation-local-attempts-v125.json']),
            dict(kind='profile',status='pass',description='Batched exact connection audit32cachedDBs/1352originalpendingTASKs plus16CPU BOOL24 call chains;no timestamp matching/no model/devicework.',artifacts=[O/n for n in names]+inputs),
            dict(kind='design-check',status='pass',description='Independent exactSQL join verification5565preflightchecks,actualfalse-field/modelId/devicepacketwitnesses;all9/1352unknown operands retained.',artifacts=[O/'operand-correlation-preflight-v125.json',O/'operand-correlation-validation-history-v125.json',O/'coverage-current-obligations-v125.json',O/'whole-scope-requirement-audit-v125.json'])],
        report=report,read_first=names+['coverage-current-obligations-v125.json'],publication_rows={},
        gate_updates=dict(overall_complete=False),scripts=[N/'analyze_operand_correlation_v125.py',N/'finish_operand_correlation_witnesses_v125.py',
            N/'verify_operand_correlation_v125.py',N/'persist_profile_refinement_v90.py'],
        executor_prefix='operand-correlation',main_script=Path(__file__),source_read=False)


if __name__=='__main__':main()
