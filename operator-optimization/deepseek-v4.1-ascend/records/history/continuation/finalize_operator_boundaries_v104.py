"""Persist operator-only M00/M03 semantic scope and saved zero-scheduling evidence."""
from copy import deepcopy
from pathlib import Path
import hashlib
from finalize_origin_coverage_v9 import N,O,R,read,write
from persist_profile_refinement_v90 import persist_refinement

def main():
    assert not(O/'REVIEW-CHECKPOINT-v104.json').exists()
    assert read(O/'operator-boundary-analysis-preflight-v104.json')['status']=='PASS'
    names=['operator-boundary-inputs-v104.json','operator-boundary-source-contract-v104.json',
        'operator-boundary-saved-work-domain-v104.json','operator-boundary-semantic-inventory-v104.json',
        'operator-boundary-history-crosswalk-v104.json','operator-boundary-demand-and-theory-contract-v104.json']
    gap=('M00/M03 operator-selection/readiness scope is semantically reconciled, not all runtime selectors or branches closed. '
        '64 saved local-zero scheduler entries do not attest no device work or actual dummy execution; finished/block-free fields, executor/KV/EC flags, loaded callbacks, all-rank same-call bounds and publication/readiness epochs remain missing. '
        'Initialization, steady operator work, instrumented host wrappers and historical169 CPU samples retain separate domains. '
        'No numeric physical ceiling, all-module MFU, native phase/assignment promotion or whole-goal closure.')
    inv=read(O/'operator-boundary-semantic-inventory-v104.json')['modules']
    rel=deepcopy(read(O/'candidate-relation-clauses-v103.json'))
    rel.update(parent_relation_index='candidate-relation-clauses-v103.json',new_pair_clauses=0,new_higher_order_groups=0,
        operator_boundary_review_v104=dict(evidence=names,no_new_relations=True,
            scope='Operator selection/operand-generation/row-readiness constraints on existing candidate domains only; no new pair compatibility, runtime flags or work credit.'),overall_complete=False)
    write(O/'candidate-relation-clauses-v104.json',rel)
    cov=deepcopy(read(O/'candidate-relation-coverage-v103.json'))
    cov.update(parent='candidate-relation-coverage-v103.json',parent_relation_index='candidate-relation-clauses-v104.json',
        operator_boundary_review_v104=dict(new_pairs=0,new_groups=0,new_native_promotions=0,evidence=names),overall_complete=False)
    write(O/'candidate-relation-coverage-v104.json',cov)
    fps=deepcopy(read(O/'candidate-operator-footprints-v103.json'));ix={r['id']:r for r in read(O/'candidate-record-index-v103.json')['model_candidates']}
    for r in fps['candidates']:
        h=hashlib.sha256((R/'candidates'/(r['id']+'.json')).read_bytes()).hexdigest();assert h==ix[r['id']]['master_sha256']
        r['captured_master_sha256']=h;r['captured_master_version']=103
        if r['id']in ['F21','F31']:r['operator_boundary_review_v104']=dict(evidence=names,selection_and_state_generation_only=True,new_native_atoms=0)
    fps.update(parent='candidate-operator-footprints-v103.json',overall_complete=False);write(O/'candidate-operator-footprints-v104.json',fps)
    nav=deepcopy(read(O/'candidate-pair-navigation-v103.json'))
    nav.update(parent='candidate-pair-navigation-v103.json',parent_footprints='candidate-operator-footprints-v104.json',parent_relation_index='candidate-relation-clauses-v104.json',
        operator_boundary_review_v104=dict(evidence=names,operator_readiness_crosswalk_not_new_pair_coverage=True),overall_complete=False)
    write(O/'candidate-pair-navigation-v104.json',nav)
    op=deepcopy(read(O/'operator-analysis-coverage-ledger-v103.json'))
    op['operator_boundary_review_v104']=dict(saved_execute_entries=3712,local_zero_scheduled_entries=64,
        no_zero_device_work_or_actual_dummy_attestation=True,new_native_regimes=0,new_native_executions=0,
        new_native_attribution_or_phase_promotions=0,partial_operator_graph_counts_unchanged=[813,1297])
    write(O/'operator-analysis-coverage-ledger-v104.json',op)
    audit=deepcopy(read(O/'whole-scope-requirement-audit-v103.json'))
    audit.update(previous_goal_turn='PROGRESS:v103 provisional selection/attention state and work domains;151 postchecks',input_checkpoint=103,followup_checkpoint=104)
    for r in audit['requirements']:
        if r['id']=='ALL_MODULE_CONDITIONAL_BRANCHES':
            r.update(state='PARTIAL',semantic_modules=26,branch_records=353,chronological_only=[],
                gap='All26provisionalboundariesnowhaveoperator-semanticrows;thisclosesonlythechronological-formatgap. Actualselectors/loadedobjects/state/operand/row/readinessconditionsandallapplicableoperatorbranchesremainopen.')
            r.setdefault('evidence',[]).extend(names)
        if r['id']in ['MODEL_VERSION_HARDWARE_LOAD','CRITICAL_EVIDENCE_AND_FINAL_CLOSURE']:r.setdefault('evidence',[]).extend(names)
    audit['operator_boundary_review_v104']=dict(reconciled_old_chronological_records=26,old_records_dropped=0,new_operator_semantic_rows=15,
        net_record_change=-11,branch_records=353,semantic_modules=26,
        local_zero_scheduled_entries=64,actual_flags_or_dummy_or_zero_device_work_attested=False,
        previous_native_phase_and_physical_uncertainty_preserved=True,whole_requirement_closed=False)
    write(O/'whole-scope-requirement-audit-v104.json',audit)
    write(O/'PROFILE-CODE-QUESTION-GATE-v104.json',dict(status='SAVED_SCALAR_SCHEDULE_CENSUS_THEN_ONLY_ZERO_WORK_OPERATOR_OBLIGATION_SOURCE',
        questions=['Which implementation/operand-generation/readiness questions in chronological M00/M03 actually affect operator fusion or ceilings?',
            'Does local scheduler zero imply no state/transfer/collective work?',
            'Can existing operator work/precision analyses be linked without double-counting them at host boundaries?',
            'Which historical/current/missing evidence domains must remain separate?'],
        new_target_source_files=0,new_public_documents=0,cached_point_read_source_files=1,
        no_framework_wide_research=True,no_server_model_Torch_NPU_capture_or_implementation=True,overall_complete=False))
    write(O/'operator-boundary-validation-history-v104.json',dict(refinements=[
        'v103 predecessor revalidated PASS151 before refinement. Protected artifacts and completed finalizers were not modified or rerun.',
        'M00 had14 chronological records and M03 had12. All26 exact old records are retained with explicit mappings to15 scoped semantic rows; helper archives the previous full modules. No unresolved historical/current requirement is silently marked closed.',
        'All16 existing metadata streams rehashed and matched the saved index.3712 execute entries,64 local-zero and3648 positive;each process has4 zero calls. TP process replicas are not unique logical steps or native processed-row counts.',
        'Zero events have empty scheduled/new/cached-request lists, but finished/free-block and executor/KV/EC predicates are absent. Normal LM source calls _update_states before local-zero guard and optionally handles KV preemptions earlier; external_launcher&&DP>1 invokes dummy coordination before empty/KV return.',
        'Saved installed model_runner hashes match the source snapshot. This binds reported on-disk bytes, not all actual callback objects or branch predicates. The eight source cases are conditional;no actual dummy/zero-device/native phase promotion.',
        'Zero-call instrumented wrapper wall/threadCPU summaries are retained as measured host surface, not cause-specific timing/NPU completion/critical-path savings. Historical169 response wait/onCPU evidence stays historical.',
        'A local analysis reference used a wrong ring-contract filename and stopped before shared-state mutation. It was corrected to the existing ring-storage-source-contract-v79.json and all evidence references were checked.',
        'A bounded saved-service-log query found no executor/KV/EC selector strings; no absence-to-False inference was made. No server or runtime access was performed.',
        'Semantic-format coverage becomes26modules/353rows, a net reduction11 after reconciliation;not an optimization percentage or all-condition closure. Native9/1352,849PMU,16unboundgraphs,3136ambiguity,relations131/25/464 and partial graph813/1297/46 remain unchanged. All-module MFU and whole physical ceilings remain open.'],overall_complete=False))
    updates={cid:dict(review_v104=dict(evidence=names,operator_boundary_scope_only=True,
        saved_local_zero_scheduled_entries=64,actual_executor_KV_EC_or_zero_device_work_attested=False,
        old_chronological_rows_retained=26,new_semantic_boundary_rows=15,
        implementation=False,native_test=False,new_runtime_executions=0),conclusion_changing_gap=gap)for cid in ['F21','F31']}
    mods={
        'M00':dict(candidates=['F21'],finding='Eight semantic operator-selection families replace14 chronological rows with full history/crosswalk retained. Quant/native/attention/MC2/HC/Engram/load/sampling identities and operand epochs are conditional;startup and steady domains remain separate.',evidence=names,gap=gap,
            profile_scope='Only implementation selection and parameter/operand-generation constraints;not framework behavior or a new startup profile.',scope_audit=dict(old_rows=14,semantic_rows=8,whole_closed=False)),
        'M03':dict(candidates=['F31'],finding='Seven semantic row-demand/readiness/publication/state-consumer families replace12 chronological rows.64 saved local-zero calls retain prior state/transfer/DP obligations;actual flags/dummy/zero device work are not attested.',evidence=names,gap=gap,
            profile_scope='Saved worker scheduler scalars and conditional source obligations;historical169 EngineCore cost is not current166 attribution.',scope_audit=dict(old_rows=12,semantic_rows=7,whole_closed=False))}
    remaining=list(read(O/'module-coverage-gaps-v103.json')['remaining']);remaining.append(
        'v104: Remaining chronological M00/M03 format gap reconciled into8+7 operator-selection/readiness rows. All26 exact old records retained/crosswalked;total353rows/26semantic modules,not all-branch closure. Saved3712worker entries include64local-zero4perprocess;state update/KV preemptions precede return and DP dummy is conditional on external_launcher&&DP>1. Actual predicates/finished/free-block/publication epochs/allrank bounds remain unknown;host wrappers not device completion. No nativephase/assignment or zero-device promotion.131/25/464relations,813/1297/46partialgraphs,2073/285598conditional9/1352pending849PMU16phase3136ambiguity0whole/no allmoduleMFU unchanged.')
    report='''本轮把最后两个仍按历史工作顺序排列的边界M00、M03，整理为直接影响算子结论的语义台账。没有展开框架API研究。M00只覆盖实现选择与参数/布局版本；M03只覆盖算子输入规模、DP/TP/EP参与、准备/发布就绪以及跨迭代状态读者。详细算子语义仍由原M04–M25分析承担，不在这些边界重复计矩阵或通信工作。

M00原14条历史记录映射到8类算子选择：模型/硬件/CANN/量化/并行和图模式；dense/norm/quant；attention/Indexer/CP；MC2/MoE/共享专家；HC/DSpark；Engram；加载回调、静态专家布局与运行时EPLB；采样后端与greedy/random等guard。M03原12条映射到7类：实际调度行与prefix/chunk；跨卡参与和全局padding；异步发布和buffer所有权；draft/acceptance长度；零本地工作；移除/恢复/回滚/slot版本；最终输出和终止读者。全部26条旧记录逐条保留，并保存旧模块快照，没有因重排而关闭缺口。

新核对的保存元数据共16个worker进程、3712次execute进入，其中64次本地scheduled_tokens=0，每进程4次；其余3648次为正。每条零调用均未登记scheduled/new/cached请求，但没有记录finished_req_ids、free/reused block或全部状态写入。因此空调度列表不能证明没有状态清理，也不能把进程副本数量当成独立全局步骤或NPU行数。

定点源码只看execute_model中会改变零行算子合同的分支。正常LM路径会先处理条件性的KV preemption并更新持久batch状态，然后才判断本地调度token是否为0；若是external_launcher且DP>1，还会条件性执行_dummy_run(1)以参加DP协调。随后返回empty output或走kv_connector_no_forward。实际后端/KV/EC谓词没有在这些事件中直接记录，所以本轮没有声称这64次一定执行了dummy，也没有声称它们没有设备工作。

这给出明确的优化边界：不能仅凭本地query行数为0，把前置状态、KV传输或集体通信参与删掉；需要对应的状态写集合、参与者和代际证明。新fast path还必须保持已有错误、preemption-before-slot-reuse和消费者确认顺序。已有代码本就具备零行返回，本轮没有把它认领为新融合机会。

这些零调用的已有instrumented wrapper wall记录范围为332400–1562840ns，中位数635625ns；thread CPU另行保存。它们是带取证开销的宿主包装范围，包含的具体工作和设备完成时间未分解，不能作为可删除延迟。原169的EngineCore轮询/onCPU资料仍保留历史归属，不能替代当前166证据。

理论部分保留INCOMPLETE。局部query行数为0没有决定全部必要工作；选择器和参数版本影响合法实现域、缓存有效性及启动/稳态分母，不能凭缺少ComputeInfo把初始化或宿主边界记为0成本，也不能据此选出“最低利用率模块”。同口径全模块MFU仍未建立。

当前26个暂定模块都有算子语义记录；总记录从364条变为353条，原因是26条历史条目被15条语义条目替代。这关闭了“台账仍为时间顺序”的格式缺口，没有证明所有条件分支完整。现有35项模型候选、131条条件关系/25组联合合同/464对待汇入或复核、局部图813节点行/1297边行，以及9个regime/1352次执行关键字段、849 PMU/16阶段图/3136归属歧义均未被虚假提升。整体目标继续进行。
'''
    action=('继续原Goal,v104PROGRESS；v103PASS151。本轮179prechecks。先读operator-boundary-summary-v104/report/semantic-inventory/history-crosswalk/saved-work-domain/demand-theory/wholeaudit。'
        'M00/M03最后chronological范围已operator-only语义归并：14+12old exactrecords全部保留并archive，8+7newrows=>total353/26semantic modules。仅关闭格式缺口，allconditions/phases/whole仍open；不得继续扩展framework API研究或重复归并计作发现。'
        '新正证据：16savedworker streams3712enter中64localzero(4each)，scheduled/new/cached空但finished/freeblocks未知；normalLM source update_states在zero guard之前、KVpreemptions可更早、external_launcher&&DP>1条件dummy(1)，然后empty/KV noforward。实际executor/KV/EC flags不在事件，不能说zeroNPU或actualdummy；16installed sourcehash匹配on-disk，非所有loadedobject attestation。'
        'Zero wrapper wall median635625ns range332400..1562840，仅instrumented host surface非devicecompletion/可删延迟；historical169 H01 CPU不替代166。无native phase/assignment promotion。'
        'Counts131/25/464relations,813/1297/46partialgraphs,2073/285598conditional9/1352pending849PMU16phase3136ambiguous,0whole/no allmoduleMFU unchanged。'
        'Next perform requirement-level coverage/evidence dependency audit on current353semantic rows: distinguish actual unexpanded operator dataflow from resolved conditional questions and irreducible missing runtime/native/capacity evidence. Prioritize evidence that can change final fusion/theory conclusions; do not enumerate near-duplicate variants or inflate row counts. Minimal bounded evidence planv98 remains prepared/unexecuted and critical9native operands still missing. '
        'All local jobs terminal;no server/model/Torch/NPU/capture/service/environment/implementation. GoalACTIVE TaskACTIVE Loop004EVALUATING；verify_operator_boundaries_v104 mustPASS；never rerun completed finalizer or edit completed artifacts.')
    inputs=[Path(r['path'])for r in read(O/names[0])['inputs']]
    jobs=[dict(kind='profile',status='pass',description='Hash-check16 saved schedule streams and census64 local-zero entries;point-read normal-LM source state/transfer/coordination obligations with no actual flag or native-phase promotion.',artifacts=[O/n for n in names]+inputs),
        dict(kind='design-check',status='pass',description='Retain/crosswalk26 chronological boundary records into15 operator-semantic rows,eight conditional zero-work source cases and explicit incomplete theory;no framework catalogue or duplicate work.',
            artifacts=[O/'operator-boundary-analysis-preflight-v104.json',O/'operator-boundary-validation-history-v104.json',O/'whole-scope-requirement-audit-v104.json'])]
    persist_refinement(version=104,previous=103,title='Operator implementation selection and zero-scheduled readiness boundaries',
        summary=dict(filename='operator-boundary-summary-v104.json',previous_turn_classification='PROGRESS:v103 provisional selection/attention fusion domains;151 postchecks',
            saved_worker_streams=16,saved_execute_entries=3712,local_zero_scheduled_entries=64,
            actual_dummy_or_zero_device_work_attested=False,old_chronological_rows_retained=26,new_semantic_boundary_rows=15,net_branch_record_change=-11,
            new_target_source_files=0,new_public_documents=0,refined_candidates=2,refined_modules=2,
            branch_records=353,semantic_modules=26,new_pairs=0,new_groups=0,relation_clauses=131,higher_order_groups=25,pending_central_pair_reconciliation=464,
            partial_graph_node_rows=813,partial_graph_edge_rows=1297,structural_pair_hints=46,scoped_reviewed_regimes=2073,scoped_reviewed_executions=285598,
            pending_regime_reconciliation=9,pending_execution_reconciliation=1352,new_native_promotions=0,
            actual_MFU_numeric_physical_ceiling_or_speedup_proved=False,implementation=False,native_test=False,
            remote_job_status='All local saved-profile/source-obligation/coverage checks finished;no server/model/Torch/NPU/capture/service/environment or implementation.'),
        changes=['Reconcile remaining chronological M00/M03 into15 operator-selection/readiness rows;retain all26 old records and open gaps.',
            'Census64 saved local-zero scheduler entries;source preserves state/KV and conditional DP obligations before return.',
            'Separate recorded scalar/host wrapper/historical CPU/source predicate evidence;no zero-device or native phase promotion.',
            'All26 provisional modules now semantic/353rows;only format gap closed,all-condition/physical/theory/MFU/whole goal remain open.'],
        candidate_updates=updates,module_updates=mods,semantic_branches=inv,remaining=remaining,action=action,jobs=jobs,report=report,
        read_first=names+['whole-scope-requirement-audit-v104.json'],publication_rows={},gate_updates=dict(overall_complete=False),
        scripts=[N/'analyze_operator_boundaries_v104.py',N/'verify_operator_boundaries_v104.py',N/'persist_profile_refinement_v90.py'],
        executor_prefix='operator-boundary',main_script=Path(__file__),source_read=False)

if __name__=='__main__':main()
