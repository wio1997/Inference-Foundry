"""Persist current-load operator-domain evidence without promoting unrecorded runtime flags."""
from pathlib import Path
from copy import deepcopy
import hashlib
from finalize_origin_coverage_v9 import N, O, R, read, write
from persist_profile_refinement_v90 import persist_refinement

def main():
    assert not (O/'REVIEW-CHECKPOINT-v102.json').exists()
    assert read(O/'sampling-applicability-analysis-preflight-v102.json')['status']=='PASS'
    names=['sampling-applicability-inputs-v102.json','sampling-load-provenance-v102.json',
        'sampling-saved-request-domain-v102.json','sampling-operator-guard-contract-v102.json',
        'sampling-case-fusion-and-ceiling-routing-v102.json']
    gap=('Frozen client/controller pin and ten saved IDs support explicit-temperature-zero request intent. '
        'Valid nonempty all-greedy source domain bypasses post-return random recovery; this is not per-call effective flag attestation. '
        'Actual all_greedy/all_random, closed external admissions, inherited loaded body/metadata epochs, all raw/processor/logprob/entropy/synthetic RNG readers and native comparator/alias/storage/communication capacities remain conditional. '
        'Empty sets make both booleans True; greedy kernel presence also admits mixed rows. No MFU, numeric whole physical ceiling, native promotion or whole-goal closure.')
    rel=deepcopy(read(O/'candidate-relation-clauses-v101.json'))
    row=next(r for r in rel['clauses']if r['id']=='REL73-26')
    row['sampling_case_applicability_v102']=dict(evidence=names,
        contract='Request/row plan reuse remains an existing F31/F35 conditional edge. In a valid nonempty sealed greedy source domain, recovery consumer is unreachable; do not credit random-recovery interfaces as current fusion savings. No field identity or actual per-call flag promotion.',
        all_runtime_gates_proved=False)
    group=next(g for g in rel['higher_order_groups']if g['id']=='HG73-TAIL')
    group['sampling_case_applicability_v102']=dict(evidence=names,
        contract='Route the chosen tail graph by effective nonempty request domain before costing. Preserve raw/snapshot/processors, pre-return entropy and synthetic uniform effects; distinct source-domain budgets are not additive.',whole_group_closed=False)
    rel.update(parent_relation_index='candidate-relation-clauses-v101.json',new_pair_clauses=0,new_higher_order_groups=0,overall_complete=False)
    write(O/'candidate-relation-clauses-v102.json',rel)
    cov=deepcopy(read(O/'candidate-relation-coverage-v101.json'))
    cov.update(parent='candidate-relation-coverage-v101.json',parent_relation_index='candidate-relation-clauses-v102.json',
        sampling_case_applicability_v102=dict(evidence=names,new_pairs=0,new_groups=0,new_native_promotions=0),overall_complete=False)
    write(O/'candidate-relation-coverage-v102.json',cov)
    fps=deepcopy(read(O/'candidate-operator-footprints-v101.json'))
    ix={r['id']:r for r in read(O/'candidate-record-index-v101.json')['model_candidates']}
    for r in fps['candidates']:
        p=R/'candidates'/(r['id']+'.json');h=hashlib.sha256(p.read_bytes()).hexdigest();assert h==ix[r['id']]['master_sha256']
        r['captured_master_sha256']=h;r['captured_master_version']=101
        if r['id']=='F35':r['sampling_case_applicability_v102']=dict(evidence=names,current_native_atom_or_regime_removal=0,actual_guard_attested=False)
    fps.update(parent='candidate-operator-footprints-v101.json',overall_complete=False);write(O/'candidate-operator-footprints-v102.json',fps)
    nav=deepcopy(read(O/'candidate-pair-navigation-v101.json'))
    nr=next(r for r in nav['pairs']if{r['a'],r['b']}=={'F31','F35'})
    nr['sampling_case_applicability_v102']=dict(existing_relation='REL73-26',evidence=names,no_native_promotion=True)
    nav.update(parent='candidate-pair-navigation-v101.json',parent_footprints='candidate-operator-footprints-v102.json',parent_relation_index='candidate-relation-clauses-v102.json',overall_complete=False)
    write(O/'candidate-pair-navigation-v102.json',nav)
    op=deepcopy(read(O/'operator-analysis-coverage-ledger-v101.json'))
    op['sampling_case_applicability_v102']=dict(saved_worker_streams=16,saved_request_id_union=10,execute_entries=3712,
        per_worker_DP0_entries=233,per_worker_DP1_entries=231,actual_runtime_guard_attested=False,
        new_native_regimes=0,new_native_executions=0,new_native_attribution_or_promotions=0,operator_graph_counts_unchanged=[813,1297])
    write(O/'operator-analysis-coverage-ledger-v102.json',op)
    branches=deepcopy(next(m['branches']for m in read(O/'module-phase-branch-matrix-v101.json')['modules']if m['id']=='M17'))
    for r in branches:
        if r['id']in ['M17-OP-01','M17-OP-02','M17-OP-03','M17-OP-04','M17-OP-05','M17-OP-06','M17-OP-07','M17-OP-09','M17-OP-12','M17-OP-14','M17-OP-15','M17-OP-16']:
            r['sampling_case_applicability_v102']=dict(evidence=names,
                scope='Frozen explicit-zero request contract; valid nonempty all-greedy source-domain routing, not actual flag or all-reader closure.',
                post_return_random_savings_in_greedy_domain=0,all_runtime_flags_or_branches_closed=False)
    audit=deepcopy(read(O/'whole-scope-requirement-audit-v101.json'))
    audit.update(previous_goal_turn='PROGRESS:v101 conditional fallback bodies and precision/row-plan contracts;170 postchecks',input_checkpoint=101,followup_checkpoint=102)
    for r in audit['requirements']:
        if r['id']in ['MODEL_VERSION_HARDWARE_LOAD','ALL_MODULE_CONDITIONAL_BRANCHES','THEORY_AND_MEASURED_COMPARISON']:
            r.setdefault('evidence',[]).extend(names)
    audit['sampling_case_applicability_v102']=dict(request_intent_pinned=True,source_guard_chain_reviewed=True,
        installed_model_runner_source_hash_matches_snapshot_in_16_streams=True,
        actual_sampling_flags_attested=False,all_load_or_operator_branches_closed=False,
        graph_rows_unchanged=[813,1297],branch_records_unchanged=362,semantic_modules_unchanged=24)
    write(O/'whole-scope-requirement-audit-v102.json',audit)
    write(O/'PROFILE-CODE-QUESTION-GATE-v102.json',dict(
        status='SAVED_CLIENT_AND_SCHEDULE_METADATA_THEN_ONLY_OPERATOR_SELECTION_AND_COST_GUARDS',
        questions=['Which current-load sampling operator graph does explicit zero temperature select in the pinned source domain?',
            'Does observed greedy kernel presence prove an all-greedy batch?',
            'Which probability/RNG/reader work precedes the greedy early return?',
            'Which conditional random-fusion byte budget must not be credited to this greedy workload?'],
        existing_target_source_bodies_only=True,new_target_source_files=0,new_public_documents=0,
        no_framework_wide_research=True,no_server_model_Torch_NPU_capture_or_implementation=True,overall_complete=False))
    write(O/'sampling-applicability-validation-history-v102.json',dict(refinements=[
        'v101 predecessor was revalidated PASS170 before this refinement. Completed v101 artifacts/finalizer were not changed or rerun.',
        'Historical launch receipt SHA matches parsed controller; its last prepare definition checks the exact captured client SHA before copy, and last client definition executes that path with the phase. Neither script was executed.',
        'Initial local assertion incorrectly divided 3712 entries evenly across 16 workers. It stopped before extraction/shared-state mutation; raw streams show 233 entries per DP0 worker and 231 per DP1 worker. Analyzer now validates each stream separately; no ordinal native attribution.',
        'All 16 original metadata files hash-match the existing index. Two pre-window request IDs have computed-prefix 0 and scheduled 29696 tokens per TP copy; eight later IDs occur in both measurement windows. Roles are source-consistent, not direct response/payload identity joins.',
        'All installed metadata records report model_runner source SHA 568c3049205b9caff5c656a7f5dd4c6e79fb358abe526f89fb9fb3f70a60a43c, matching this local snapshot. Other loaded source/flag identities remain conditional; the v55 NPUInputBatch historical_control_match=False is preserved without calling it a mismatch.',
        'Decoded v55 NPUInputBatch defines only __init__, imports base InputBatch and initializes both request sets. The source MRO resolves operator guard construction, not per-call runtime state.',
        'Four set-presence truth rows and 64 finite valid greedy-only admit/remove reference transitions checked. A mixed set still enters the greedy kernel; an empty set has both flags True. Neither positive kernel presence nor all_greedy alone proves the nonempty early-return domain.',
        'v33 pre-return entropy/synthetic/error/readers and v65 byte models are reused, not new discoveries. v101 48TV random-private-plane reference is zero incremental savings when that post-return body is unreachable.',
        'No candidate/semantic branch/native regime/pair/group/graph count changed; no physical numeric ceiling, MFU or whole closure claimed. All local jobs terminal.'],overall_complete=False))
    updates={'F35':dict(review_v102=dict(evidence=names,saved_request_ids=10,workers=16,saved_execute_entries=3712,
        frozen_temperature_zero=True,actual_runtime_guard_attested=False,
        reference_set_transitions=64,existing_v65_costs_reused=True,unexecuted_branch_fusion_savings=0,
        implementation=False,native_test=False,new_runtime_executions=0),conclusion_changing_gap=gap)}
    mods={
        'M17':dict(candidates=['F35'],finding='Frozen explicit-temperature-zero intent is capture-bound. Valid nonempty source-MRO domain skips constraints and post-return random/recovery work; pre-return entropy/synthetic/errors/readers remain. No actual flag promotion.',evidence=names,gap=gap),
        'M16':dict(candidates=['F35'],finding='Current tail/head demand costing must use the selected sampling graph and required K rows. Existing pure-ID interfaces stay reader/alias/comparator conditional; nonexecuted random recovery contributes no additional current saving.',evidence=names,gap=gap)}
    remaining=list(read(O/'module-coverage-gaps-v101.json')['remaining'])
    remaining.append('v102: Historical controller/client and 16 saved scalar streams bind explicit-zero request intent, ten IDs (two pre-window prefix-consistent/eight measurement-consistent), DP0=233 and DP1=231 execute entries per TP worker. Nonempty valid all-greedy source chain is derived, not actual per-call flag attested; 16 installed model_runner hashes match local snapshot but NPU/base/params/helper runtime identities and metadata/admission closure remain conditional. Do not credit unreachable post-return random/recovery fusion; preserve pre-return entropy/synthetic/errors/raw/logprob/processors. No new regimes or counts:362/24 branches,813/1297/46 graph rows/hints,131/25/464 relations,2073/285598 conditional9/1352 pending849PMU16phase3136ambiguity,0 whole closed/no all-module MFU.')
    report='''本轮只补会改变算子候选适用范围与收益计费的证据，没有扩展框架研究，也没有登录服务器、新采集、设备测试或修改目标代码。

已有采集的客户端文件SHA与历史控制器的冻结校验一致；控制器SHA也与启动回执一致。该客户端所有请求共用temperature=0的请求体，包括前缀种子和C8测量请求。原始16份调度元数据共3712次execute进入：DP0每个TP进程233次，DP1每个231次。记录共10个请求ID，其中2个只在首个窗口之前出现，与前缀种子一致；8个后续请求都进入两个测量窗口，与有效的C8计数一致。这些是保存事件面的范围核对，不能把TP副本累计成独立token工作，也没有按跨卡序号关联原生算子。

定点源码只核对了零温度如何选择算子。显式0不会被默认温度替换；SamplingParams把它归为GREEDY。现有NPUInputBatch源码只定义构造函数，初始化greedy/random集合并继承对应的请求登记和采样元数据方法。在“所有有效活动请求均遵循该合同、源码绑定及元数据版本正确、greedy集合非空”的条件下，all_greedy=True且all_random=False。此时约束helper跳过温度/top-k/top-p处理，拒绝采样helper执行ArgMax与greedy输出后，提前返回，不执行后续processed softmax、普通uniform、recovery或random/block验证。

这仍不是实际逐调用内部标志的直接取证。16份installed记录中的model_runner源码SHA均与当前快照一致；NPUInputBatch来自已有v55回执，其historical_control_match=False没有被改写成已绑定或已不一致。其他加载实现、请求登记闭包、元数据版本以及dummy/未绑定阶段仍有条件。已观测到的72条greedy链也不能单独证明all_greedy，因为混合批次同样会执行该greedy分支。空集合还会让all_greedy和all_random同时为True，不能套用非空批次的提前返回结论。

因此，随机回退融合应保留为条件分支分析；在上述greedy域内，不能把未执行分支的48·T·V私有平面预算计作本次可节省流量。该预算来自v101，本轮没有重新认领为新发现。当前应优先评估已记录的Index→Cast→working clone→ArgMax→拒绝输出链。v65的独立物化参考22·T·V字节、纯ID参考2·T·V字节及其20·T·V接口差额仍需全部读者、精度和别名条件，既不是HBM流量，也不是确定可达收益。

温度0也不能删除所有概率或随机相关工作。raw/working处理与快照在helper之前；可选ori softmax、synthetic与block不兼容的错误检查、输出Fill及synthetic uniform都在提前返回之前。v33已经登记这些条件，本轮将其与负载适用范围连起来，没有用greedy推断它们全部关闭。融合必须先确定实际读者与副作用，再决定删除、保留或重算。

理论上限维持符号状态：在必须求出K个独立有限词表行精确ID、没有缓存winner或可用结构的比较模型中，最坏需要K·V次score观察和K·(V−1)次比较。K应由真实需求决定，不能用padding容量代替；2·K·V只是选定BF16源接口参考。各存储层、比较执行与必要全局ArgMax通信的能力上界仍需同口径证据，不能据此给物理数值上限或提速倍数。

离线前置复核通过194项；四种集合真值条件和64次有限合法登记/移除参考转换均核对。本轮没有新增候选、分支、原生regime或图节点：仍为362条分支记录、24个语义模块；中央131条条件关系、25组联合合同、464对待汇入或复核。9个regime/1352次执行的关键字段、最低模块MFU和整体物理理论上限仍未补齐，整体目标保持进行中。
'''
    action=('继续原Goal，v102PROGRESS。先读sampling-applicability-summary-v102/report/guard-contract/case-fusion-routing/wholeaudit。'
        'Operator fusion/split/reorg and theoretical ceilings only,profilefirst then pinpoint code. Frozen client temp0 newly capture-bound via historical controller/clientSHA;16savedstreams3712enter,DP0each233DP1each231,10IDs2prewindowseedconsistent8measurementconsistent. '
        '16installed model_runner source hashes match local snapshot; no actual effective sampling boolean attestation. Valid sealed nonempty all-greedy MRO domain skips constraints and post-return random recovery, but pre-return entropy/synthetic RNG/error/raw/processors/logprob observers remain. Mixedalsoentersgreedykernel;emptysetsbothflagsTrue. '
        'Do not repeat temperature/API/MRO survey or v33/v65/v101 resolved analysis; unexecuted recovery48TV gives0current saving, not all probability work0. Existing Index/Cast/workingclone/ArgMax/Fill/greedy output candidates need readers/precision/alias/epochs; requiredKnotcapacityT. '
        'No new counts:362branches24semantic,813/1297/46partialgraphs,131relations25groups464pending,2073regimes285598executions conditional9/1352pending849PMU16phase3136ambiguity,0wholeclosed/no allmoduleMFU. '
        'Next choose a genuinely unexpanded operator dataflow/conditional frontier or conclusion-changing native operand/capacity evidence from existing saved assets; do not enumerate arbitrary empty pairs, frameworks, or repeatedly relabel unknown runtime fields. Key9operands still require minimal bounded evidence plan v98, not more whole-repo code. '
        'All local jobs terminal; no server/model/Torch/NPU/capture/service/environment/implementation work. GoalACTIVE,TaskACTIVE,Loop004EVALUATING. Verify_sampling_applicability_v102 mustPASS; never rerun completed finalizer or alter completed artifacts.')
    inputs=[Path(r['path'])for r in read(O/names[0])['inputs']]
    extracted=O/'sampling-applicability-source-v102/npu_input_batch.py'
    jobs=[dict(kind='profile',status='pass',description='Reuse frozen captured client/controller receipt and hash-verified 16 saved schedule streams; bound ten IDs and source sampling-guard domain without runtime-flag promotion.',artifacts=[O/n for n in names]+inputs+[extracted]),
        dict(kind='design-check',status='pass',description='Check four request-set truth rows,64 valid reference transitions, pre-return effects and zero savings credit for an unreachable random body; symbolic ceilings and existing v65 interfaces retain all-reader gates.',
            artifacts=[O/'sampling-applicability-analysis-preflight-v102.json',O/'sampling-applicability-validation-history-v102.json',O/'whole-scope-requirement-audit-v102.json'])]
    persist_refinement(version=102,previous=101,title='Frozen-load sampling operator applicability and fusion cost routing',
        summary=dict(filename='sampling-applicability-summary-v102.json',previous_turn_classification='PROGRESS:v101 conditional fallback operator bodies and precision contracts;170 postchecks',
            saved_request_ids=10,saved_worker_streams=16,saved_execute_entries=3712,explicit_zero_request_intent_pinned=True,
            actual_runtime_guard_attested=False,reference_set_transitions=64,new_target_source_files=0,new_public_documents=0,
            refined_candidates=1,refined_modules=2,new_semantic_families=0,branch_records=362,semantic_modules=24,
            graph_node_rows=813,graph_edge_rows=1297,structural_pair_hints=46,new_pairs=0,new_groups=0,
            relation_clauses=131,higher_order_groups=25,pending_central_pair_reconciliation=464,
            scoped_reviewed_regimes=2073,scoped_reviewed_executions=285598,pending_regime_reconciliation=9,pending_execution_reconciliation=1352,
            new_native_promotions=0,actual_MFU_numeric_physical_ceiling_or_speedup_proved=False,implementation=False,native_test=False,
            remote_job_status='All local saved-profile/source-contract checks finished; no server/model/Torch/NPU/capture/service/environment or implementation work.'),
        changes=['Bind explicit-zero frozen request intent to historical controller/client pins and ten saved request identities in16worker streams.',
            'Resolve only operator-selection MRO/guards; distinguish valid nonempty all-greedy, mixed and empty-set cases from unrecorded actual flags.',
            'Exclude unreachable random-recovery budget from current greedy-domain savings; retain pre-return observers and required-K symbolic theory.',
            'No candidate/branch/pair/graph/native count promotions; all-module MFU and whole physical ceilings remain open.'],
        candidate_updates=updates,module_updates=mods,semantic_branches={'M17':branches},remaining=remaining,
        action=action,jobs=jobs,report=report,read_first=names+['whole-scope-requirement-audit-v102.json'],
        publication_rows={},gate_updates=dict(overall_complete=False),
        scripts=[N/'analyze_sampling_applicability_v102.py',N/'verify_sampling_applicability_v102.py',N/'persist_profile_refinement_v90.py'],
        executor_prefix='sampling-applicability',main_script=Path(__file__),source_read=False)

if __name__=='__main__':main()
