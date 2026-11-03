from pathlib import Path
from datetime import datetime, timezone
from copy import deepcopy
import hashlib, json
from finalize_origin_coverage_v9 import N, O, R, W, T, L, read, write

def receipt(name):
    r = read(N/name)
    assert r['returncode'] == 0, name
    return json.loads(r['stdout'])

def main():
    assert not (O/'outer-consumer-summary-v11.json').exists()
    now = datetime.now(timezone.utc).isoformat()
    outer = receipt('profile-outer-segments-copies166-v11.json')
    copies = receipt('profile-copy-trace-patterns166-v11.json')
    installed = receipt('installed-fullhc-consumers166-v11.json')
    health = receipt('server-readonly-health166-v11.json')
    assert outer['summary']['outer_compute_records'] == 1221
    assert outer['summary']['graph_definition_location_variants'] == {'1':111}
    assert copies['summary']['exact_interval_trace_match_count_hist'] == {'1':561}
    assert all(c['graph_equals_eager_without_boundary_copies'] for c in copies['summary']['graph_eager_Engram_pattern_checks'])
    assert len(copies['summary']['graph_eager_Engram_pattern_checks']) == 8
    hashes = {x['path']:x['sha256'] for x in installed['result']['matched_files']}
    source_pins = {
      'vllm-ascend/vllm_ascend/models/deepseek_v41/model.py': '101f96b38e9fc47e7be147ddfe56976aee896867c73a33a7d95c3e3c833a6e91',
      'vllm-ascend/vllm_ascend/models/deepseek_v41/vl_model.py':'bb98a2e2a0e24c18aafe17229eb593937aaed2e37c1fd7cc2003c45ce872d282',
      'vllm-ascend/vllm_ascend/worker/model_runner_v1.py':'568c3049205b9caff5c656a7f5dd4c6e79fb358abe526f89fb9fb3f70a60a43c',
      'vllm/vllm/v1/worker/gpu_model_runner.py':'87c29d08c0bbf66993d8b984811e7325e35e6436242a773feb55ec88f16b2c51'}
    for rel, sha in source_pins.items():
        assert hashlib.sha256((W/'frozen-source-snapshot'/rel).read_bytes()).hexdigest() == sha
        assert hashes['/vllm-workspace/'+rel] == sha
    matches = installed['result']['matches']
    assert len(matches) == 90 and len(installed['result']['matched_files']) == 26
    groups = outer['first_outer_task_groups']
    assert sum(len(g['tasks']) for g in groups) == 111
    owners=[]
    for g in groups:
        loc=g['location']
        for i,t in enumerate(g['tasks']):
            entry=dict(temporal_location=loc, within_segment_ordinal=i, **t)
            if loc == 'BEFORE_FIRST_MARKER':
                if i < 15:
                    entry.update(module='M05', role='TP-vocabulary mask/index/embedding and reduction', consumer='SP input shard and initial mHC state', evidence_grade='DERIVED_PROFILE_PATTERN_AND_EXISTING_SOURCE_CONTRACT')
                elif i == 15:
                    entry.update(module='M05', role='Additional input zero-initialization', consumer='EXACT_FIELD_CONSUMER_NOT_BOUND', evidence_grade='MEASURED_LOCATION_AND_SHAPE_ONLY')
                elif i == 16:
                    entry.update(module='M06', role='Initial HC repeat after target SP partition', consumer='first target HC pre/post residual', evidence_grade='DERIVED_PROFILE_SHAPE_AND_SOURCE_BOUND_TARGET_REPEAT')
                else:
                    entry.update(module='M06', role='Initial pre_mix construction', consumer='first HC pre mix; preserve original coefficient/residual consumers', evidence_grade='DERIVED_PROFILE_PATTERN_AND_EXISTING_SOURCE_CONTRACT')
            elif loc in ['GAP_0_1','GAP_13_14']:
                entry.update(module='M14', role='Engram applied before configured source layer '+('1' if loc=='GAP_0_1' else '14'), consumer='next decoder HC input and residual; ready/lookup versions retained', evidence_grade='DERIVED_EXACT_TYPE_SEQUENCE_AND_SOURCE_BOUND_EAGER_CONTRACT')
            elif loc in ['GAP_36_37','GAP_37_38','GAP_38_39']:
                entry.update(module='M15', role='Auxiliary incoming-HC mean then TP gather before source layer '+loc.split('_')[-1], consumer='DSpark three-way auxiliary concat -> context FC (15360 input features)', evidence_grade='DERIVED_GRAPH_PATTERN_AND_OUTER_FORWARD_ORDER')
            else:
                assert loc=='AFTER_LAST_MARKER'
                roles=['full-HC TP gather','premix TP gather','full-HC buffer snapshot','HC cast to accumulation dtype','HC weighted product','HC component reduction','HC cast to output dtype','final full-row RMS normalization']
                consumers=['snapshot and collapse','collapse','getter returns buffer; Ascend v1 DSpark does not consume its contents','HC weighted product','HC reduction','output cast','final RMS normalization','selected logits and remaining normalized-hidden consumers']
                entry.update(module='M15', role=roles[i], consumer=consumers[i], evidence_grade='DERIVED_GRAPH_SHAPE_AND_SOURCE_BOUND_TAIL_CONTRACT')
            entry['not_proved']=['replay tensor/storage identity','all alias/event generations','removable critical-path time']
            owners.append(entry)
    assert len(owners)==111
    owner_ledger=dict(generated_utc=now, graph_replays=11, outer_compute_records=1221, outer_compute_definitions=111,
       definitions=owners, temporal_segments=outer['summary']['first_execution_groups'],
       scope='One fixed EP3 capture. Full task records remain in pinned server result; this ledger is an explicitly graded semantic overlay, not a direct replay API or tensor join.',
       exact_per_field_consumer_complete=False, noncompute_copy_records_separate=True, whole_modules_closed=0, overall_complete=False)
    write(O/'outer-compute-owner-ledger-v11.json',owner_ledger)
    method_contract=dict(
      scope='Recorded Ascend model_runner_v1 DSpark path, pure-text frozen workload. Not all implementations or external getter observers.',
      profile_trigger='The captured graph tail contains [24,20480] TensorMove before HC collapse, plus two TP gathers; eager full callchains bind the same full-HC producer contract. DSpark auxiliary context has 3x5120 input width.',
      question='Does the full-HC snapshot have a tensor-content reader on this exact Ascend DSpark call path?',
      search=dict(exact_symbols=['get_mtp_target_hidden_states','_mtp_hidden_buffer'], roots=installed['result']['source_roots'], matches=90, matched_files=26,
                  reason_for_installed_search='Frozen snapshot is partial; searching the installed roots for the two exact symbols avoids treating missing frozen files as missing consumers.', broad_source_audit=False),
      source_sha256=source_pins,
      narrow_reads=[dict(file='model.py',lines='950-992;1595-1605 (existing v10 contract);1798-1802',answer='Broad speculative allocation; forward copies full HC before collapse; getter is a pure buffer return.'),
                    dict(file='vl_model.py',lines='283-284',answer='Getter delegates directly to the language model.'),
                    dict(file='worker/model_runner_v1.py',lines='2052-2124',answer='Returned buffer replaces hidden_states only if method == mtp; both later branches use auxiliary context when enabled, otherwise the existing hidden_states. No later use of the mtp_hidden_states local.'),
                    dict(file='gpu_model_runner.py',lines='5190-5240',answer='Counterexample: another implementation can rebind unconditionally when the buffer is present; preserve its contract.')],
      scoped_answer='DOCUMENTED: On the inspected Ascend v1 DSpark path, the full-HC getter is invoked, but the returned buffer contents are not used to replace or construct target_hidden_states. Auxiliary hidden states are a separate live consumer. This settles the direct drafter-content question for this call path.',
      not_global_dead_state=True,
      other_paths_to_preserve=['Ascend method == mtp','GPU v1 unconditional consumer','GPU v2 model_runner and speculator callers','Ascend worker/v2/model_runner.py getter caller','other models or external/public getter observers'],
      variants=[dict(id='F26a',status='SCOPED_CONDITIONAL_CANDIDATE',change='Suppress only the unobserved snapshot write on the supported Ascend v1 DSpark path while retaining existing gather/collapse/norm order and addresses.',
                     numeric_effect='No arithmetic reordering in the proposal; no implementation or numerical validation performed.',guards=['Exact active implementation and speculative method','No additional registered/external content observer','Captured output/buffer/event lifetime and parameter update contract'],
                     logical_snapshot_bytes=24*20480*2,byte_status='DERIVED_LOGICAL_TENSOR_PAYLOAD_NOT_MEASURED_HBM_OR_GAIN'),
                dict(id='F26b',status='STRONGER_SEPARATE_CONDITIONAL_CANDIDATE',change='Avoid producing the unobserved raw-HC buffer; use the existing buffer-is-None branch that collapses locally and then gathers collapsed hidden.',
                     change_in_graph='Two full-HC/premix gathers plus copy may be replaced by local collapse and a collapsed-hidden gather, subject to proof; existing alternative is not a newly implemented optimization.',
                     guards=['All F26a observer/lifetime conditions','Exact BF16/FP32 product/reduction/rounding and lowering across changed local/global row shapes','TP group order, graph outputs/addresses, readiness and all normalized-hidden consumers'],
                     speedup=None)],
      remaining=['External/unrecorded observers and precise alias/event/lifetime contract for any edit','All normalized-hidden consumers and required row identities for row-select fusion','Numerical and collective proof for F26b','Measured resource/critical-path value'],
      historical_loaded_bytes_attested=False, code_changes=False, candidate_closed=False, legal_fusion_proved=False, overall_complete=False)
    write(O/'F26-consumer-contract-v11.json',method_contract)
    gate=read(O/'PROFILE-CODE-QUESTION-GATE-v11.json')
    gate['questions'][0]['allowed_search'] += ' Extended to the same two exact symbols in installed source roots because the frozen snapshot is partial; only returned active callsites/allocation/getter guards and one counterexample were read.'
    gate.update(actual_source_reads=method_contract['narrow_reads'], no_repository_wide_source_audit=True)
    write(O/'PROFILE-CODE-QUESTION-GATE-v11.json',gate)
    write(O/'PROFILE-CODE-QUESTION-REVIEW-v11.json',dict(generated_utc=now, workflow='PROFILE_FIRST_EXACT_CONCLUSION_CHANGING_QUESTION_ONLY',
        answers=[dict(question_id='F26_FULL_HC_SNAPSHOT_CONSUMER',answer=method_contract['scoped_answer'],contract='F26-consumer-contract-v11.json'),
                 dict(question_id='M14_CAPTURED_ENGRAM_MATCH',answer='8/8 exact type-sequence comparisons pass: graph38 equals source-bound eager40 without its two TensorMove endpoints. Separate MEMCPY tasks remain, so no boundary-copy elimination is inferred.')],
        source_pins=source_pins, source_read_does_not_supply_missing_values_or_timing=True, whole_modules_closed=0, overall_complete=False))
    memory=dict(generated_utc=now, summary=copies['summary'], full_result=copies['full_result'], result_sha256=copies['result_sha256'],
      graph_replays=11, copies_per_replay=51, matched_records=561, all_capture_MEMCPY_TASK_rows=2156,
      distribution=dict(before_first_marker=4,inside_40_marker_blocks=41,in_two_Engram_gaps=6,after_last_marker=0),
      bytes_available=False, pointers_available=False, values_available=False,
      interpretation='Temporal locations and exact task/event intervals only. No MEMCPY->TensorMove replacement, tensor identity, ready event or removable-time claim.',
      whole_modules_closed=0,overall_complete=False)
    assert sum(memory['distribution'].values())==51
    write(O/'graph-copy-accounting-v11.json',memory)
    candidates=deepcopy(read(O/'candidate-profile-ledger-v10.json'))
    candidates.update(generated_utc=now,parent_ledger='candidate-profile-ledger-v10.json',review_v11_utc=now,candidate_universe_complete=False,overall_complete=False)
    notes={
      'F02':'Graph prefix21 includes TP embedding/mask/reduction, one unresolved zero-init field, local [3,1,5120]->[3,4,5120] repeat and premix init. Target already partitions SP before repeat.',
      'F03':'111 outer compute definitions are placed in seven stable segments. Preserve Engram before source layers1/14 and auxiliary incoming-HC reads before layers37/38/39; no empty-boundary inference from marker blocks.',
      'F24':'Two graph gaps each contain38 ordinary Engram compute primitives; exact type sequence matches source-bound eager40 after excluding endpoint TensorMove from the compute comparison. Three additional MEMCPY tasks per gap remain separate and are not assigned by temporal proximity.',
      'F25':'Seven-segment ownership and separate51copies per replay refine cross-boundary constraints; all raw/normalized/auxiliary outputs and ready generations must still be preserved.',
      'F26':'Active Ascend v1 DSpark getter return is not used as tensor-content input because rebinding is guarded by method == mtp. F26a suppresses only unused snapshot write; F26b separately changes to local collapse + collapsed gather. Other MTP/GPU/v2 and external getter paths remain outside this exclusion.',
      'F27':'Graph source-layer37/38/39 incoming-HC mean/gather pairs feed the separate three-way DSpark context contract; do not conflate these live auxiliary states with the full-HC snapshot.'}
    for c in candidates['candidates']:
        if c['id'] in notes:
            c['review_v11']=dict(finding=notes[c['id']], evidence=['outer-compute-owner-ledger-v11.json','graph-copy-accounting-v11.json','F26-consumer-contract-v11.json'],complete=False)
            c.setdefault('evidence_files',[]).extend(['outer-compute-owner-ledger-v11.json','graph-copy-accounting-v11.json'])
        if c['id']=='F26':
            c['profile_review_state']='ASCEND_V1_DSPARK_RAW_SNAPSHOT_CONTENT_USE_EXCLUDED_ON_INSPECTED_PATH; NORMALIZED_CONSUMERS_AND_CHANGE_CONTRACT_OPEN'
            c['scoped_detail']['raw_snapshot_consumer_refinement_v11']=method_contract
            c['scoped_detail']['open']=[x for x in c['scoped_detail']['open'] if not x.startswith('All effective normalized-hidden and raw-MTP/getter consumers')]
            c['scoped_detail']['open'].append('Remaining normalized-hidden consumers and external/unrecorded raw-buffer observers; exact aliases/events; F26b numerical/collective lowering and measured value.')
            c['conclusion_changing_gap']='Normalized-hidden consumer/row closure and unrecorded observers; captured alias/event contract; F26b numerical/TP legality; same-domain performance value. Direct inspected DSpark drafter content reader is settled.'
            c['next_profile_or_contract_action']='Do not repeat the exact-symbol search. Bind remaining row/observer and capture-lifetime questions to recorded paths; supplement only conclusion-changing runtime fields.'
            c['conditional_variants_v11']=method_contract['variants']
            c['evidence_files'].append('F26-consumer-contract-v11.json')
    assert len(candidates['candidates'])==32
    write(O/'candidate-profile-ledger-v11.json',candidates)
    module_notes={
      'M05':notes['F02'], 'M06':notes['F03'], 'M14':notes['F24'], 'M15':notes['F26'], 'M18':notes['F27'],
      'M22':'561/561 graph MEMCPY rows have exactly one original trace interval match. Every replay has51copies; no bytes, pointers or values in recorded args. Copy/event and graph lifetime closure remains separate from compute coverage.',
      'M24':'Full-HC getter returned contents are unused on inspected Ascend v1 DSpark path; other public/external/implementation consumers and state/event generations remain scoped questions. Auxiliary versions stay live.'}
    modules=deepcopy(read(O/'module-profile-ledger-v10.json'))
    modules.update(generated_utc=now,parent_ledger='module-profile-ledger-v10.json',new_target_source_read=True,repository_wide_audit=False,overall_complete=False,whole_modules_closed=0)
    matrix=deepcopy(read(O/'module-phase-branch-matrix-v10.json'))
    matrix.update(generated_utc=now,parent_matrix='module-phase-branch-matrix-v10.json',all_phases_closed=False,all_conditions_closed=False,overall_complete=False,whole_modules_closed=0)
    for m in modules['modules']:
        m['phase_branch_record']='module-phase-branch-matrix-v11.json#'+m['id']
        if m['id'] in module_notes:
            m['review_v11']=dict(finding=module_notes[m['id']],evidence=['outer-compute-owner-ledger-v11.json','graph-copy-accounting-v11.json','F26-consumer-contract-v11.json'],whole_module_closed=False)
    for m in matrix['modules']:
        if m['id'] in module_notes:
            m['branches'].append(dict(branch='v11 captured outer compute / copy / exact consumer refinement',state='SCOPED_EVIDENCE_REVIEWED; FULL_MODULE_NOT_CLOSED',evidence=module_notes[m['id']],gap='See explicitly graded owner ledger, F26 scoped contract and remaining phase/alias/event/consumer gaps.'))
    write(O/'module-profile-ledger-v11.json',modules)
    write(O/'module-phase-branch-matrix-v11.json',matrix)
    remaining=[
      'One prefix zero-init field consumer is not exactly bound; semantic replay overlay is not a tensor/storage join. All51copies per graph still lack source/destination/bytes and ready generations.',
      'F26 direct Ascend v1 DSpark snapshot-content use is excluded; external/other implementation consumers, normalized-hidden row union, aliases/events and F26b numerical/collective lowering remain conditional.',
      'Initial graph0 schedule context, other15ranks per-call phase, valid masks/route receivers and native MC2 stages remain unbound.',
      'H02 exact metadata copy/pin/event fields, H01 on-CPU ready/read boundaries and H03 actual hash/frontend costs require profile-first follow-through; code cannot supply missing historical timing.',
      'All26module branch/resource/consumer and cross-layer/card/iteration coverage, same-domain ceilings/causes/global utilization ranking are not closed.']
    summary=dict(generated_utc=now,status='PROGRESS',parent_summary='graph-coverage-summary-v10.json',
      outer_compute=dict(replays=11,records=1221,definitions=111,segments_per_replay=7,segment_counts=[21,38,38,2,2,2,8],type_sequence_comparisons_pass=8,
                         definition_location_variants={'1':111},semantic_overlay='GRADED_PROFILE_PATTERN_PLUS_EXISTING_CONTRACT',all_tensor_consumers_closed=False),
      graph_memory=dict(records=561,per_replay=51,exact_trace_matches=561,bytes_or_pointer_fields=False,locations=memory['distribution']),
      F26_scoped_consumer_answer=method_contract['scoped_answer'],F26_variants=method_contract['variants'],
      installed_symbol_census=dict(symbols=method_contract['search']['exact_symbols'],matches=90,files=26,active_source_SHA_equal=True,broad_audit=False),
      health=dict(receipt='server-readonly-health166-v11.json',full_result=health['full_result'],sha256=health['result_sha256'],point_in_time=True,recheck_before_live_device_work=True),
      saved_server_results=[dict(path=x['full_result'],sha256=x['result_sha256']) for x in [outer,copies,installed]],
      remaining=remaining,no_new_inference_or_NPU_work=True,no_optimization_implementation=True,whole_modules_closed=0,overall_complete=False)
    write(O/'outer-consumer-summary-v11.json',summary)
    write(O/'module-coverage-gaps-v11.json',dict(generated_utc=now,remaining=remaining, next_action='Finish existing-profile H02 callsite/field census before any target-source read; supplement bounded runtime evidence only for actual conclusion-changing gaps.',whole_modules_closed=0,overall_complete=False))
    report='''本轮取得两项实质进展，整体优化分析仍未完成。

固定 EP3 profile 中，每次图的 111 条块外计算都已按位置、形状、已有源绑定合同分账，共 11 次、1221 条执行。七段依次为：输入/初始状态 21，两处 Engram 各 38，三处辅助状态 mean/gather 各 2，最终输出 8。台账逐条保留证据等级；前缀一条 ZerosLike 的精确字段消费者仍未知，图重放归属也不冒充张量别名证明。

两处 Engram 的 38 条类型序列，和同次采集源绑定 eager 40 条去除首尾 TensorMove 后完全相同，8 项比较全部通过。图内另有每次 51 条 MEMCPY，共 561 条，逐条按精确起止时间各命中原 trace 一条事件；其中 4 条在首块前、41 条在 40 个 marker 块内、6 条在两处 Engram 间隙。原事件只有任务/流/连接等 7 个字段，没有字节数、地址或值。因此不能把缺少 TensorMove 记录解释成复制消失，也不能把位置相邻当作复制来源。

F26 的问题来自 profile 尾部 [24,20480] 快照、两次 gather 和辅助状态路径。定点查询两个确切符号，覆盖完整安装源码而非不完整冻结快照，返回 90 处匹配、26 个文件；只补读实际调用点、分配/返回/分支及一个反例。四个已读取实现 SHA 与冻结证据一致。Ascend v1 只有 method == mtp 才把 getter 返回值替换成草稿 hidden；固定 DSpark 分支不读取这个快照的张量内容，而三份辅助状态仍是独立的真实输入。这个结论限定于该调用路径；GPU、Ascend v2、MTP 与外部 getter 观察者不能一并删除。

F26a 是只省去未观察到的快照写入，保留 gather、collapse、norm 顺序；F26b 是更强的独立条件候选，使用已有 buffer-is-None 分支先局部 collapse，再 gather collapsed hidden。F26a 仍需确认所有观察者与捕获生命周期；F26b 另需数值、形状 lowering、TP 顺序和全部归一化 hidden 消费者证明。单次快照逻辑载荷 983040 B，不是实测 HBM 流量或性能收益。本轮未实现候选，也没有性能加速结论。

健康只读检查显示本任务模型服务健康、容器/镜像与启动脚本符合预期，没有额外未归属 NPU 进程；之后只读取已有归档与安装源码，没有新推理或设备操作。新设备诊断前仍须重新核验占用。

接着追踪已有 24 个 prepare input 区间中的复制、锁页与同步的实际记录/调用来源；先尽用 profile，必要时才按缺失字段定点补代码。H01 CPU/ready、H03 hash/前端函数成本、跨卡同调用关联、初次图阶段、数值/别名与同域资源上界仍未闭合。26 个整模块仍未全部审完，Goal 保持 active。
'''
    (O/'PROFILE-FINDINGS-v11.md').write_text(report)
    print(json.dumps(dict(summary=str(O/'outer-consumer-summary-v11.json'),definitions=len(owners),copy_matches=561,consumer_search_matches=90,whole_modules_closed=0,overall_complete=False),ensure_ascii=False))

if __name__=='__main__':main()
