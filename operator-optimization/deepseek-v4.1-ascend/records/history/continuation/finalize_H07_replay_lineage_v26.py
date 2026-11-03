from copy import deepcopy
from datetime import datetime, timezone, timedelta
import hashlib, json
from finalize_origin_coverage_v9 import N, O, R, W, T, L, read, write, ctl


def payload(label, embedded=False):
    receipt = read(N / (label + '.json'))
    assert receipt['returncode'] == 0, label
    obj = json.loads(receipt['stdout'])
    if embedded:
        encoded = (json.dumps(obj['result'], ensure_ascii=False, separators=(',', ':')) + '\n').encode()
        assert hashlib.sha256(encoded).hexdigest() == obj['result_sha256'], label
    return obj


def main():
    assert not (O / 'REVIEW-CHECKPOINT-v26.json').exists(), 'refuse duplicate finalization'
    now = datetime.now(timezone.utc)
    stamp = now.isoformat()
    assert read(N / 'H07-profile166-v26.json')['returncode'] == 1
    prof = payload('H07-profile166-v26-v2')
    src = payload('H07-source166-v26', True)
    audit = payload('H07-domain-audit166-v26', True)
    ps, ds = prof['summary'], audit['result']
    assert (ps['external_target_embedding_COMM_OPs'], ps['persistent_embedding_copy_APIs'],
            ps['graphs_bound_to_one_root'], ps['prepare_roots']) == (88, 88, 64, 96)
    assert ds['source']['sha256'] == prof['result_sha256']
    assert ds['primitive_line85_API_records'] == ds['primitive_unique_roots'] == 88
    assert ds['all80_graph_prefix_gather_domains'] == {'"24,5120"': 80}
    files = src['result']['files']
    embedding = files[0]['functions']
    assert len(embedding) == 1
    assert embedding[0]['owners'] == ['UnquantizedEmbeddingMethod']
    assert embedding[0]['start'] == 84 and embedding[0]['end'] == 85
    assert 'F.embedding(input_, layer.weight)' in '\n'.join(embedding[0]['source'])
    assert len(files) == 5
    refs = [dict(label=label, receipt=str(N / (label + '.json')),
                 full_result=obj['full_result'], sha256=obj['result_sha256'])
            for label, obj in [('H07-profile166-v26-v2', prof), ('H07-source166-v26', src),
                               ('H07-domain-audit166-v26', audit)]]
    write(O / 'H07-exact-profile-lineage-v26.json', dict(
        generated_utc=stamp, evidence_grade='MEASURED', profile_summary=ps,
        exact_domain_audit=ds, evidence_refs=refs,
        lineage_scope='Original native API/COMM flow, exact same-thread containing execute marker and unique prepare root. No nearest-time/ordinal association.',
        lineage_is_not_tensor_identity=True, no_new_device_work=True, overall_complete=False))
    contract = dict(
        generated_utc=stamp, evidence_grades=['MEASURED', 'DOCUMENTED', 'DERIVED', 'UNKNOWN'],
        profile_trigger='88 external target embedding collectives and 88 copy_ calls; exact 64 replay-root associations leave producer purity and buffer consumers unanswered by trace.',
        files=[dict(path=f['path'], sha256=f['sha256']) for f in files],
        source_excerpt_receipt=refs[1], reuse_capture_contract='profile-raised-source-contract-v16.json',
        local_questions_answered=[
            'All 88 recorded external primitive calls reach upstream vocab_parallel_embedding.py:85, owned by UnquantizedEmbeddingMethod.embedding:84-85.',
            'This method calls F.embedding(input_, layer.weight) without max_norm; installed Torch functional.embedding:2457 defaults max_norm=None and :2555-2566 guards weight renorm with max_norm is not None. The Python renorm concern is excluded in this recorded call branch.',
            'VL embed_input_ids:190-209 returns the language embedding unchanged when multimodal_embeddings is absent/empty; the nonempty merge branch must remain supported.',
            'The upstream persistent buffer has initialization, prompt copy/fill, multimodal preprocessing, prompt text insertion, dummy/capture and randomization uses. The Ascend override has prompt CPU fill and dummy/capture uses.',
            'In both named execute_model methods, local inputs_embeds is obtained from preprocessing and handed to forward. No direct inputs_embeds identifier handoff was found in the specifically named sample_tokens/propose_draft_token_ids methods. This is a bounded identifier result, not an alias/lifetime proof.',
            'Reuse v16: raw-ID capture selects inputs_embeds=None; cached ACL entry replays captured storage and ignores runtime args/kwargs. Raw IDs are independently required for Engram/history/routing.'],
        not_proved=[
            'Actual original captured-entry input bindings, its precise hit guard and every affected rank taking the same collective schedule.',
            'No future read of the external persistent-buffer generation through aliases, hooks, capture, branch transitions or padded tails before overwrite.',
            'All implicit event/dependency effects remain preserved when the external collective/copy is omitted.',
            'Any general purity statement for quantized or max_norm-enabled embedding variants, arbitrary native binaries, or all model branches.',
            'Removable critical-path latency or numerical equivalence of an unimplemented graph transformation.'],
        stop_rule='The Python max_norm question is answered. Do not expand into all Torch/backend internals to repeat a conditional candidate. Only inspect a specific alias/entry/event consumer if it changes the current conclusion.',
        target_implementation_changed=False, whole_consumer_closure=False, overall_complete=False)
    write(O / 'H07-primitive-and-buffer-contract-v26.json', contract)
    # These are analysis graphs only. The target implementation is unchanged.
    nodes = ['input_id_update', 'clamp', 'external_embedding', 'external_TP_allreduce',
             'persistent_buffer_copy', 'runtime_inputs_embeds_argument', 'cached_entry_call',
             'captured_raw_ID_embedding', 'Engram_metadata_history', 'graph_body']
    edges = [
        ('input_id_update', 'clamp', 'DATA'),
        ('clamp', 'external_embedding', 'DATA'),
        ('external_embedding', 'external_TP_allreduce', 'DATA'),
        ('external_TP_allreduce', 'persistent_buffer_copy', 'DATA'),
        ('persistent_buffer_copy', 'runtime_inputs_embeds_argument', 'DATA'),
        ('runtime_inputs_embeds_argument', 'cached_entry_call', 'CALL_ARGUMENT_IGNORED_ONLY_FOR_EXISTING_CACHED_ENTRY'),
        ('clamp', 'Engram_metadata_history', 'DATA_AND_STATE'),
        ('clamp', 'captured_raw_ID_embedding', 'CAPTURED_STORAGE_INPUT_CONDITIONAL'),
        ('cached_entry_call', 'captured_raw_ID_embedding', 'CONTROL'),
        ('captured_raw_ID_embedding', 'graph_body', 'DATA'),
        ('Engram_metadata_history', 'graph_body', 'METADATA_AND_STATE')]
    assert len(nodes) == len(set(nodes))
    assert all(a in nodes and b in nodes for a, b, _ in edges)
    pending = set(nodes)
    while pending:
        ready = {n for n in pending if not any(b == n and a in pending for a,b,_ in edges)}
        assert ready, 'conditional illustration must be acyclic'
        pending -= ready
    variants = dict(
        generated_utc=stamp, candidate_id='H07', analysis_only=True,
        graph_scope='Source-consistent conditional data/control illustration, not a reconstructed tensor-pointer or event-complete native DAG.',
        baseline=dict(nodes=nodes, edges=[dict(source=a, target=b, kind=k) for a,b,k in edges]),
        variants=[
            dict(id='H07-A', status='CONDITIONAL;UNPRICED',
                 transformation='Under an existing raw-ID cached-entry and dead external-buffer-generation contract, omit only external embedding/TP collective/copy production; retain raw-ID updates/clamp, metadata/history, graph replay, all other live effects and persistent storage needed elsewhere.',
                 eligible_observed_control_lineage_roots=64, actually_validated_removal_roots=None,
                 excluded_from_blanket_removal='24 long-input roots; capture/dummy/prompt/MM/eager branches;16 graph executions without exact phase bindings.',
                 group_condition='Collective omission must be uniform across the affected TP8 communicator. Observed cohorts are not a future guard proof.'),
            dict(id='H07-B', status='CONDITIONAL_ALTERNATIVE;UNPRICED',
                 transformation='A differently captured inputs_embeds-consuming graph may reuse the external producer and omit internal embedding, preserving all raw-ID consumers and capture/update lifetimes.',
                 cannot_reuse_same_M_assumption='B2 external extent12 differs from captured graph embedding extent24. Establish padded-tail and consumer semantics separately.',
                 relation_to_A='Alternative producer ownership; do not add both savings.'),
            dict(id='H07-C', status='CONDITIONAL_COMMUNICATION_REORGANIZATION;UNPRICED',
                 transformation='Review embedding TP reduction with subsequent SP consumers as a separate redistribution candidate.',
                 requirements='Exact row ownership, TP/SP routing, all raw-ID/hidden consumers, padding, communicator and arithmetic contracts; no byte/count-only claim.')],
        physical_memory_or_latency_gain=None, source_contract='H07-primitive-and-buffer-contract-v26.json',
        profile_contract='H07-exact-profile-lineage-v26.json', overall_complete=False)
    write(O / 'H07-embedding-producer-variants-v26.json', variants)
    branch_data = [
        ('exact_external_to_root', 'MEASURED_LOCAL_QUESTION_CLOSED', '88 external COMM records uniquely link to 88 of96 roots through native/fullcaller and exact execute containment.'),
        ('decode_subset', 'MEASURED_LOCAL_QUESTION_CLOSED', '64 exact roots:16 scheduled12 and48 scheduled24;each has1 graph,1 external collective and1 copy_ API.'),
        ('long_input', 'MEASURED_RETAINED_BRANCH', '24 roots scheduled6400/6528/6668,8 each,have external embedding/copy and no bound graph;do not blanket remove.'),
        ('empty_input', 'MEASURED_RETAINED_BRANCH', '8 zero-token roots have no selected external embedding/copy or bound graph.'),
        ('phase_unbound_graphs', 'UNKNOWN_PHASE_RETAINED', '16 of80 graph executions remain unbound to an exact prepare root;do not guess decode/capture phase.'),
        ('embedding_extent', 'MEASURED_LOCAL_QUESTION_CLOSED', 'All80 graph prefixes GatherV2 output24x5120;B2 external12x5120 differs,B4 external24x5120 matches only extent.'),
        ('primitive_owner', 'MEASURED_AND_DOCUMENTED_LOCAL_QUESTION_CLOSED', '88 source-linked calls at upstreamline85 map to UnquantizedEmbeddingMethod.embedding.'),
        ('max_norm', 'DOCUMENTED_PYTHON_RENORM_EXCLUDED_IN_THIS_BRANCH', 'Current2-argument F.embedding call uses max_norm=None;guarded renorm is not taken. No universal embedding-purity claim.'),
        ('raw_id_capture_entry', 'DOCUMENTED_SOURCE_CONTRACT;RUNTIME_BINDING_OPEN', 'Reuse v16 raw-ID dummy and cached ACL replay branches;actual original entry bindings/guard/generation remain conditional.'),
        ('TP_uniformity', 'OBSERVED_COHORTS_REVIEWED;FUTURE_GUARD_OPEN', '8 cohorts cover complete TP8 rank sets0..7 or8..15;future collective omission must be group-uniform.'),
        ('prompt_and_MM', 'DOCUMENTED_RETAINED_BRANCH', 'Preserve nonempty multimodal merge,prompt positions and persistent-buffer fill/copy behavior.'),
        ('raw_ids_and_Engram', 'DOCUMENTED_RETAINED_EFFECT', 'Keep input clamp/rawIDs/positions/lookback/history and routing;do not skip whole preprocessing.'),
        ('buffer_lifetime', 'PARTIAL_SOURCE_LOCALIZATION;ALIAS_GENERATION_OPEN', 'Bounded direct identifiers localized;no proof every alias/future branch read is overwritten before becoming live.'),
        ('post_forward', 'BOUNDED_IDENTIFIER_RESULT;NOT_ALIAS_CLOSURE', 'No direct inputs_embeds identifier handoff in named sample/propose methods;execute forwards the local value. Indirect state remains open.'),
        ('copy_cost', 'UNPRICED_NATIVE_ID_LINK_OPEN', '88 copy_ APIs are real;0 direct matches against native TASK API IDs is a layer mismatch/absence of link,not zero device copy cost.'),
        ('event_schedule', 'UNKNOWN_RETAINED_CONDITION', 'Do not erase live stream/event dependencies by deleting a collective that appears data-dead.'),
        ('alternative_ownership', 'DERIVED_NONADDITIVE', 'External-skip and internal-skip are alternative ownership graphs;TP/SP reorganization needs its own contracts.'),
        ('gain_and_validation', 'UNKNOWN;NOT_IMPLEMENTED', 'No NPU experiment,model import,numerical comparison,critical-path gain or physical-bandwidth/MFU inference this round.')]
    branches = [dict(id=f'H07-v26-{i+1:02d}', branch=a, state=b, evidence=c) for i,(a,b,c) in enumerate(branch_data)]
    write(O / 'H07-condition-ledger-v26.json', dict(generated_utc=stamp, candidate_id='H07',
        conditions=branches, evidence_refs=refs, all_conditions_closed=False, overall_complete=False))
    model = deepcopy(read(O / 'candidate-profile-ledger-v25.json'))
    framework = deepcopy(read(O / 'framework-candidate-ledger-v25.json'))
    modules = deepcopy(read(O / 'module-profile-ledger-v25.json'))
    matrix = deepcopy(read(O / 'module-phase-branch-matrix-v25.json'))
    assert len(model['candidates']) == 34 and len(framework['candidates']) == 7
    h = next(c for c in framework['candidates'] if c['id'] == 'H07')
    h['status'] = 'EXACT_64_REPLAY_ROOT_ASSOCIATIONS_AND_PYTHON_RENORM_REVIEWED;REMOVAL_CONDITIONAL;UNPRICED'
    h['profile_evidence'] += [
        'v26:88 external embedding/copy roots=64 exact decode graph roots+24 long-input roots;8 additional empty roots.80 graph embedding prefixes all24x5120,16 remainphase-unbound.',
        'v26:8 exact observed TP8 cohorts,2 scheduled12 and6 scheduled24. Same-root control lineage does not prove tensor identity,deadness or rank-uniform future cached-entry guards.']
    h['constraints'] = [c for c in h['constraints'] if not c.startswith('Actual embedding primitive purity')]
    h['constraints'] += [
        'Recorded UnquantizedEmbeddingMethod calls F.embedding without max_norm,so its guarded Python weight renorm is excluded. Persistent-buffer alias/generation,event and actual cached-entry bindings remain conditional.',
        'Do not delete or stop maintaining the entire persistent inputs_embeds bank across prompt/MM/capture/eager transitions.',
        'B2 external12 and graph24 are different row domains;maintain padded tails and every raw-ID consumer. TP8 collective omission needs group-uniform guards.']
    h['minimal_missing_evidence'] = [
        'For an unconditional removal conclusion:bind the exact original raw-ID cached-entry/hit guard,all live alias/generation consumers and collective/event schedule for the claimed fixed workload domain.',
        'Current profile/source supports a conditional opportunity;do not require implementing it to record the analysis. Future numerical/performance checks belong to separately scoped implementation work.']
    h['review_v26'] = dict(profile='H07-exact-profile-lineage-v26.json',
        source='H07-primitive-and-buffer-contract-v26.json', variants='H07-embedding-producer-variants-v26.json',
        condition_ledger='H07-condition-ledger-v26.json', local_max_norm_question_closed=True,
        all_consumers_closed=False, whole_module_closed=False)
    for obj, parent, dest in [
        (model, 'candidate-profile-ledger-v25.json', 'candidate-profile-ledger-v26.json'),
        (framework, 'framework-candidate-ledger-v25.json', 'framework-candidate-ledger-v26.json'),
        (modules, 'module-profile-ledger-v25.json', 'module-profile-ledger-v26.json')]:
        obj.update(generated_utc=stamp, parent_ledger=parent, whole_modules_closed=0, overall_complete=False)
        if obj is model:
            obj['framework_candidate_ledger'] = 'framework-candidate-ledger-v26.json'
        if obj is modules:
            for m in obj['modules']:
                m['phase_branch_record'] = 'module-phase-branch-matrix-v26.json#' + m['id']
                if m['id'] in h['modules']:
                    m['review_v26'] = dict(finding='H07 exact replay-root subset,primitive max_norm guard and bounded persistent-buffer handoffs reviewed;entry/alias/generation/events remainconditional.',
                        evidence=['H07-exact-profile-lineage-v26.json', 'H07-primitive-and-buffer-contract-v26.json', 'H07-condition-ledger-v26.json'], whole_module_closed=False)
        write(O / dest, obj)
    matrix.update(generated_utc=stamp, parent_matrix='module-phase-branch-matrix-v25.json')
    for m in matrix['modules']:
        if m['id'] in h['modules']:
            m['branches'].append(dict(branch='H07 external embedding/copy versus captured raw-ID producer across exact roots and TP8 cohorts',
                state='EXACT_SUBSET_AND_PYTHON_RENORM_REVIEWED;ENTRY_ALIAS_EVENT_CONDITIONS_OPEN',
                evidence='H07-exact-profile-lineage-v26.json;H07-primitive-and-buffer-contract-v26.json;H07-condition-ledger-v26.json',
                gap='Originalcachedentryguard/binding,bufferaliases/futuregeneration,padded-tail andcollective-eventeffects.'))
    write(O / 'module-phase-branch-matrix-v26.json', matrix)
    remaining = deepcopy(read(O / 'module-coverage-gaps-v25.json')['remaining'])
    remaining[-1] = ('H07 now has exact64 decode-root associations,88 actual unquantized primitive calls,Python max_norm branch excluded,and directbuffer/handoff localization. '
        'Do not repeat these as wholly unknown. Conditional removal still needs originalentrybinding/groupguard andlivealias/generation/events. '
        'F33 andallotherboundaryconsumers remainopen;preserve3136candidate-setresource rows/849missingPMU/16phase-unboundgraphs/26theoryINCOMPLETE.')
    write(O / 'module-coverage-gaps-v26.json', dict(generated_utc=stamp, remaining=remaining,
        whole_modules_closed=0, overall_complete=False))
    summary = dict(generated_utc=stamp, status='PROGRESS', previous_turn_classification='PROGRESS:v25V41allocation/anchorcapacityexclusion/currentlog',
        external_embedding_collectives=88, persistent_embedding_copy_APIs=88, exact_decode_root_pairs=64,
        long_input_roots_not_blanket_removable=24, empty_roots=8, phase_unbound_graphs=16,
        all_graph_embedding_rows=24, exact_TP8_cohorts=8, unquantized_primitive_API_calls=88,
        Python_max_norm_renorm_excluded_in_recorded_branch=True, all_buffer_consumers_closed=False,
        actual_removal_validated=False, measured_end_to_end_gain=None, conditions=len(branches),
        model_candidates=34, framework_candidates=7, whole_modules_closed=0, overall_complete=False,
        evidence_refs=refs, all_jobs_terminal=True, no_new_model_import_NPU_health_or_service_work=True,
        remaining=remaining)
    write(O / 'H07-replay-lineage-summary-v26.json', summary)
    gate = read(O / 'PROFILE-CODE-QUESTION-GATE-v26.json')
    gate.update(status='PROFILE_EXHAUSTED_FOR_LOCAL_QUESTIONS;NECESSARY_POINT_SOURCE_REVIEWED',
        answered_by=['H07-exact-profile-lineage-v26.json', 'H07-primitive-and-buffer-contract-v26.json'],
        closed_scope='Exactexternal/replayrootassociation,actualprimitiveowner/max_normguard andbounded directbufferhandoffs.',
        not_closed='Originalcachedentrybindings/guard,allaliases/generations/eventeffects,wholemodulecoverage or gain.',
        stop_rule=contract['stop_rule'], overall_complete=False)
    write(O / 'PROFILE-CODE-QUESTION-GATE-v26.json', gate)
    report = '''本轮继续按“profile优先，无法定案再定点补代码”推进H07。保存的原生COMM/API完整调用链和同线程execute标记把88次图外target embedding归到88个唯一prepare root，并与88次持久inputs_embeds copy_相对应。96个root中，64个与具体decode图重放关联，24个为6400/6528/6668行的长输入执行，8个为空。没有靠最近时间、次数相等或执行序号补配。16个图执行仍缺精确prepare/phase绑定。

64个关联root分为16个12行和48个24行；它们各有一次图外embedding、copy_和一个图重放。全部80个图前缀的GatherV2输出均为24×5120，TP AllReduce count为122880。因此B=2时图外12行与图内24行并非相同完整张量域。八个精确调度cohort均覆盖完整TP8，分别在rank0–7与8–15上各有一组12行、三组24行。这是本次观测的同组调度证据，不是未来cached-entry命中条件一致的证明。删通信必须保持整个参与组的合法顺序。

Profile另保留856条目标embedding原生API、7条canonical调用链、1296条关联TASK，以及88条input-clamp原生API。actual primitive在88个不同root中都命中upstream vocab_parallel_embedding.py:85。仅据此定点读取方法，确认owner为UnquantizedEmbeddingMethod，调用F.embedding(input_, layer.weight)。安装版Torch的max_norm默认None，权重renorm只有max_norm非None才执行；这个Python权重重写疑问在当前记录分支内已回答，无需继续遍历整个Torch或编译器。该结论不推广到其他quant/max_norm分支或任意native副作用。

另一个profile不能独自回答的问题是持久缓冲区后续读者。只对两份runner中的self.inputs_embeds标识与execute/sample/propose命名方法交接点取局部上下文，复用v16已有raw-ID capture、model hidden-state选择及ACL cached replay合同。直接使用包括prompt复制与填充、多模态/文本embedding写回、dummy/capture及随机化；execute接收局部inputs_embeds后传给forward。sample/propose命名方法没有直接inputs_embeds标识交接，但这不证明不存在别名或间接state读者。VL方法在无多模态embedding时直接返回language embedding，有多模态时仍需merge。

因此H07-A的条件机会更具体：若既有图entry实际由raw IDs捕获、忽略本次runtime inputs_embeds，且被省略的外部缓冲代际在覆盖前没有活跃读者，那么可评估裁去这段图外embedding/TP collective/copy生产。仍须保留input-ID更新与clamp、Engram/lookback/history、所有图输入/元数据更新及其他活跃事件依赖。不能删整个_preprocess或整个持久buffer，不能把24个长输入root、prompt/MM/capture/eager和16个phase未绑定图一并当成已证明冗余。

H07-B是另一种producer所有权：另行捕获消费外部inputs_embeds的图后，才可能避免图内embedding。它与H07-A是替代关系，不能叠加收益，B=2填充尾部也须另审。Embedding TP→SP通信重组保留为独立条件分支。候选图只在分析资料中表达，没有修改目标代码、部署或运行优化。

88个copy_ API是真实记录；在native TASK link字段中直接拿其API rowid匹配得到0条，原因是关联层级不同或缺少该直接link，不能称没有device copy或耗时为0。821262条TASK缺该可选link字段，也保留为unlinked，未制造来源。首版查询曾因此KeyError停止；原脚本和失败receipt保留，修正版完成，并用独立固定脚本逐root核对行域、copy形状、primitive与TP cohort。

H07仍是条件候选，未给可删除耗时、物理带宽、MFU或端到端加速数字。新增18项分支/条件记录并更新相关模块。总台账保持34项模型候选、7项框架候选；26个整模块均未据此宣告闭合，整体目标继续active。其余跨模块/卡/迭代消费者、CPU成本及同域理论证据缺口保留。此次仅离线数据与限定源码读取，没有新增NPU负载、服务变更或健康检查；最后健康证据仍为v12post。
'''
    (O / 'PROFILE-FINDINGS-v26.md').write_text(report)
    action = ('继续Loop004，v26为实质进展。先读H07-replay-lineage-summary-v26.json、H07-exact-profile-lineage-v26.json、'
        'H07-primitive-and-buffer-contract-v26.json、H07-condition-ledger-v26.json、module-coverage-gaps-v26.json。'
        '88external/copy=64exactdecodegraphroots+24longinputroots，另8empty。80graphGather均24x5120，16phaseunbound。'
        'B2external12!=captured24，B4external24；8cohorts完整TP8，未来skipcollective仍须groupuniformguard。'
        '实际88primitiveAPI指向upstreamline85 UnquantizedEmbeddingMethod→F.embedding两参；max_normNone所以Pythonrenormguard不走，局部问题已回答，勿继续扫Torch/backend。'
        '两runner持久bank直接用途和execute交接已局部定位，无sample/propose直接标识不是aliasclosure。'
        'H07-A只在exactrawIDcachedentry+deadbuffer-generation+完整event/group合同内裁外producer；H07-B另inputs_embeds图裁内部，两者替代非叠加；保留prompt/MM/eager/capture/clamp/Engram/history/paddedtail。'
        'copy_88而nativeTASK直接link0只是跨层ID不匹配/无直接link，非copycost0。首版optionalTASKlinkKeyError保留，v2成功；无nearest/ordinal配对。'
        'Profile结果M/H07-external-embedding-copy-and-exact-replay-chains-v26-v2 SHAe98f77e2b4da4b42c0f085f6d87eda1ce256751da2a4423dbbff5a454f174791；'
        'sourcecontract SHAd62a28af6c0ce7688c2c6748bbebe399b58986fc2def455afca65d82950989f4；domainaudit SHAc32450a632e04f4d73379cc3dfa025af22be54043492c7a7dd27426bfabbd44e。'
        '下一批优先转F33/C1跨层双projection或尚未闭合模块边界，用已保存profile/fullcaller/shape/consumer证据先定位；'
        'H07条件分析已足以登记，除非某个entry/alias/event缺口影响更强结论，不必重复全量方法或为analysis要求实现优化。'
        '保留v25V41SWA-anchor下ring单独减半allocation收益排除；F34-A另owner/ABI条件。'
        '34model/7framework/0wholemodule；3136candidate-set资源/849missingPMU/16unboundgraphs/26theoryINCOMPLETE不变。'
        '当前EngineCore/producerready/prefixhash成本若决定性仍需既有有限授权freshhealth/ownership/frozenworkload/clone/restore后取证，不能用旧PID或假定空闲。'
        '全部v26 CPU-only工作terminal，无modelimport/NPU/服务/health操作，lasthealthv12post，Goalactive，整体未完成。')
    jobs = [
        ('profile', 'error', 'Initial H07 query stopped on an optional native_linked_API_rowids field; retain KeyError receipt and script; no nearest-time repair.',
            [N/'query_H07_embedding_replay_chains_v26_166.py', N/'H07-profile166-v26.json']),
        ('profile', 'pass', 'Corrected saved-evidence query preserves absent TASK links and binds88 external collectives/copies to roots;64 exact graph pairs and all80 graph prefixes retained.',
            [N/'query_H07_embedding_replay_chains_v26_166_v2.py', N/'H07-profile166-v26-v2.json']),
        ('review', 'pass', 'Profile-raised named primitive/max_norm and bounded persistent-buffer/handoff contexts read;no model or Torch import;no whole-repository audit.',
            [N/'read_H07_buffer_and_primitive_contract_v26_166.py', N/'H07-source166-v26.json']),
        ('review', 'pass', 'Prepared mechanical audit verifies96 roots,88 primitive roots,all copy extents,80 graph Gather domains and8 complete TP8 cohorts from the SHA-pinned saved result.',
            [N/'audit_H07_exact_domains_v26_166.py', N/'H07-domain-audit166-v26.json']),
        ('review', 'pass', 'Main reconciles Python-renorm local exclusion,conditional external/internal producer ownership,18 condition branches and explicit entry/alias/event/gain gaps.',
            [O/'H07-exact-profile-lineage-v26.json', O/'H07-primitive-and-buffer-contract-v26.json', O/'H07-embedding-producer-variants-v26.json', O/'H07-condition-ledger-v26.json', O/'PROFILE-FINDINGS-v26.md'])]
    runs = []
    for i,(kind,status,desc,arts) in enumerate(jobs):
        rid = 'run-' + (now-timedelta(seconds=len(jobs)-i)).strftime('%Y%m%dT%H%M%SZ')
        run = ctl('new-run', '--loop-dir', str(L), '--run-id', rid,
            '--case-key', 'c8-input32k-output1k-prefix90-166', '--kind', kind,
            '--command', 'Preparedv26retrospectiveregistration: '+desc,
            '--code-ref', 'Saved166profile and named dependency source;no implementation/NPU/modelimport/servicework',
            '--working-directory', str(N))
        args = ['finish-run', '--run-dir', run, '--status', status,
                '--correctness', 'not-applicable', '--summary', desc]
        for p in arts:
            assert p.exists(), p
            args += ['--artifact', str(p)]
        ctl(*args)
        runs.append(dict(run_dir=run, status=status, description=desc))
    cp = dict(status='PROGRESS', previous_turn_classification=summary['previous_turn_classification'],
        current_verified_changes=['Exact64 external/copy→decode replay control lineages;24 long-input/8empty roots and16unbound graph phases kept separate.',
            'Actual88 unquantized primitive calls and max_norm=None Python guard bound;directpersistentbuffer/handoffs localized without aliasclosure.',
            'H07 conditional producer alternatives and18branch conditions reconcile TPuniformity,padding,lifetime/events andunpriced copy/gain.'],
        summary=str(O/'H07-replay-lineage-summary-v26.json'),
        candidate_ledger=str(O/'candidate-profile-ledger-v26.json'), framework_candidate_ledger=str(O/'framework-candidate-ledger-v26.json'),
        module_ledger=str(O/'module-profile-ledger-v26.json'), branch_matrix=str(O/'module-phase-branch-matrix-v26.json'),
        report=str(O/'PROFILE-FINDINGS-v26.md'), source_gate=str(O/'PROFILE-CODE-QUESTION-GATE-v26.json'),
        coverage_gaps=str(O/'module-coverage-gaps-v26.json'), pending_full_scope_domains=remaining,
        remote_job_status='1retained query error;corrected profile/source/domain-audit jobs terminalpass. No model/NPU/service/healthwork.',
        next_action=action, model_candidates=34, framework_candidates=7, whole_modules_closed=0, overall_complete=False, updated_utc=stamp)
    write(O/'REVIEW-CHECKPOINT-v26.json', cp)
    for p in [L/'state.json', T/'index.json']:
        d = read(p)
        assert d['status'] in ['EVALUATING', 'ACTIVE']
        d.update(next_action=action, updated_at=stamp)
        write(p,d)
    for p in [L/'status.md', T/'status.md']:
        p.write_text('\n'.join(('- 下一步：'+action) if x.startswith('- 下一步：') else
            ('- 更新时间：`'+stamp+'`') if x.startswith('- 更新时间：') else x
            for x in p.read_text().splitlines())+'\n')
    (O/'profile-summary-before-v26.json').write_bytes((R/'profile-summary.json').read_bytes())
    p = read(R/'profile-summary.json')
    p['supplementary_H07_lineage_v26'] = dict(summary=cp['summary'], exact_decode_root_pairs=64,
        max_norm_Python_guard_closed=True, removal='CONDITIONAL;UNPRICED', whole_modules_closed=0, overall_complete=False)
    write(R/'profile-summary.json', p)
    for p in [N/'RECOVERY.json', W/'STATE.json', R/'STATE.json', R/'PARALLEL-RESUME-PACK.json']:
        d = read(p)
        d.update(latest_profile_first_review=cp, overall_complete=False, whole_modules_closed=0, updated_utc=stamp)
        d['next_read_only_action' if p == R/'STATE.json' else 'next_action'] = action
        if p == W/'STATE.json':
            d.update(profile_candidate_ledger=cp['candidate_ledger'], profile_module_ledger=cp['module_ledger'])
        if p in [N/'RECOVERY.json', R/'PARALLEL-RESUME-PACK.json']:
            d['read_first'] = [cp['summary'], str(O/'H07-primitive-and-buffer-contract-v26.json'),
                               str(O/'H07-condition-ledger-v26.json'), cp['coverage_gaps']]
        write(p,d)
    compact = ctl('compact', '--task-dir', str(T))
    validation = ctl('validate', '--task-dir', str(T))
    write(O/'task-run-registration-v26.json', dict(runs=runs, compact_output=compact,
        validation_output=validation, whole_modules_closed=0, overall_complete=False))
    arts = [p for p in O.glob('*v26*') if p.is_file() and not p.name.startswith('checkpoint-manifest')]
    arts += [N/(label+'.json') for label in ['H07-profile166-v26', 'H07-profile166-v26-v2', 'H07-source166-v26', 'H07-domain-audit166-v26']]
    arts += [N/name for name in ['query_H07_embedding_replay_chains_v26_166.py', 'query_H07_embedding_replay_chains_v26_166_v2.py',
        'read_H07_buffer_and_primitive_contract_v26_166.py', 'audit_H07_exact_domains_v26_166.py', 'finalize_H07_replay_lineage_v26.py']]
    arts += [p for p in N.glob('zcode-H07-*v26*20261031*') if p.is_file()]
    arts += [N/'RECOVERY.json', W/'STATE.json', R/'STATE.json', R/'PARALLEL-RESUME-PACK.json', R/'profile-summary.json', L/'state.json', T/'index.json']
    arts = sorted(set(arts))
    write(O/'checkpoint-manifest-v26.json', dict(generated_utc=stamp,
        artifacts=[dict(path=str(p), bytes=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in arts],
        whole_modules_closed=0, overall_complete=False))
    print(json.dumps(dict(checkpoint=str(O/'REVIEW-CHECKPOINT-v26.json'), runs=len(runs),
        pass_runs=4, retained_error_runs=1, manifest_artifacts=len(arts), validation=validation,
        model_candidates=34, framework_candidates=7, whole_modules_closed=0, overall_complete=False), ensure_ascii=False))


if __name__ == '__main__':
    main()
