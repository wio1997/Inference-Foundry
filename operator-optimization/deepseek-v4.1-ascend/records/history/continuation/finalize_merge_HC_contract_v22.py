from datetime import datetime, timezone, timedelta
from copy import deepcopy
import json
import hashlib
from finalize_origin_coverage_v9 import N, O, R, W, T, L, read, write, ctl


def main():
    assert not (O / 'REVIEW-CHECKPOINT-v22.json').exists(), 'refuse duplicate finalization'
    now = datetime.now(timezone.utc)
    stamp = now.isoformat()
    labels = ['merge-HC166-v22', 'merge-HC-source166-v22', 'merge-hooks166-v22',
              'compute-attrs166-v22', 'merge-domain-review166-v22']
    hashes = ['67ea56947962067c41cd3942a6cc37d791dea4db9beec5c609d63eff226fd7db',
              'b1e32ce7e9e7534b9147c7ced944c9d23ad04f224fb269b358d747c2086f7db8',
              '34ef1f4c2104cc05423d44f6188b4c2d7f84aa9b69cecf54f92c26f673ec3342',
              '46e59d2de1773f4e8f9fd0e717b5ae68a99094644986d1613a8cc8b9caf6f147',
              '765ff128cff0fa1f0cdf6e068d60ed6f80d9c50711c044b182dcb231850150e5']
    data = {}
    for label, sha in zip(labels, hashes):
        receipt = read(N / (label + '.json'))
        assert receipt['returncode'] == 0 and not receipt['stderr']
        z = json.loads(receipt['stdout'])
        assert z['result_sha256'] == sha
        data[label] = z
    refs = [dict(label=k, path=data[k]['full_result'], sha256=data[k]['result_sha256'],
                 receipt=str(N / (k + '.json'))) for k in labels]
    profile = data[labels[0]]['summary']
    src = data[labels[1]]['result']
    hooks = data[labels[2]]['result']
    attr = data[labels[3]]['summary']
    domains = data[labels[4]]['result']
    arithmetic = read(O / 'merge-rounding-and-dependency-review-v22.json')
    assert profile['all_merge_executions'] == 4472
    assert profile['graph_merge_HC_pairs'] == 3200
    assert profile['graph_same_TASK_stream_counts'] == {'True': 3200}
    assert not profile['intervening_compute_op_counts'] and not profile['overlapping_COMM_OP_type_counts']
    assert len(domains['normalized_graph_pair_type_groups']) == 1
    assert domains['target_merge_row_M_absent_from_raw_MC2'] == [800]
    assert domains['affected_target_M800_merge_executions'] == 320
    assert domains['attribute_reuse_counts'] == {'same_raw_attrInfo_ID': 3200, 'same_attribute_text': 3200}
    assert attr['compute_executions'] == 286950 and attr['compute_types'] == 83
    assert sum(attr['attribute_lookup_execution_states'].values()) == 286950
    assert [(a['original_output'], a['rewritten_output']) for a in arithmetic['counterexamples']] == [(1.5, 1.5078125), (0.0, 0.00390625)]
    assert arithmetic['graph_review']['baseline_acyclic'] and arithmetic['graph_review']['proposed_graph_cyclic']

    findings = [
        dict(id='C22-1', grade='MEASURED', finding='All 4472 ForeachAddList executions are retained: 1271 DIRECT, 3200 GRAPH, 1 QUEUE, in 9 shape/context regimes. The 3200 graph pairs are exact structural layer-block pairs, not nearest-time native invocation guesses.', evidence=refs[0]),
        dict(id='C22-2', grade='MEASURED', finding='All 3200 graph ForeachAddList -> FFN HcPost pairs use the same recorded TASK stream, with 20/20/41 ns minimum/median/maximum boundary gap, no intervening ComputeInfo operator and no overlapping native COMM_OP interval in the gap. 2560 pairs are decode-compatible and 640 remain phase-unbound. This is visible-order evidence, not tensor-pointer identity or causal removable time.', evidence=[refs[0], 'shared-MC2-observed-order-contract-v20.json']),
        dict(id='C22-3', grade='DERIVED', finding='Removing global definition IDs leaves one typed graph-pair signature: BF16 [3,5120] merge -> BF16 [1,3,5120] HC input, with BF16 residual [1,3,4,5120] and FP32 post [1,3,4]/comb [1,3,4,4]. The original 680 full-definition signatures are not 680 shape groups.', evidence=refs[4]),
        dict(id='C22-4', grade='DOCUMENTED', finding='Pinned AscendMoERunner.forward calls torch._foreach_add([shared_output],[fused_output])[0] without alpha. Installed Torch static list-list declaration defaults alpha=1. Any nonunit runner routed scaling is applied earlier by the inherited helper, before the routed output transform and plain add. The effective runner gamma is not newly bound; the model config literal or raw attr_7 cannot substitute for it.', evidence=[refs[1], refs[2]]),
        dict(id='C22-5', grade='DOCUMENTED', finding='The current runner returns Tensor; the model legacy tuple muls_add/FP16 reciprocal-scaling path is a separate conditional domain. Runtime comm-dependent reduction hooks, feature-dimension trims, output transforms and optional ZeroExpertRouter contribution cannot be globally deleted. The final reduction override uses static layer/SP policy and context-aware custom ops rather than freezing a capture-time output_is_reduced boolean.', evidence=[refs[1], refs[2]]),
        dict(id='C22-6', grade='DERIVED', finding='320 target native merge tensors have M=800, while the raw target native MC2 M regimes are 816 x640 and 834 x320. This rejects a universal same-M raw-MC2-to-final-merge replacement. No individual M800 merge has been assigned to an M816/M834 invocation by ordinal or nearest time. The newly read _maybe_pad_hidden_states helper pads/trims feature dimensions, so it does not explain the token-row mismatch.', evidence=[refs[2], refs[4]]),
        dict(id='C22-7', grade='DOCUMENTED', finding='The decoder passes final MoE x, the attention-HC residual, FFN post and comb to HC post, then returns hidden_states and ffn_pre. A fusion must preserve all HC channels and the independent next-layer ffn_pre state; previous Engram/auxiliary consumer constraints also persist. Source flow does not close external alias, generation or native tile-ready obligations.', evidence=[refs[1], 'F26-consumer-contract-v11.json']),
        dict(id='C22-8', grade='DERIVED', finding='Two finite FP32/BF16-RNE arithmetic examples distinguish removing a materialization from removing a BF16 rounding boundary: bypassing merge rounding gives 1.5 vs1.5078125; collapsing conditional routed scaling and add gives 0 vs0.00390625. These refute unconditional identities in the explicitly stated arithmetic model, not actual captured-input reachability or native HC binary equivalence.', evidence='merge-rounding-and-dependency-review-v22.json'),
        dict(id='C22-9', grade='DERIVED', finding='With the observed producer order retained as a constraint, making MC2 completion wait for later SharedDown adds a cycle. Therefore an upstream MC2-epilogue-only merge that preserves this order is excluded in that limited domain. A redesigned kernel or producer schedule is a different domain, not mathematically excluded here.', evidence=['merge-rounding-and-dependency-review-v22.json', 'shared-MC2-observed-order-contract-v20.json']),
        dict(id='C22-10', grade='MEASURED', finding='Raw attrInfo->STRING_IDS lookup is retained for all286950 executions/83 types:121920 nonempty string candidates,165030 empty/placeholders,181 op/string groups. All3200 FFN HcPost rows reuse the exact attrInfo ID/text of preceding ForeachAddList. Attribute ownership/exporter association remains unbound; input_2:1 is not an HC parameter and no exporter bug is asserted.', evidence=[refs[3], refs[4]]),
        dict(id='C22-11', grade='DERIVED', finding='The summed BF16 merge-output interface payload is8135065600 bytes; a one-write/one-read model is16270131200 bytes, of which graph-only196608000 bytes. These are logical models across captured rank/process records, not actual HBM/network traffic, necessary I/O, request latency or achievable gain. Preserving BF16 rounding does not itself require a DRAM store.', evidence=refs[4]),
        dict(id='C22-12', grade='UNKNOWN', finding='Loaded native HC rounding/tiling, tensor pointers/all consumers, actual per-tile ready generations, post-hook effects for every replay and physical gain remain unproved. No legal native fusion or complete M06/M12/M13 closure is claimed.', evidence='M13-merge-HC-condition-ledger-v22.json')]
    source_files = []
    for group in [src, hooks]:
        for f in group['files']:
            source_files.append(dict(path=f['path'], sha256=f['sha256'], functions=[dict(name=q['name'], start=q['start'], end=q['end']) for q in f['functions']]))
    contract = dict(generated_utc=stamp, status='SCOPED_PROFILE_AND_SOURCE_CONTRACT_REFINED', findings=findings,
                    source_files=source_files, installed_torch_static_schema=hooks['installed_torch_static_schema'],
                    graph_boundary_shape_signature=domains['normalized_graph_pair_type_groups'][0],
                    current_add_alpha=1, current_effective_runner_gamma=None,
                    loaded_native_HC_rounding_proved=False, legal_native_fusion_proved=False,
                    measured_gain=None, whole_module_closed=False, overall_complete=False)
    write(O / 'merge-to-HC-contract-v22.json', contract)

    branches = [
        dict(id='MERGE-PRESENCE', condition='shared present / None', current='PRESENT_IN4472MERGES', judgment='Observed shared-present path; shared-absent returns routed output without foreach.', obligation='Retain the absent/shared/mix domain as a separate branch.'),
        dict(id='MERGE-RETURN', condition='runner Tensor / model legacy tuple', current='TENSOR_SOURCE_RETURN_BOUND', judgment='Current foreach runner path is separate from legacy tuple muls_add and FP16 reciprocal scaling.', obligation='Do not import tuple formulas into this plain-add path.'),
        dict(id='MERGE-ALPHA', condition='explicit/default foreach alpha', current='SOURCE_OMITS_ALPHA;STATIC_SCHEMA_DEFAULT1', judgment='Default alpha=1 in the pinned source contract.', obligation='Raw HC attr reuse is not independent scalar proof.'),
        dict(id='ROUTED-SCALE', condition='gamma==1 / gamma!=1 and dtype FP16/nonFP16', current='EFFECTIVE_GAMMA_UNBOUND;CAPTURED_MERGE_INPUTS_BF16', judgment='NonFP16 conditional branch scales routed in place before transform/add; FP16 shared branch scales shared reciprocally.', obligation='Preserve source dtype-rounding if absorbing scale; a boundary accepting already postscaled routed input need not invent gamma.'),
        dict(id='ROUTED-TRANSFORM', condition='input/output transforms present/None; pretransformed input', current='WRAPPERS_SOURCE_BOUND;GLOBAL_IDENTITY_NOT_ASSUMED', judgment='Input gather/readiness and output transform/join wrappers precede the add.', obligation='Preserve transform semantics, layout, reduction and event dependencies.'),
        dict(id='ROUTED-REDUCTION', condition='runtime MoECommType and static early-reduce policy', current='ASCEND_CONTEXT_AWARE_OVERRIDE_BOUND', judgment='ALLTOALL/MC2/FUSED_MC2 or ALLGATHER+SP may be reduced; local-DP shared mode requests early routed reduction.', obligation='Do not freeze capture-specialized output_is_reduced as replay-invariant.'),
        dict(id='SHARED-REDUCTION', condition='physical TP/SP/shared-replica parallel mode', current='FINITE_MODE_SUBTREE_FROMV21_REUSED', judgment='NonTP shared modes skip this helper; TP mode uses static early policy or context-aware custom reduction.', obligation='Preserve both branch contribution domains; shared DP flag is not overlap flag.'),
        dict(id='FINAL-HOOKS', condition='not SP/not early-reduce; feature truncation; ZeroExpertRouter', current='SOURCE_BRANCHES_BOUND;ALL_REPLAY_EFFECTS_NOT_CLOSED', judgment='Final context-aware reduce, last-dimension slice and optional zero-expert addition follow foreach.', obligation='Either include their effects before HC or prove identity in the stated execution domain.'),
        dict(id='ROW-DOMAIN', condition='graphM3 / target nativeM800,816,834 / draftM2,3', current='ALL9MERGE_REGIMES;M800_MISMATCH320', judgment='Same raw-MC2 and merge row domain cannot be assumed globally.', obligation='Bind valid-row/finalization mapping using exact calls; feature padding helper is not token-row mapping.'),
        dict(id='MERGE-ROUNDING', condition='materialize BF16 X / round BF16 in fused temporary / retain higher precision', current='BF16_INTERFACE_BOUND;NATIVE_ARITHMETIC_UNBOUND', judgment='Dropping materialization and dropping rounding are different transformations.', obligation='Preserve accepted numerical contract; N22-A/B are explicit-model counterexamples, not native validation.'),
        dict(id='HC-INPUTS', condition='x/residual/post/comb all ready for selected token-feature tile', current='CALLER_AND_SHAPES_BOUND;TILE_READY_UNBOUND', judgment='All4 HC streams/channels, coefficients and original accumulation order remain required under their contracts.', obligation='No partial-rank sum, expert-only completion or unproven external tile-ready protocol.'),
        dict(id='HC-CONSUMERS', condition='next-layer ffn_pre/outer Engram/auxiliary/final consumers', current='DECODER_RETURNS_FFN_PRE;PRIOR_OUTER_CONTRACT_REUSED', judgment='HC output and auxiliary pre_mix are distinct live results.', obligation='Retain aliases, fixed addresses and generations; ordinary source use is not whole-program all-consumer proof.'),
        dict(id='UPSTREAM-ONLY-FUSION', condition='retain MC2->SharedDown order and wait for SharedDown within MC2 completion', current='CONDITIONAL_DAG_CYCLE', judgment='Excluded only in that unchanged-producer-order domain.', obligation='A changed producer schedule or combined kernel is a different conditional candidate.'),
        dict(id='GRAPH-LIFETIME', condition='captured decode / unbound graphs / eager / replay generation', current='2560DECODE_COMPATIBLE;640UNBOUNDPAIRS', judgment='All3200 typed graph pairs reviewed structurally; phase not invented for640.', obligation='Retain stream/control events, graph addresses and cross-iteration buffer lifetime.'),
        dict(id='ATTRIBUTE-OWNERSHIP', condition='raw attrInfo lookup / actual own-op attribute / inherited API-context metadata', current='ALL3200_FFN_HC_REUSE_PRECEDING_MERGE_ID', judgment='Raw attribute lookup is complete for retained executions, semantic ownership is not.', obligation='Next source question is exact installed exporter attrInfo binding, not a broad model/SDK audit.'),
        dict(id='PAYLOAD-AND-GAIN', condition='logical interface / actual storage traffic / necessary I/O / request critical path', current='LOGICAL_BYTES_ONLY', judgment='8135065600 output bytes;16270131200 one-write/read model, not measured traffic or a lower bound.', obligation='Re-derive fused required work/I/O and capacity bounds; no sum-of-duration speedup.')]
    write(O / 'M13-merge-HC-condition-ledger-v22.json', dict(generated_utc=stamp, branches=branches,
          scope='Finite profile-raised merge/HC boundary subtree; neither all M13 nor all model branches.',
          source_contract_questions_answered=['default plain-add alpha', 'inherited scale location', 'Tensor/tuple separation', 'post-hook/feature-trim and HC caller state'],
          whole_module_closed=False, overall_complete=False))

    variants = [
        dict(id='F23-V22-A', arity='TWO_OPERATORS', graph='ForeachAddList -> HcPost', state='CONDITIONAL_CANDIDATE', domain='Final shared/routed add inputs are ready; all post-add transformations are identity or included; BF16 merge rounding and HC semantics/consumers preserved.', open=['Native HC exact contract/tiling', 'Aliases/generations/all consumers', 'External tile readiness and resource limits']),
        dict(id='F23-V22-B', arity='THREE_OPERATORS', graph='SharedDown -> ForeachAddList -> HcPost', state='CONDITIONAL_CANDIDATE', domain='Routed final output and HC residual/coefficients ready; preserve shared down dequantization/scales/output rounding, then plain-add BF16 rounding and HC order.', open=['Shared/routed finalization and row mapping', 'Down output contract and consumers', 'Native capacity and numerical compatibility']),
        dict(id='F23-V22-C', arity='MULTI_OPERATOR_REORGANIZATION', graph='Routed finalization + shared pipeline + merge + HC with a changed producer order', state='CONDITIONAL_DOMAIN_ONLY', domain='Allows changing producer schedule/communication boundary rather than pretending current MC2 epilogue can wait on later same-order work.', open=['New order/ready/progress protocol', 'Runtime reduction, row domain, native resource and numeric contracts', 'Graph lifetimes']),
        dict(id='F23-V22-D', arity='RESTRICTED_UPSTREAM_EPILOGUE', graph='Put merge into MC2 completion; retain current MC2 -> SharedDown producer-order edges', state='EXCLUDED_IN_STATED_DOMAIN', domain='Conditional DAG has MC2_COMPLETE -> SharedDown -> MC2_COMPLETE cycle.', open=['Does not exclude a redesigned kernel or changed order.']),
        dict(id='F23-V22-E', arity='NUMERICAL_REWRITE', graph='Delete BF16 merge/scaling boundaries or replace both by one high-precision expression', state='UNCONDITIONAL_IDENTITY_REJECTED_IN_EXPLICIT_ARITHMETIC_MODEL', domain='N22-A/B finite FP32/BF16-RNE model; actual native binary/effective gamma/reachable input domain unbound.', open=['Accepted tolerance or independently proved exact domain may allow a conditional change.']),
        dict(id='F23-V22-F', arity='EXISTING_LEGACY_BRANCH', graph='tuple muls_add / FP16 reciprocal path -> conditional collective -> HC', state='SEPARATE_SOURCE_DOMAIN;NOT_CAPTURED_FOREACH_GRAPH', domain='Reuse prior tuple contracts, not a new fusion or current alpha formula.', open=['Actual legacy-domain invocation and native contract if that domain is later in scope.'])]
    relations = [
        dict(kind='requires', a='F23-V22-A/B', b='Correct post-hook and row-domain finalization plus BF16 boundaries', evidence='merge-to-HC-contract-v22.json'),
        dict(kind='conflicts', a='F23-V22-D', b='Unchanged current MC2->SharedDown producer order', evidence='merge-rounding-and-dependency-review-v22.json'),
        dict(kind='interacts', a='F20/F21 backend/placement/reduction domain', b='F22/F23 final inputs and resource schedule', evidence='F20-F23-shared-overlap-domains-v21.json'),
        dict(kind='requires', a='Any cross-layer extension', b='F03/F26 ffn_pre, Engram and auxiliary consumer contracts', evidence='F26-consumer-contract-v11.json')]
    write(O / 'F23-merge-HC-variants-v22.json', dict(generated_utc=stamp, variants=variants, relations=relations,
          new_main_candidate_ids=False, implementation=False, legal_native_fusion_proved=False,
          measured_gain=None, overall_complete=False))
    write(O / 'raw-profile-attribute-intake-v22.json', dict(generated_utc=stamp,
          profile_summary=attr, raw_lookup_complete_in_retained_scope=True,
          per_operator_semantics_closed=False, evidence=[refs[3], refs[4]],
          FFN_HC_preceding_merge_ID_reuse=3200,
          forbidden_inferences=['Do not interpret preceding merge input_2:1 as HC parameter.',
                                'Do not treat positional attr literals as current own-op runtime values before source binding.',
                                'Do not infer native tiling/binary identity or zero cost from absent text.',
                                'Do not assert an exporter bug from ID reuse.'],
          next_profile_question='Which installed exporter path populates COMPUTE.attrInfo and how is its metadata attached to graph tasks?',
          next_source_gate='Only exact installed attrInfo producer/assignment path and one-hop schema/caller; first inspect already saved schema/exporter index.',
          overall_complete=False))

    model = deepcopy(read(O / 'candidate-profile-ledger-v21.json'))
    for c in model['candidates']:
        if c['id'] in ['F01', 'F02', 'F03', 'F20', 'F21', 'F22', 'F23', 'F26']:
            c['merge_HC_boundary_v22'] = 'merge-to-HC-contract-v22.json;F23-merge-HC-variants-v22.json;finite boundary/rounding/row/schedule review only.'
        if c['id'] == 'F23':
            c.update(profile_review_state='ALL4472_MERGES_AND3200_GRAPH_HC_BOUNDARIES_REVIEWED;FINITE_VARIANTS_AND_SOURCE_CONTRACT_REFINED',
                     conclusion_changing_gap='Loaded HC contract, exact aliases/ready generations, all consumers and per-replay post-hook effects remain open; raw attributes need exporter ownership binding.',
                     next_profile_or_contract_action='First bind saved attrInfo exporter semantics; only then use raw attribute strings. Preserve Tensor plain-add alpha1, earlier conditional scale, row finalization and BF16 boundaries.',
                     target_source_gate='ONLY_EXACT_PROFILE_RAISED_UNDECIDABLE_QUESTION',
                     scalar_value_bound_from_profile=False,
                     default_add_alpha_in_pinned_source_contract=1,
                     effective_runner_gamma=None,
                     candidate_closed=False, legal_fusion_proved=False)
    assert len(model['candidates']) == 33
    model.update(generated_utc=stamp, parent_ledger='candidate-profile-ledger-v21.json')
    write(O / 'candidate-profile-ledger-v22.json', model)
    framework = deepcopy(read(O / 'framework-candidate-ledger-v21.json'))
    framework.update(generated_utc=stamp, parent_ledger='framework-candidate-ledger-v21.json')
    write(O / 'framework-candidate-ledger-v22.json', framework)
    notes = {
        'M06': 'All3200 graph merge->FFN HC typed boundaries reviewed; preserve BF16 X, residual/post/comb, all HC channels and independent ffn_pre. Native binary/consumer closure not claimed.',
        'M12': 'Raw target MC2 M816/834 versus320 final merge M800 rejects universal same-M substitution. Preserve valid-row/finalization mapping; do not pair native invocations by time/ordinal.',
        'M13': 'Default alpha1 plain foreach, inherited pre-add scale and Tensor-vs-tuple domains bound;16 finite merge/HC branch obligations and6 analysis variants recorded.',
        'M22': 'All3200 graph pair gaps20..41ns have no intervening compute/COMM;640phase-unbound retained. Existing control overlap counts are pair-occurrences, not removable waits.',
        'M23': 'Runtime reduction/static early policy and shared parallel modes remain part of merge-domain semantics; current DP+MC2 schedule is not free asynchronous overlap.',
        'M24': 'Current-order upstream epilogue-only waiting creates a conditional DAG cycle. Changed schedule remains conditional; exact tile ready, aliases and graph generations not proved.'}
    modules = deepcopy(read(O / 'module-profile-ledger-v21.json'))
    matrix = deepcopy(read(O / 'module-phase-branch-matrix-v21.json'))
    for m in modules['modules']:
        m['phase_branch_record'] = 'module-phase-branch-matrix-v22.json#' + m['id']
        if m['id'] in notes:
            m['review_v22'] = dict(finding=notes[m['id']], evidence=['merge-to-HC-contract-v22.json', 'M13-merge-HC-condition-ledger-v22.json', 'F23-merge-HC-variants-v22.json'], whole_module_closed=False)
    for m in matrix['modules']:
        if m['id'] in notes:
            m['branches'].append(dict(branch='Profile-raised shared/routed merge -> HC boundary and conditional rewrite domains', state='FINITE_BOUNDARY_SUBTREE_REVIEWED;WHOLE_MODULE_OPEN', evidence=notes[m['id']], gap='See remaining aliases, generations, native arithmetic, all consumers and full-module coverage.'))
    assert len(modules['modules']) == 26 and not any(m['whole_module_closed'] for m in modules['modules'])
    modules.update(generated_utc=stamp, parent_ledger='module-profile-ledger-v21.json')
    matrix.update(generated_utc=stamp, parent_matrix='module-phase-branch-matrix-v21.json')
    write(O / 'module-profile-ledger-v22.json', modules)
    write(O / 'module-phase-branch-matrix-v22.json', matrix)
    remaining = [
        'Whole26-module finite branch/consumer/cross-operator-layer-card-iteration analysis remains incomplete. A closed local source question or finished report is not whole-module completion.',
        'First inspect saved schema/exporter index and point-read the exact installed COMPUTE.attrInfo assignment path: all raw strings are recovered, but HC reuses preceding-op metadata and own-op meaning is not yet bound.',
        'F23 default plain-add alpha1, inherited scale location, Tensor/tuple branch and post-hook sources are now bound; do not repeat them as wholly unknown. Loaded HC arithmetic/tiling, actual input readiness/aliases and all-consumer/per-replay post-hook effects remain open.',
        'Target merge M800 versus rawMC2 M816/834 requires exact valid-row/finalization mapping. Feature-dimension padding is not its explanation; no nearest-time or ordinal pairing.',
        'Current MC2==1 config disables existing shared multistream; the v21 explanation persists. Future compatible scheduling/fusion and altered-backend variants remain conditional, not free additive gain.',
        'F21 custom-map post-load weight/FP+packed-scale/capture-generation reconciliation, EP16 physical topology and same-call MC2 internal work/wait remain open.',
        'Current166 EngineCore/producer-ready/prefix-hash CPU costs remain absent. If decisive, use already authorized finite diagnostics with fresh health/ownership/frozen-workload/clone/restore guards.',
        'F33/H07/compressor/ring/rollback and other boundary consumers remain open. Preserve3136 candidate-set resource rows,849 missingPMU,16 unbound graph phases and26 theory statuses INCOMPLETE; no peak/MFU or request gain invented.']
    write(O / 'module-coverage-gaps-v22.json', dict(generated_utc=stamp, remaining=remaining, whole_modules_closed=0, overall_complete=False))
    summary = dict(generated_utc=stamp, status='PROGRESS', previous_turn_classification='PROGRESS:v21 current shared serial path explained by config mutex and fallback',
                   all_merge_executions=4472, graph_merge_HC_pairs=3200,
                   graph_boundary_gap_ns=profile['merge_to_HC_gap_ns'], normalized_graph_pair_types=1,
                   graph_pair_phase=dict(decode_compatible=2560, unbound=640),
                   current_source_add_alpha=1, current_effective_runner_gamma=None,
                   target_M800_merges_requiring_row_domain_preservation=320,
                   raw_attribute_executions=286950, raw_attribute_types=83,
                   graph_FFN_HC_preceding_merge_attribute_ID_reuse=3200,
                   numeric_model_counterexamples=2, merge_HC_condition_branches=len(branches), candidate_subvariants=len(variants),
                   point_source_files=3, installed_static_schema_files=1,
                   new_model_import_or_device_work=False, new_service_changes=False, all_jobs_terminal=True,
                   model_candidates=33, framework_candidates=7, whole_modules_closed=0,
                   legal_native_fusion_proved=False, measured_gain=None, overall_complete=False,
                   evidence_refs=refs, remaining=remaining)
    write(O / 'merge-HC-summary-v22.json', summary)
    gate = read(O / 'PROFILE-CODE-QUESTION-GATE-v22.json')
    gate.update(answered_by=['merge-to-HC-contract-v22.json', 'M13-merge-HC-condition-ledger-v22.json'],
                closed_question_scope='Pinned runner add default, inherited scale location, Tensor return, named post-hooks and decoder HC caller/ffn_pre boundary.',
                source_read_scope='3 exact source files/named functions plus installed Torch static declarations; no repository or whole-SDK audit.',
                remaining_profile_first_question='COMPUTE.attrInfo own-op/source-context association in the installed exporter; raw lookup is complete but meaning is not.',
                numerical_and_native_limit='Finite arithmetic/DAG evidence is analysis only; no native correctness/performance or complete consumer closure.')
    write(O / 'PROFILE-CODE-QUESTION-GATE-v22.json', gate)
    report = '''本轮有实质进展，整体仍未完成。继续按“充分使用profile，不能单独定案时再定点查代码”执行；没有运行模型/NPU、修改目标源码或变更服务。

完整保留了4,472次ForeachAddList：1,271次直接、3,200次图内、1次精确队列桥接，共9个shape/context组合。3,200个图内合流到FFN HcPost都由已有层块结构精确定位，同一TASK stream，间隔最小/中位数20ns、最大41ns；间隔内无其他计算算子，也无原生COMM_OP区间重叠。这是保存时间线中的可见边界，不能由此证明tensor指针、所有消费者或可节省延迟。其中2,560对decode-compatible，640对仍未绑定阶段。

去除全局定义ID后，这些图内边界只有1种shape/dtype签名：合流两个BF16[3,5120]输入，产出BF16[3,5120]，交给HC的视图为[1,3,5120]；HC另收BF16[1,3,4,5120] residual及FP32 post/comb。最初680种完整定义签名不是680种shape。控制区间重叠22,854次是“区间与层间隙”的配对次数，实际涉及6,787个不同TASK键；宽MODEL_WAIT_COMPLETE不能重复计作3,200次独立等待。

profile不能单独说明缩放/alpha和后处理语义，所以本轮只取了三个固定源码文件的相关函数，以及已安装Torch的静态声明。当前runner调用torch._foreach_add([shared_output],[fused_output])[0]，未传alpha，声明默认alpha=1。若runner有效gamma不为1，非FP16路径先单独原位缩放routed，再做输出变换及普通add；FP16 shared分支采用另一缩放方式。没有把配置里的模型gamma或位置属性文本当成当前runner有效gamma。当前runner返回Tensor，与model里的legacy tuple/muls_add分支分开记录。

合流前后的shared/routed归约、输出变换、feature维裁剪和可选zero-expert贡献仍有条件。特别是Ascend final-reduce使用层的静态SP/early策略及context-aware custom op，不能将capture时output_is_reduced布尔值冻结成每次replay的真值。图中无可见中间计算只能约束这些已捕获执行，不能证明所有分支的后处理恒为identity。

另一个profile直接给出的边界是行数：320次target原生合流输出M=800，而原始target MC2只有M816和M834两个原生行数域。不能将全部raw MC2输出与最终merge张量视为同M。此次读到的_maybe_pad_hidden_states处理feature维，不解释token行数差异；尚未按序号或最近时间把某次M800强配到某次M816/M834。

数值边界也分开登记：删除中间存储，与删除BF16舍入是两件事。两个stdlib CPU有限算术例子在明确的FP32/BF16最近偶数舍入模型下，分别得到1.5与1.5078125、0与0.00390625，说明任意保留更高精度或合并scale/add并非无条件等价。这不代表冻结模型必然达到这些输入，也不是已加载HC二进制的数值验证；实际native精度/顺序仍须绑定。若仅删除外存materialization，仍可要求保留原有dtype舍入。

候选图覆盖两算子Add->HC、三算子SharedDown->Add->HC，以及允许改变生产顺序的多算子重组条件域。另有一个明确受限排除：保持现有MC2先完成、SharedDown后执行的边，却让MC2尾部等待SharedDown，会形成MC2->SharedDown->MC2环。这只排除“仅改上游epilogue且保持原生产顺序”的方案，不排除新内核或新排程。当前基线没有被判定为死锁，也没有设计或实现融合。

HC仍须保留x、全部residual通道、post/comb，以及原有积累/舍入合同；decoder单独返回ffn_pre供后层使用，不能融合后丢弃。已有Engram、auxiliary及最终状态消费者约束继续有效。普通源码调用流不等于所有alias、固定地址、ready事件和跨迭代buffer代际已闭合。

本轮还补取了原DB尚未解码的attrInfo字符串：覆盖286,950次计算执行和83类算子，121,920次命中非空候选字符串，165,030次为空或placeholder，共181个op/string组合。但3,200个FFN HcPost全部复用了前一ForeachAddList的同一属性ID和文本。这个事实要求先核对已安装exporter的字段关联，不能把input_2:1当HC参数，也不能据此断言exporter有bug。下一步先查保存schema/exporter索引，只有它无法解释时才点查精确赋值路径。

合流输出逻辑payload合计8,135,065,600字节；一写一读接口模型为16,270,131,200字节，图内部分为196,608,000字节。它们不是实际HBM/网络流量、普适必要I/O或请求收益，不能用来承诺提速。融合后必要工作、存储层和容量界需重新推导。

主台账仍为33项模型候选、7项框架候选；新增16项有限边界条件和6个F23子域，不新增重复主候选。26个整模块均未宣称关闭，Goal保持active。全局资源归属、物理拓扑、MC2内部原因、当前EngineCore成本及其他消费者仍有明确缺口；最后健康观测仍是旧v12post，本轮无新的健康或空闲声明。
'''
    (O / 'PROFILE-FINDINGS-v22.md').write_text(report)
    action = ('继续Loop004，v22实质进展；先读merge-HC-summary-v22.json、merge-to-HC-contract-v22.json、M13-merge-HC-condition-ledger-v22.json、raw-profile-attribute-intake-v22.json。'
              '全部4472merge/3200图内merge->FFN HC边界已复核，1typedshape；20/20/41ns gap无中间compute/COMM，2560decode-compatible/640phase-unbound。'
              '当前Tensor runner plain foreach alpha默认1；非unit有效gamma若存在在add前单独scale，gamma实际值未新增绑定，不能套legacytuple公式。3sourcefiles+Torchstub精确取证，posthooks/runtimecomm/featuretrim/zeroexpert不能全域删。'
              '320targetmergeM800与rawtargetMC2M816640/M834320不等域，不能按序号/近邻配对；feature padding不是token padding。'
              '16有限分支/6F23子域含Add-HC、Down-Add-HC、多算子新生产顺序、原顺序upstream-only cycle排除、数值重写、legacy域；只是分析，未实现/未证明native合法或收益。'
              '2显式FP32/BF16RNE例子不是历史输入/native精度证明；保留BF16舍入与删除materialization分开。ffn_pre和全部HC/Engram/aux消费者保留。'
              '下一具体问题先profile/schema：v22已取全32DB286950/83类rawattrInfo字符串(121920非空/165030空)，但3200FFN HcPost全部复用前foreach的ID/text。先看既有SDK/schema索引，再只点查installed COMPUTE.attrInfo赋值路径及一跳关联，绝不整库/整SDK源码扫；不把HC input_2:1或attr positional值当本算子真实值，不断言bug。'
              '五server结果均PMU目录：shared-routed-merge-to-HC-boundary-v22 SHA67ea56947962067c41cd3942a6cc37d791dea4db9beec5c609d63eff226fd7db；merge-HC-output-source-contract-v22 SHAb1e32ce7e9e7534b9147c7ced944c9d23ad04f224fb269b358d747c2086f7db8；merge-inherited-scale-reduction-and-schema-v22 SHA34ef1f4c2104cc05423d44f6188b4c2d7f84aa9b69cecf54f92c26f673ec3342；saved-COMPUTE-attribute-field-census-v22 SHA46e59d2de1773f4e8f9fd0e717b5ae68a99094644986d1613a8cc8b9caf6f147；merge-row-domain-attribute-identity-and-payload-review-v22 SHA765ff128cff0fa1f0cdf6e068d60ed6f80d9c50711c044b182dcb231850150e5。'
              '逻辑output8,135,065,600B/onewrite+read16,270,131,200B不是实际HBM/必要I/O/收益；控制22854pair-occurrences仅6787不同TASK，不能重复计等待。'
              'v21 MC2=1强制sharedoverlapFalse解释仍成立，不重查已闭合问题；F21postloadweights/scales/generation/EP16物理拓扑/内部时序、F33/H07/compressor/ring/rollback消费者、当前EngineCore/producerready/prefixhash成本仍缺。'
              '若需补采按既有health/ownership/frozenworkload/clone/restore授权先准备有限诊断，不盲重启或假定空闲。3136candidate resource/849missingPMU/16unboundgraphs/26theoryINCOMPLETE保留。33model/7framework/0wholemodulesclosed，Goalactive；本轮纯CPU保存证据/定点source，无modelimport/NPU/服务变更，lasthealthv12post。')
    scripts = ['query_merge_HC_boundary_v22_166.py', 'read_merge_HC_output_contract_v22_166.py',
               'read_merge_inherited_hooks_v22_166.py', 'query_saved_compute_attributes_v22_166.py',
               'review_merge_domains_and_attributes_v22_166.py']
    descriptions = [
        'Complete4472merge census and exact3200structural graph merge-HC boundaries reviewed without nearest-time native pairing.',
        'Pinned runner posthooks and decoder HC caller functions bound in2 exact files; Tensor/tuple and ffn_pre separated.',
        'Inherited scale/feature-padding/zero hooks and Ascend predicates plus installed Torch static alpha default bound.',
        'All32savedDB attrInfo exact-ID string lookup covers286950/83types with DB SHA/stat unchanged; ownership semantics kept open.',
        'Typed graph signature, raw attribute-ID reuse, distinct control intervals, M800 mismatch and logical payload reviewed.']
    jobs = [('profile' if i in [0, 3] else 'review', descriptions[i], [N / (labels[i] + '.json'), N / scripts[i]]) for i in range(5)]
    jobs += [('review', 'Two explicit finite FP32/BF16-RNE arithmetic examples and conditional producer-order DAG cycle reviewed; not native execution.', [N / 'review_merge_rounding_and_dependency_v22.py', O / 'merge-rounding-and-dependency-review-v22.json', N / 'zcode-merge-rounding-v22-output-20261031.json']),
             ('review', 'Main review reconciles16finite conditions/6F23domains, profile/source evidence and unresolved native/consumer/attribute ownership gaps.', [O / 'merge-to-HC-contract-v22.json', O / 'M13-merge-HC-condition-ledger-v22.json', O / 'F23-merge-HC-variants-v22.json', O / 'PROFILE-FINDINGS-v22.md'])]
    runs = []
    for i, (kind, desc, arts) in enumerate(jobs):
        rid = 'run-' + (now - timedelta(seconds=len(jobs) - i)).strftime('%Y%m%dT%H%M%SZ')
        run = ctl('new-run', '--loop-dir', str(L), '--run-id', rid, '--case-key', 'c8-input32k-output1k-prefix90-166', '--kind', kind,
                  '--command', 'Preparedv22 evidence; retrospective registration: ' + desc,
                  '--code-ref', 'Saved166profile/exact3sourcefiles/Torchstub; no model import/NPU/service work', '--working-directory', str(N))
        args = ['finish-run', '--run-dir', run, '--status', 'pass', '--correctness', 'not-applicable', '--summary', desc]
        for p in arts:
            assert p.exists(), p
            args += ['--artifact', str(p)]
        ctl(*args)
        runs.append(dict(run_dir=run, status='pass', description=desc))
    cp = dict(status='PROGRESS', previous_turn_classification=summary['previous_turn_classification'],
              current_verified_changes=['All4472merges/3200graphHC boundaries and row/attribute/control-interval domains reviewed.',
                                        'Current Tensor plain-add alpha1, inherited pre-add scale, posthooks and ffn_pre contract bound.',
                                        '16finite condition branches/6F23domains; explicit-model rounding and restricted-order cycle exclusions; no native fusion/gain claim.',
                                        'All286950/83types raw attribute lookup recovered;3200FFN HC preceding-op ID reuse makes exporter ownership the next exact profile/source question.'],
              summary=str(O / 'merge-HC-summary-v22.json'), candidate_ledger=str(O / 'candidate-profile-ledger-v22.json'),
              framework_candidate_ledger=str(O / 'framework-candidate-ledger-v22.json'), module_ledger=str(O / 'module-profile-ledger-v22.json'),
              branch_matrix=str(O / 'module-phase-branch-matrix-v22.json'), report=str(O / 'PROFILE-FINDINGS-v22.md'),
              source_gate=str(O / 'PROFILE-CODE-QUESTION-GATE-v22.json'), coverage_gaps=str(O / 'module-coverage-gaps-v22.json'),
              pending_full_scope_domains=remaining, remote_job_status='5prepared166CPU-only jobs and1local stdlib arithmetic job terminal pass; no new model/NPU/service/health work.',
              next_action=action, whole_modules_closed=0, overall_complete=False, updated_utc=stamp)
    write(O / 'REVIEW-CHECKPOINT-v22.json', cp)
    for p in [L / 'state.json', T / 'index.json']:
        d = read(p)
        assert d['status'] in ['EVALUATING', 'ACTIVE']
        d.update(next_action=action, updated_at=stamp)
        write(p, d)
    for p in [L / 'status.md', T / 'status.md']:
        p.write_text('\n'.join(('- 下一步：' + action) if x.startswith('- 下一步：') else ('- 更新时间：`' + stamp + '`') if x.startswith('- 更新时间：') else x for x in p.read_text().splitlines()) + '\n')
    (O / 'profile-summary-before-v22.json').write_bytes((R / 'profile-summary.json').read_bytes())
    prof = read(R / 'profile-summary.json')
    prof['supplementary_merge_HC_v22'] = dict(summary=cp['summary'], graph_pairs=3200, all_merge_executions=4472, default_source_alpha=1,
                                              raw_attributes_ownership='UNBOUND_DESPITE_COMPLETE_RAW_LOOKUP', whole_modules_closed=0, overall_complete=False)
    write(R / 'profile-summary.json', prof)
    for p in [N / 'RECOVERY.json', W / 'STATE.json', R / 'STATE.json', R / 'PARALLEL-RESUME-PACK.json']:
        d = read(p)
        d.update(latest_profile_first_review=cp, overall_complete=False, whole_modules_closed=0, updated_utc=stamp)
        d['next_read_only_action' if p == R / 'STATE.json' else 'next_action'] = action
        if p == W / 'STATE.json':
            d.update(profile_candidate_ledger=cp['candidate_ledger'], profile_module_ledger=cp['module_ledger'])
        if p in [N / 'RECOVERY.json', R / 'PARALLEL-RESUME-PACK.json']:
            d['read_first'] = [cp['summary'], str(O / 'merge-to-HC-contract-v22.json'), str(O / 'raw-profile-attribute-intake-v22.json'), cp['coverage_gaps']]
        write(p, d)
    compact = ctl('compact', '--task-dir', str(T))
    validation = ctl('validate', '--task-dir', str(T))
    write(O / 'task-run-registration-v22.json', dict(runs=runs, compact_output=compact, validation_output=validation, whole_modules_closed=0, overall_complete=False))
    arts = [p for p in O.glob('*v22*') if p.is_file() and not p.name.startswith('checkpoint-manifest')]
    arts += [N / (label + '.json') for label in labels] + [N / s for s in scripts]
    arts += [N / 'review_merge_rounding_and_dependency_v22.py', N / 'zcode-merge-rounding-v22-output-20261031.json',
             N / 'RECOVERY.json', W / 'STATE.json', R / 'STATE.json', R / 'PARALLEL-RESUME-PACK.json',
             R / 'profile-summary.json', L / 'state.json', T / 'index.json', N / 'finalize_merge_HC_contract_v22.py']
    write(O / 'checkpoint-manifest-v22.json', dict(generated_utc=stamp, artifacts=[dict(path=str(p), bytes=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in arts], whole_modules_closed=0, overall_complete=False))
    print(json.dumps(dict(checkpoint=str(O / 'REVIEW-CHECKPOINT-v22.json'), runs=len(runs), validation=validation, overall_complete=False, whole_modules_closed=0), ensure_ascii=False))


if __name__ == '__main__':
    main()
