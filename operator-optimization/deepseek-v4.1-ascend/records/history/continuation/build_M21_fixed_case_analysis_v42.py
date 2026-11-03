"""Profile-led fixed-case analysis. No target implementation or device execution."""
from copy import deepcopy
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
import hashlib
import json

from finalize_origin_coverage_v9 import N, O, W, read, write
from finalize_Engram_whole_frontier_v30 import packet, graph


def main():
    assert not (O / 'REVIEW-CHECKPOINT-v42.json').exists()
    stamp = datetime.now(timezone.utc).isoformat()
    config = packet('M21-config166-v42', True)
    failure = read(N / 'M21-predicates166-v42.json')
    assert failure['returncode'] == 255
    assert 'Permission denied' in failure['stderr']
    assert len(config['result']['records']) == 4
    # Verify every already point-read v42 source against its frozen bytes.
    pins = {}
    for artifact in sorted(O.glob('*source-v42.json')):
        x = read(artifact)
        for f in x.get('files', [x]):
            if 'path' in f and 'sha256' in f:
                p = Path(f['path'])
                assert hashlib.sha256(p.read_bytes()).hexdigest() == f['sha256'], p
                pins[str(p)] = f['sha256']
    ascend_sha = 'd5ea2cd9a5499e5c0d7a41114005569d3c9092027e0cf03510c40dc4951ac4cd'
    assert ascend_sha in json.dumps(read(O / 'shared-overlap-activation-contract-v21.json'))

    records = {Path(r['path']).name: r for r in config['result']['records']}
    engine = {r['field']: r['literal'] for r in records['service.log']['selected_fields']
              if r.get('record_type') == 'EFFECTIVE_ENGINE_LOG'}
    assert engine['tensor_parallel_size'] == '8'
    assert engine['data_parallel_size'] == '2'
    assert engine['decode_context_parallel_size'] == '1'
    assert engine['cudagraph_capture_sizes'] == '[8, 16, 24]'
    assert engine['enforce_eager'] == 'False'
    assert not any(r.get('field') == 'prefill_context_parallel_size'
                   for r in records['service.log']['selected_fields'])
    facts = [
        dict(grade='MEASURED_SAVED_CONFIGURATION', statement='Pinned saved scripts select DP2/TP8/EP, max_num_seqs4/max_num_batched_tokens8192, dspark/5 draft tokens with nested enforce_eager=True, additional enable_dsa_cp=True, enable_flashcomm1=True, enable_fused_mc2=1. These are capture configuration facts, not current health or live object values.'),
        dict(grade='MEASURED_SAVED_CONFIGURATION', statement='Effective main Engine startup log L103 records TP8/PP1/DP2/DCP1, enforce_eager=False, chunked prefill, warmups1, capture sizes[8,16,24], copy_inputs=False. Parser L25 capture sizes[1,2,4,8,16,24]/warmups0 are earlier input settings; do not replace the effective Engine values.'),
        dict(grade='DOCUMENTED', statement='DeepseekV41CacheBackend.get_impl_cls/get_builder_cls delegate to get_v41_cp_classes. That factory rejects simultaneous legacy DSACP and PCP, otherwise selects CP, PCP, or base class. Current-model construction context matters; its target/draft comment is not current captured configuration evidence.'),
        dict(grade='DOCUMENTED', statement='AscendConfig normalizes enable_dsa_cp by has_indexer AND parallel_config.use_sequence_parallel_moe. CLI true alone is not proof of the normalized boolean. The exact enable_dsa_cp/enable_pcp helper bodies remain unread after authentication failure.'),
        dict(grade='DOCUMENTED_PLUS_MEASURED', statement='v35 exact positive draft fullcaller is _update_caches342 -> _preprocess354 -> CPImpl._prepare_inputs_and_caches192 -> base forward701 -> _run_merged_draft1494. It contains no PCPImpl.forward490 wrapper. Together with the factory source, this supports the observed direct legacy CP draft path. It does not measure an integer PCP degree for main or draft or prove every phase dispatch.'),
        dict(grade='DOCUMENTED_PLUS_MEASURED', statement='DSpark constructor sets use_cuda_graph=False after super; base ACLGraphWrapper construction requires use_cuda_graph. Nested draft enforce_eager=True also disables it. The current draft outer ACLGraph wrapper is excluded by this source/configuration conjunction; nested compilation bodies and main graph behavior are separate questions.'),
        dict(grade='DOCUMENTED', statement='speculative_config.method=dspark chooses the algorithm; DynamicSpecConfig.method defaults to None and is a separate adaptive-budget selector. The saved whitelisted additional config has no selected dynamic_spec_config override. Source loader may return a cached AscendConfig; no live object snapshot proves absence of other mutation.'),
        dict(grade='DOCUMENTED', statement='sample_from_anchor defaults True but may be overridden in draft HF config. Q per request is D if true, D+1 otherwise; max_query_tokens=max_batch_size*Q. For D5 and max_batch4 the constructor alternatives are20/24. Positive repeats10/20 do not independently prove the HF flag or all bank capacities.'),
        dict(grade='DOCUMENTED', statement='Draft groups intersect draft layer names with actual cache groups, unwrap per-layer UniformType specs, group by backend full class plus actual spec, and preserve per-gid kernel block size and independent query/context buffers. Physical page size cannot replace the logical kernel block size.'),
        dict(grade='DOCUMENTED', statement='Metadata constructor dispatch distinguishes compressor-state/SWA/index-K-folded/index-K/long-KV, rejects unsupported specs, and allocates dtype/capacity-specific fixed banks. Packed getter/device class is not fully bound here; native legacy SMLA/QLI is positive evidence for those observed calls, not universal exclusion of every packed constructor branch.'),
    ]
    write(O / 'M21-captured-config-and-selected-dispatch-contract-v42.json', dict(
        generated_utc=stamp, source_files=pins, saved_configuration=config['result'],
        evidence_ref=dict(receipt=str(N/'M21-config166-v42.json'), full_result=config['full_result'], sha256=config['result_sha256']),
        facts=facts, live_PCP_degree=None, live_dynamic_spec_method=None,
        source_default_is_not_live_object=True,
        missing_definitions=['attention/utils.py:enable_pcp', 'attention/context_parallel/common_cp.py:is_pcp_decode_sharding_enabled',
                             'exact enable_dsa_cp helper and DeviceOperator packed/device selection predicate'],
        authentication_failure=str(O/'M21-predicate-remote-authentication-failure-v42.json'),
        whole_module_closed=False, overall_complete=False))

    # Record only the small producer span needed for the eager zero-write question.
    dsa = W / 'frozen-source-snapshot/vllm-ascend/vllm_ascend/attention/dsa_v41.py'
    assert hashlib.sha256(dsa.read_bytes()).hexdigest() == '3505769857f83fa7e67f652984f50bd0478158f1d24f185cab091203e70392fd'
    lines = dsa.read_text().splitlines()
    write(O/'M21-eager-empty-SMLA-producer-point-source-v42.json', dict(
        path=str(dsa), sha256=hashlib.sha256(dsa.read_bytes()).hexdigest(), start=1161, end=1207,
        source=[f'{i+1}: {lines[i]}' for i in range(1160, 1207)],
        question='Does the empty-query producer only zero the plan bank, and does a later positive callback replace the full destination?',
        no_model_or_torch_import=True))

    pcp = dict(
        generated_utc=stamp, evidence=['M21-selected-constructor-group-and-PCP-alternative-source-v42.json',
            'M21-independent-local-global-plan-owner-source-v42.json', 'M21-PCP-global-hidden-consumer-point-source-v42.json'],
        current_observed_draft_path='DIRECT_LEGACY_CP; PCP considered as an alternative rather than measured PCP execution',
        main_PCP_degree='UNKNOWN', decode_shard_selector='is_pcp_decode_sharding_enabled(vllm_config), exact definition UNREAD',
        cases=[
            dict(case='PCP_CONSTRUCTOR', condition='DCP != 1', result='NotImplementedError; replicated cache PCP requires DCP1.'),
            dict(case='GLOBAL_METADATA', condition='global_batch.is_prefilling_np.any() OR symbolic_decode_sharded', result='All ranks use the same global decision; retain global cache metadata and hidden gather/restore even when local query is empty.'),
            dict(case='REPLICATED_DECODE', condition='not global_prefill AND not symbolic_decode_sharded', result='Return local metadata; no PCP global hidden restore in this source branch.'),
            dict(case='EMPTY_LOCAL_QUERY', condition='needs_global_metadata AND local_common.num_actual_tokens == 0', result='Empty local metadata skips SMLA/QLI plan ops, but retains global cache view and all necessary collectives.'),
            dict(case='FULL_GRAPH', condition='full_graph_mode', result='Preserve common graph shapes; request banks resized before capture, capacity local=max(request_factor*max_reqs,max_capture)+1 and global=max(max_reqs,max_capture)+1.'),
            dict(case='DUMMY', condition='explicit skip_ring_state_update OR global_batch.is_dummy', result='Skip C2 ring advance; not permission to skip other metadata/cache/communication effects.'),
            dict(case='C2_OWNER', condition='needs_global_metadata', result='Global cache owner builds compressed-state metadata; local build_compressor_metadata=False. Otherwise local owner builds it.'),
            dict(case='ROPE_OWNER', condition='SWA source tables prepared', result='Existing per-owner table-pointer dedup plus independent persistent local-Q/global-KV cos/sin banks; keep these owners and addresses.'),
            dict(case='HIDDEN', condition='metadata is PCP type', result='all_gather(contiguous local padded hidden,dim0) then index_select(hidden_restore_idx); verify exact restored row space before any pool/history update.'),
            dict(case='O_PROJ_WEIGHTS', condition='weight_sharding enabled AND needs_global_metadata', result='Use full weights, prepare aliases during warmup, wait pending handles and switch back in finally. Source does not prove all-gather happens on every iteration.'),
        ],
        global_hidden_consumers=['SWA KV projection and cache write', 'compressed-source projection when role.is_kv_source',
                                 'multistream preprocessing when enabled; its downstream consumers must also be transformed or retain original input'],
        existing_optimizations_not_new_gains=['Replicated decode no gather', 'Empty-local no SMLA/QLI build',
            'source-table pointer dedup', 'independent graph-stable RoPE output banks', 'global/local C2 ownership split'],
        implementation=False, native_validation=False, critical_path_gain_us=None, whole_module_closed=False, overall_complete=False)
    write(O/'M21-PCP-alternative-case-and-consumer-contract-v42.json', pcp)

    # Finite symbolic/integer example: row-local projection commutes with concatenate/select.
    # Repeated rows and physical padding are included. No native GEMM parity is inferred.
    def project(rows, weight, bias):
        return [[sum(x[k]*weight[k][j] for k in range(len(x)))+bias[j]
                 for j in range(len(bias))] for x in rows]
    parts = [[[1, 2], [0, 0]], [[3, -1], [7, 5]]]
    joined = [row for part in parts for row in part]
    weights = [([[1, 2], [-2, 3]], [4, -5]), ([[3], [7]], [-2])]
    cases = 0
    for length in range(5):
        for selector in product(range(len(joined)), repeat=length):
            for weight, bias in weights:
                restored = [joined[i] for i in selector]
                before = project(restored, weight, bias)
                projected_parts = [row for part in parts for row in project(part, weight, bias)]
                after = [projected_parts[i] for i in selector]
                assert before == after
                cases += 1
    # Counterexample: adjacent-row pooling cannot move before arbitrary global restoration.
    physical = [1, 100, 2, 200]
    restore = [0, 2, 1, 3]
    after_restore_pool = [physical[restore[0]]+physical[restore[1]], physical[restore[2]]+physical[restore[3]]]
    pool_before_restore = [physical[0]+physical[1], physical[2]+physical[3]]
    assert after_restore_pool == [3, 300] and pool_before_restore == [101, 202]
    write(O/'F32-PCP-row-projection-and-restore-proof-v42.json', dict(
        generated_utc=stamp, grade='DERIVED', identity='F_j(S_pi concat_r H_r) = S_pi concat_r F_j(H_r)',
        domain='Same row-local affine function F_j(x)=xW_j+b_j on every rank, same physical carrier rows and restoration indices. Selection may repeat rows; it need not be a permutation.',
        finite_integer_examples=cases, examples_passed=True,
        counterexample=dict(physical=physical, restore=restore, pool_after_restoration=after_restore_pool, pool_before_restoration=pool_before_restore),
        implication='Project each required row-local global-hidden consumer before PCP gather and restore the projected planes. Keep C2 pooling/history/ring writes after correct row restoration unless separately proved.',
        complete_consumer_requirement=pcp['global_hidden_consumers'],
        logical_payload_comparison='Per equal carrier row: original D*b_H versus sum_j P_j*b_j for every required output plane. Include any remaining hidden consumer and additional quant/scale planes. This is not a measured wire-byte or latency model.',
        native_precision_limit='Different per-rank GEMM shape/quantization/reduction tiling may change numerical results; integer row identity does not establish BF16/native bitwise equivalence.',
        not_proved=['actual owner-specific weights/layouts and all consumers', 'physical padding/mask/restore validity and capture banks',
                    'native roundings, quant scales, actual transfer bytes, topology, overlap, and causal gain'],
        implementation=False, native_validation=False, critical_path_gain_us=None, overall_complete=False))

    m19 = read(O/'M19-current-forward-patterns-and-dispatch-v35.json')
    full = m19['exact_fullcallers']['selected_fullcallers']
    empty = [r for r in full if any('_update_caches' in f for f in r.get('project_frames', []))]
    assert empty and any('dsa_v41_cp.py(192)' in f for r in empty for f in r['project_frames'])
    zero = read(O/'M21-current-native-and-global-host-profile-v37.json')['metadata_zero_callbacks']
    assert zero['DRAFT_PROPOSAL_ANCESTRY'] == 19
    write(O/'F31-eager-empty-query-plan-write-review-v42.json', dict(
        generated_utc=stamp, grade='DOCUMENTED_PLUS_MEASURED_WITH_CONDITIONAL_DERIVATION',
        source=['M21-eager-empty-SMLA-producer-point-source-v42.json', 'M21-backend-factory-and-empty-query-reader-source-v42.json',
                'M21-PCP-global-hidden-consumer-point-source-v42.json', 'M21-draft-runnable-guard-source-v42.json',
                'M21-selected-constructor-group-and-PCP-alternative-source-v42.json'],
        producer='Legacy query-plan producer callback zeroes _smla_metadata when num_actual_tokens==0; a positive callback copy_(value) replaces the full destination bank before attention reads it.',
        consumer='Inherited forward prepares global cache work first, then branches on metadata.swa.num_actual_tokens. Zero query bypasses index/SMLA reader but still calls CP output projection/communication. This zero query branch is distinct from attn_metadata is None, whose output zero branch returns earlier.',
        draft_outer_wrapper='EAGER_BY_SAME_SOURCE_PLUS_SAVED_DRAFT_CONFIGURATION; not a universal no nested graph theorem',
        positive_zero_callback_counts=zero, positive_no_query_pattern_counts=19,
        no_query_counts_not_joined_to_callbacks=True,
        positive_update_fullcaller=empty[0],
        conditional_transformation='For an exclusively owned eager no-query bank with all readers bypassed, omit only zero_ value writes inside the callback. Retain the task, stage/frontier event, input-ready, reuse/release fence, and all cache/output collective participation.',
        next_use_rule='The next positive callback must fully overwrite this bank before any read. No captured/full-bank/debug/version observer may depend on the skipped write.',
        exclusions=['Main logical-zero graph replay can retain physical query work; actual graph/parameter-bank body remains unbound.',
                    'QLI native plan has different guards and format; do not borrow this SMLA proof.',
                    'Do not remove all metadata tasks: executor requires stable nonempty frontiers.',
                    'No deletion of cache writes, output AllToAll/ReduceScatter, or dummy branch effects by zero query count.'],
        local_value_deadness_condition_analyzed=True, actual_all_observer_and_epoch_attestation=False,
        implementation=False, native_validation=False, critical_path_gain_us=None, whole_module_closed=False, overall_complete=False))

    variants = [
        dict(id='F32-v42-PCP-PROJECT-THEN-RESTORE', parent='F32', transformation='Move every required row-local global-hidden projection before gather and restore projected planes; retain original hidden gather if any consumer still needs it.', proof='F32-PCP-row-projection-and-restore-proof-v42.json'),
        dict(id='F16-v42-PCP-WEIGHT-ACTIVATION-SCHEDULE', parent='F16', transformation='Compare compatible weight-gather/full-weight epochs with activation exchange/output-projection scheduling while preserving all alias/handle/switch-back states.', proof='M21-PCP-alternative-case-and-consumer-contract-v42.json'),
        dict(id='F31-v42-PCP-COORDINATE-ROPE-PRODUCER', parent='F31', transformation='Compose pure row-coordinate/table-index production into the independent local-Q/global-KV fixed RoPE destinations; existing table dedup and empty-local bypass are baseline.', proof='M21-PCP-alternative-case-and-consumer-contract-v42.json'),
        dict(id='F31-v42-EAGER-NOQ-SMLA-ZERO', parent='F31', transformation='Omit a value-dead zero_ in an exclusive eager no-query SMLA callback while preserving its complete publication protocol.', proof='F31-eager-empty-query-plan-write-review-v42.json'),
    ]
    conditions = [
        ('CASE', 'Bind main/draft construction contexts separately. Do not infer PCP degree, dynamic budget or anchor flag from absence/counts/defaults.'),
        ('FACTORY', 'Legacy DSACP and PCP are mutually exclusive at factory entry. Alternative PCP requires exact parallel/group selector mapping.'),
        ('PHASE', 'Main/draft eagerness, startup parser settings, effective main graph sizes, actual capture and replay are separate domains.'),
        ('GROUPS', 'Keep cache-kind/spec/backend grouping, all layer owners, logical kernel block sizes, dtypes and bank capacities.'),
        ('PCP_GLOBAL', 'Use globally consistent prefill/decode-sharding decision and retain empty-query rank collective participation.'),
        ('RESTORE', 'Preserve contiguous physical carrier padding, hidden_restore_idx ordering, selection multiplicity and every output row mapping.'),
        ('ALL_HIDDEN_READERS', 'SWA KV plus compressed-source and multistream consumers must be transformed or original global hidden remains live.'),
        ('ROW_LOCAL', 'Only functions whose inputs are local to the selected row commute freely; keep pool/history/C2 ring updates in restored request order.'),
        ('WEIGHTS', 'Per-rank projection weights/bias/quant scales must represent the same function or include explicit weight shard ownership transformation.'),
        ('NATIVE_MATH', 'Row identities do not prove BF16/FP32/quantized native rounding equivalence after GEMM shape/tiling changes.'),
        ('O_PROJ_EPOCH', 'Weight all-gather handles, warmup aliases, full/shard switch-back and actual generation lifetimes remain; no assumed per-iteration transfer.'),
        ('ROPE_BANKS', 'Keep independent local/global cos/sin output addresses and table-owner aliases; existing pointer dedup is not a new gain.'),
        ('C2_OWNER', 'Only the global or local owner advances compressed-state ring according to needs_global_metadata and dummy/skip rules.'),
        ('EMPTY_GUARD', 'SMLA value-zero elision is limited to verified eager num_actual_tokens0 with all plan readers bypassed.'),
        ('NEXT_WRITER', 'Any later query reader must see complete plan overwrite; preserve inactive/full-bank reads, alias observers and version semantics.'),
        ('FRONTIER', 'Keep nonempty stable tasks, input-ready, per-stage events, captured-producer semantics, buffer_reusable and release.'),
        ('OUTPUT', 'No-query ranks can receive other query heads and must retain output projection/AllToAll/ReduceScatter and global cache writes.'),
        ('OPAQUE_PLANS', 'SMLA, QLI and packed-plan banks are distinct opaque formats; no shape-only or address-only plan CSE.'),
        ('CAPTURE', 'Current draft outer wrapper exclusion does not establish main graph/QLI body or all captured parameter/readers.'),
        ('OVERLAP_COST', 'Payload formulas/callback counts and conditional alternatives are not measured traffic/critical path/gain; do not add overlapping F31/F32/F16 variants.'),
    ]
    for v in variants:
        v.update(implemented=False, native_validation=False, critical_path_gain_us=None)
    write(O/'M21-fixed-case-conditional-variants-v42.json', dict(
        generated_utc=stamp, variants=variants,
        conditions=[dict(id='M21-v42-'+k, requirement=v, actual_current_legality_proved=False) for k, v in conditions],
        new_model_candidates=[], implementation=False, whole_module_closed=False, overall_complete=False))

    graphs = {
        'conditional_projection': graph(['rank_local_padded_hidden', 'all_required_local_projections', 'gather_projected_planes', 'restore_global_rows', 'SWA_and_compressed_consumers', 'C2_pool_history_ring'], [
            ('rank_local_padded_hidden', 'all_required_local_projections', 'SAME_ROW_FUNCTION_AND_WEIGHT_OWNERS'),
            ('all_required_local_projections', 'gather_projected_planes', 'ALL_RANKS_PARTICIPATE'),
            ('gather_projected_planes', 'restore_global_rows', 'EXACT_RESTORE_INDICES'),
            ('restore_global_rows', 'SWA_and_compressed_consumers', 'NO_DROPPED_GLOBAL_HIDDEN_CONSUMER'),
            ('SWA_and_compressed_consumers', 'C2_pool_history_ring', 'KEEP_REQUEST_ORDER_AND_STATE_OWNER')]),
        'conditional_no_query': graph(['input_ready_and_prior_reader_fence', 'empty_callback_keep_frontier', 'eager_forward_no_query_guard', 'global_cache_and_output_collectives', 'all_last_readers', 'reuse_fence', 'next_positive_full_overwrite', 'next_plan_reader'], [
            ('input_ready_and_prior_reader_fence', 'empty_callback_keep_frontier', 'TASK_EXISTS_NO_VALUE_WRITE'),
            ('empty_callback_keep_frontier', 'eager_forward_no_query_guard', 'EVENT_PROTOCOL_RETAINED'),
            ('eager_forward_no_query_guard', 'global_cache_and_output_collectives', 'QUERY_PLAN_NOT_READ'),
            ('global_cache_and_output_collectives', 'all_last_readers', 'PRESERVE_CACHE_OUTPUT_WORK'),
            ('all_last_readers', 'reuse_fence', 'NO_OVERWRITE_BEFORE_LAST_READER'),
            ('reuse_fence', 'next_positive_full_overwrite', 'NEXT_GENERATION'),
            ('next_positive_full_overwrite', 'next_plan_reader', 'FULL_BANK_VALUE_READY')]),
    }
    write(O/'M21-fixed-case-conditional-graphs-v42.json', dict(
        generated_utc=stamp, graphs=graphs, graphs_acyclic=True,
        scope='Conditional data/protocol graphs, not measured timing graphs or proven live alias/readiness graphs.',
        implementation=False, whole_module_closed=False, overall_complete=False))

    # Exact finite reconciliation only for the 27 guards whose methods were point-read.
    sem = deepcopy(read(O/'M21-semantic-branch-inventory-v41.json'))
    owner_notes = {
        'AscendDSAV41Impl._get_layer_metadata': 'Optional source prefixes select distinct SWA, long-KV, index-K and compressor-state banks. Query/index cache owner depends on role; preserve these independent owners.',
        'AscendDSAV41Impl._update_caches': 'Empty global hidden returns; otherwise SWA KV always updates and compressed source updates only for is_kv_source. Local query emptiness does not prove global hidden empty.',
        'AscendDSAV41Impl._preprocess': 'Slice local query/global cache by their metadata; no query calls cache update then returns. Nonempty path distinguishes multistream and cache-present/kv-source branches; all global hidden consumers retained.',
        'AscendDSAV41Impl._forward_attention': 'Packed and legacy readers differ. Optional long source/topK lengths and ratio1/2 choose source/metadata/blocks. Legacy SMLA plan wait/read only reached after nonzero query forward branch.',
        'AscendDSAV41Impl.forward': 'Optional output allocation, no-attn-metadata zero-output early return, nonzero local-query attention vs empty attention, and CP heads; output projection still called after query-count branch.',
    }
    ev = ['M21-PCP-global-hidden-consumer-point-source-v42.json', 'M21-backend-factory-and-empty-query-reader-source-v42.json',
          'M19-embedding-HC-CP-row-and-consumer-contract-v35.json', 'F31-eager-empty-query-plan-write-review-v42.json']
    matched, pending = [], []
    for row in sem['static_branch_sites']:
        if row['semantic_family'] != 'M21-S13':
            continue
        if row['owner'] in owner_notes:
            row.update(audit_state='GUARD_SOURCE_DATAFLOW_RECONCILED;ACTUAL_FLAGS_ALIAS_AND_GRAPH_BINDING_CONDITIONAL',
                       semantic_evidence=ev, review_status='POINT_SOURCE_GUARD_RECONCILED', runtime_binding='OBSERVED_PATH_PLUS_SOURCE_CONDITIONAL_DOMAINS')
            matched.append(dict(id=row['id'], owner=row['owner'], line=row['line'], condition=row['condition'],
                                semantics=owner_notes[row['owner']], evidence=ev, all_actual_variant_flags_bound=False))
        else:
            pending.append(dict(id=row['id'], owner=row['owner'], condition=row['condition']))
    assert len(matched) == 27 and len(pending) == 36
    write(O/'M21-adjacent-guard-crosswalk-v42.json', dict(
        generated_utc=stamp, original_S13_static_sites=63, guard_dataflow_reconciled=matched,
        remaining_specific_guard_crosswalk=pending, note='Method source read is not whole-module completion; 36 other static guards still need finite existing-neighbor-proof reconciliation.',
        whole_module_closed=False, overall_complete=False))
    updates = {
        'M21-S01': ('FIXED_MAIN_DRAFT_CONFIG_AND_GROUP_CONTRACTS_BOUND;EXACT_HELPER_PREDICATES_PENDING',
                    'Saved DP2/TP8/mainDCP1 and effective main graph sizes versus eager draft separated; constructor/spec/group capacities analyzed. Exact enable_pcp/enable_dsa_cp and packed/device predicate bodies remain unread; HF anchor flag/live cached config are not measured.'),
        'M21-S07': ('ANALYZED_CONDITIONAL_EAGER_NOQ_WRITE_ELISION', 'Exclusive eager no-query SMLA zero-write deadness conditions analyzed; retain all publication tasks/events/cache/output work. Actual observer/alias epochs and main graph readers remain conditional/S15.'),
        'M21-S11': ('DUMMY_INIT_BOUND;DRAFT_OUTER_EAGER_SEPARATED;MAIN_CAPTURE_EXTENTS_PENDING', 'Draft outer ACLGraph wrapping excluded by source+saved nested eager configuration. Main capture/full-bank readers and dummy output/no-attention branch remain separate; no blanket dummy cache-work claim.'),
        'M21-S12': ('EAGER_DRAFT_OUTER_MODE_BOUND;MAIN_CAPTURE_BANK_PROTOCOL_PENDING', 'Draft mode identified; no-query callback task/frontiers/reuse retained. Main captured producers, nested compilation and full parameter-bank/last-reader reconciliation remain S15/M22.'),
        'M21-S13': ('27_OF_63_GUARDS_DATAFLOW_RECONCILED;36_SPECIFIC_NEIGHBOR_GUARDS_PENDING', '27 exact guard sites across metadata selection/cache-update/preprocess/attention/forward now reconciled. 36 named guard sites remain; reuse M07-M11/M19 evidence rather than whole-file audit.'),
        'M21-S14': ('PCP_COUNTERFACTUAL_CASE_AND_CONSUMER_ANALYZED;EXACT_SELECTOR_MAP_PENDING', 'All small PCP interface methods analyzed including global/local ownership, empty query, restore, RoPE and weight states. Current observed draft fullcaller is direct legacy CP; main PCP degree and exact decode-shard/enable predicates not established. PCP remains a conditional cross-card alternative.'),
        'M21-S15': ('DRAFT_OUTER_WRAPPER_BOUND;MAIN_GRAPH_BANK_PHASE_BODY_PENDING', 'Source+saved config bind eager draft outer wrapper only. Actual 80 graph bodies/QLI parameters/all readers/16 unbound phases and nested compilation remain open.'),
    }
    evidence = ['M21-captured-config-and-selected-dispatch-contract-v42.json', 'M21-PCP-alternative-case-and-consumer-contract-v42.json',
                'F32-PCP-row-projection-and-restore-proof-v42.json', 'F31-eager-empty-query-plan-write-review-v42.json',
                'M21-adjacent-guard-crosswalk-v42.json', 'M21-fixed-case-conditional-variants-v42.json']
    for b in sem['branches']:
        if b['id'] in updates:
            b['state'], b['gap'] = updates[b['id']]
            b['evidence'] = list(dict.fromkeys(b['evidence']+evidence))
    sem.update(generated_utc=stamp, parent_inventory='M21-semantic-branch-inventory-v41.json', whole_module_closed=False, overall_complete=False)
    assert len(sem['branches']) == 18 and len(sem['static_branch_sites']) == 268
    write(O/'M21-semantic-branch-inventory-v42.json', sem)
    audit = deepcopy(read(O/'M21-original-requirement-completion-audit-v41.json'))
    audit.update(generated_utc=stamp, parent_audit='M21-original-requirement-completion-audit-v41.json', result='NOT_COMPLETE',
                 prioritized_next=['36 remaining S13 exact guard crosswalks from existing evidence',
                                   'M22 actual graph/QLI body/parameter-bank/readiness and 16 phase-unbound graph bindings',
                                   'S16 same-denominator resource bounds; S18 full-process/transport audit',
                                   'Exact dispatch/PCP helper definitions only after authentication restored'])
    for row in audit['requirements']:
        row['evidence'] = list(dict.fromkeys(row['evidence']+evidence+['M21-semantic-branch-inventory-v42.json']))
        row['completion_proved'] = False
    write(O/'M21-original-requirement-completion-audit-v42.json', audit)
    print(json.dumps(dict(fixed_source_files_verified=len(pins), conditional_variants=len(variants), conditions=len(conditions),
                          integer_row_examples=cases, reconciled_neighbor_guards=len(matched), remaining_neighbor_guards=len(pending),
                          semantic_families=18, original_static_sites=268, whole_module_closed=False, overall_complete=False)))


if __name__ == '__main__':
    main()
