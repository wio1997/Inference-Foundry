"""Review scoped operator conclusions from existing profiles; no target execution."""
import collections
import itertools
import json
from pathlib import Path

H = Path(__file__).resolve().parent
O = H.parent / 'profile-first-20261031'


def read(name):
    return json.loads((H / name).read_text())


def write(name, value):
    (H / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


native = read('native-join-observation.json')['review']
audit = read('observer-snapshot-audit-result.json')['OBSERVER-SNAPSHOT-ORDER-AUDIT.json']
catalog = read('operator-callsite-catalog.json')['catalog']
startup = read('startup-creator-census-result.json')['review']
assert native['joins'] == audit['checked'] == 145280
assert native['issues_count'] == audit['issues_count'] == 0
assert len(native['profiles']) == len(audit['profiles']) == 16
assert catalog['F21_native_kernels_per_exact_CPU_call'] == {'1': 736}
assert startup['assignment'] == {'unique': 54336}
assert all(p['native_replays'] == 227 and p['native_F21_per_replay'] == {'40': 227}
           and len(p['graph_kernel_streams']) == 1 for p in native['profiles'])
write('graph-order-and-snapshot-reviewed-contract.json', {
    'status': 'REVIEWED_CONDITIONAL_ORDERED_ASSOCIATION_IN_CURRENT_GRAPH_DOMAIN',
    'facts': {
        'official_replay_native_envelopes': 3632,
        'F21_calls_per_replay': 40,
        'associated_target_small_row_executions': 145280,
        'shape_order_or_cardinality_failures': 0,
        'native_counter_snapshot_order_checks': 145280,
        'snapshot_order_issues': 0,
        'startup_same_process_one_native_F21_per_CPU_operator_calls': 736,
        'eager_executions_left_unassociated': 13648,
    },
    'exact_relations': [
        'Unique recorder MSTX message -> same-thread fully contained aclmdlRIExecuteAsync.',
        'Official connection -> unique matching pid/device/stream/model execute and wait records.',
        'Strongly held source graph token + capture epoch -> exactly 40 source F21 sites.',
        'Each native envelope contains exactly 40 F21 tasks in one nonoverlapping serial stream; all input/weight/output shapes match the associated source history.',
        'Dedicated INT32 counter pack on the writer stream occurs after its F21 and before the next target F21 writer; all masks are BOOL.',
    ],
    'derivation': 'For two totally ordered finite sets of equal size, the order-preserving bijection is unique: the first source must map to the first native task, then induction fixes every remaining element.',
    'necessary_implementation_premises': [
        'The pinned current captured graph preserves the source F21 submission order on its one stream.',
        'Each selected current nonMegaMC2 source F21 call emits exactly one DispatchFFNCombine task; no extra/missing F21 call exists in the selected envelope.',
        'Graph references, capture epoch and current source bank identities retain the meanings recorded by the hook.',
    ],
    'premise_support_not_general_proof': 'Same-process startup exact CPU connections show one F21 native kernel for each of 736 observed public calls. This supports the pinned branch contract, not an all-program graph theorem or a loaded-binary/source identity certificate.',
    'not_established': [
        'Direct per-native-kernel CPU creator connection in the later NPU-only captured execution.',
        'Cross-rank communicator/iteration identity merely from equal ordinals or conservation.',
        'All baseline readers, arbitrary fusion legality, numerical equivalence or physical speedup/MFU.',
    ],
    'no_new_device_work': True,
    'historical_native_or_phase_promotions': 0,
    'whole_modules_closed': 0,
    'overall_complete': False,
})

# A single exact operator function, not another framework audit.
source = json.loads(read('sp-ma<REDACTED_TOKEN>.json')['stdout'])
pad = read('sp-mask-pad-native-type-audit.json')
by_call = collections.defaultdict(list)
for row in pad['rows']:
    by_call[(row['profile'], row['CPU_ROW'])].append(row)
assert len(by_call) == 32 and pad['profiles'] == 16
for rows in by_call.values():
    assert len(rows) == 5
    casts = [r['fields'] for r in rows if r['fields']['opType'] == 'Cast']
    assert sorted((x['inputDataTypes'], x['outputDataTypes'], x['inputShapes'], x['outputShapes']) for x in casts) == sorted([
        ('BOOL', 'INT8', '"20"', '"20"'),
        ('BOOL', 'INT8', '""', '""'),
        ('INT8', 'BOOL', '"24"', '"24"'),
    ])
    pp = [r['fields'] for r in rows if r['fields']['opType'] == 'PadV3']
    assert len(pp) == 1 and pp[0]['inputShapes'] == '"20;2;"' and pp[0]['outputShapes'] == '"24"'

# Independent finite check of the algebraic restriction of padded output.
checks = 0
for n in range(10):
    for x in itertools.product((False, True), repeat=n):
        for p in range(1, 5):
            padded = list(x) + [True] * ((-n) % p)
            c = len(padded) // p
            for r in range(p):
                reference = padded[r*c:(r+1)*c].copy()
                formula = [x[r*c+j] if r*c+j < n else True for j in range(c)]
                assert reference == formula
                checks += 1
write('SP-ma<REDACTED_TOKEN>.json', {
    'status': 'REVIEWED_EXACT_SEMANTIC_REDUCTION_IN_DECLARED_PUBLIC_FUNCTION_DOMAIN',
    'source': source,
    'actual_profile': {
        'profiles': 16, 'public_padding_calls': 32, 'native_executions': 160,
        'native_expansion_per_public_call': ['BOOL20->INT8_20', 'BOOL_scalar->INT8_scalar', 'MemSet_private_workspace_unknown', 'INT8_20+paddingINT32_2+INT8_scalar->INT8_24', 'INT8_24->BOOL24'],
        'cumulative_native_work_ns': sum(x['duration_ns'] for x in pad['rows']),
        'scope': 'Startup dummy/warmup/creation; not formal C8 baseline timing.',
        'catalog_empty_exact_20_to24_field_correction': 'The old diagnostic required the entire serialized inputShapes string to equal 20. PadV3 has compound inputShapes 20;2;, so the empty list did not mean absent execution. Exact CPU call, source line and per-native datatype/shape audit locate all 32 actual BOOL20->BOOL24 public calls.',
    },
    'declared_domain': {
        'input': 'One-dimensional arbitrary Boolean is_padding of length n; TP size p>=1, rank 0<=r<p.',
        'contract': 'Preserve Boolean values/dtype, output length, rank row ownership and a result that does not alias input. Input is not modified.',
        'allowed': 'Replace the whole functional padding-and-rank-slice computation and its internal representation.',
        'additional_guard': 'No required observer relies on the original full padded base allocation, its storage_offset or intermediate INT8 tensors. If such an observer exists, retain its observable contract or narrow the candidate.',
        'not_this_domain': 'Deleting an individual native MemSet in the old captured graph without proving its private workspace/readiness contract.',
    },
    'derivation': {
        'chunk': 'c=ceil(n/p)=(n+((-n) mod p))/p',
        'output': 'out[j]=is_padding[r*c+j] when r*c+j<n, otherwise True, for 0<=j<c.',
        'proof': 'F.pad appends exactly (-n) mod p True values. Restricting that padded vector to the returned rank interval gives the formula pointwise. The no-pad branch clones its slice, so a replacement must preserve no input aliasing. The formula covers n=0, n divisible by p, positive padding, partial local chunks and fully padded local chunks.',
        'n20_p8': {'chunk': 3, 'input_values_demanded_by_rank': [3, 3, 3, 3, 3, 3, 2, 0], 'output_values_per_rank': 3},
        'logical_external_array_bytes_in_byte_addressed_materialized_domain': {
            'full_pad20_to24': 44,
            'rank_function_n20_p8': [6, 6, 6, 6, 6, 6, 5, 3],
            'basis': 'Arbitrary demanded BOOL inputs must be read and fresh materialized BOOL outputs written once in a byte-addressed uncached external-array model. This is a conditional necessary logical I/O floor, not measured HBM traffic, wire bytes or time; zero-length output requires zero array bytes.',
        },
        'small_model_pointwise_formula_checks': checks,
        'review_limit': 'Algebra plus finite checks; not formal verification of native fusion, storage lifetime or a performance claim.',
    },
    'candidate_reuse': ['F21', 'F31', 'F32'],
    'conclusion': 'The entire public semantic boundary is a supported combination/reorganization candidate. Individual private native work is not proven removable. No more private MemSet argument probes are needed for this whole-function semantic conclusion.',
    'ceiling': {'state': 'SYMBOLIC_CEILING', 'formula': 'T>=B_required/B_external_upper only in the stated uncached byte-addressed materialized domain; actual required physical transfers and B_external_upper remain unknown.', 'physical_speedup': None},
    'new_whole_module_closures': 0, 'overall_complete': False,
})

# Module routing is an explicit frontier, not an inference of exclusive ownership
# from arbitrary ancestors in a call stack.
routes = {
    'AscendDSACPImpl._apply_wo_b': ['M11'], 'AscendDSACPImpl._forward_o_proj': ['M11'],
    'AscendDSAV41CPMetadataBuilder.build': ['M21'],
    'AscendDSAV41Impl._apply_rotary': ['M07', 'M09', 'M18'],
    'AscendDSAV41Impl._forward_attention': ['M10'],
    'AscendDSAV41Impl._write_compressed_source': ['M08'],
    'AscendDSAV41Impl.forward': ['M10'],
    'AscendDSAV41Impl.multistream_preprocess': ['M07', 'M08'],
    'AscendDSAV41MetadataBuilder.build': ['M21'],
    'AscendDSAV41MetadataBuilder.build.build_c2_metadata': ['M21'],
    'AscendDSAV41MetadataBuilder.build.build_qli_metadata': ['M21'],
    'AscendDSAV41MetadataBuilder.build.build_smla_metadata': ['M21'],
    'AscendDSparkProposer._clear_dummy_slot_mappings': ['M21', 'M24'],
    'AscendDSparkProposer.dummy_run': ['M19', 'M21'],
    'AscendEngram.forward': ['M14'], 'engram_gate': ['M14'],
    'AscendSpecDecodeBaseProposer._run_merged_draft': ['M17', 'M20', 'M24'],
    'DSparkMarkovHead.bias': ['M20'], 'DSparkMarkovHead.embed': ['M20'],
    'DeepseekV41Compressor.forward': ['M08'], 'DeepseekV41Compressor.pool_projected': ['M08'],
    'DeepseekV41DSparkModel.compute_logits': ['M16', 'M20'],
    'DeepseekV41DSparkModel.forward': ['M19'],
    'DeepseekV41DecoderLayer.forward': ['M06', 'M07', 'M12'],
    'DeepseekV41DecoderLayer.hc_collapse': ['M06', 'M15', 'M19'],
    'DeepseekV41DecoderLayer.hc_post': ['M06', 'M19'],
    'DeepseekV41DecoderLayer.hc_pre': ['M06', 'M19'],
    'DeepseekV41DecoderLayer.rms_norm_cast': ['M12', 'M19'],
    'DeepseekV41Indexer._output': ['M09'],
    'DeepseekV41Indexer.apply_query_rope': ['M09'],
    'DeepseekV41Indexer.project_weights': ['M09'],
    'DeepseekV41Indexer.quantize_query': ['M09'],
    'DeepseekV41Indexer.select_projected': ['M09'],
    'DeepseekV41Indexer.update_keys': ['M09'],
    'DeepseekV41MoE.forward': ['M12', 'M13'],
    'DeepseekV41Model.embed_input_ids': ['M05'],
    'DeepseekV41Model.forward': ['M05', 'M15'],
    'NPUModelRunner._dummy_run': ['M04', 'M22'],
    'UpdatableGraph.capture_begin': ['M22'],
    '_ascend_sp_padding_mask_impl': ['M21', 'M23'],
    '_ascend_sp_reduce_scatter_impl': ['M11', 'M13', 'M23'],
    '_ascend_sp_shard_impl': ['M23'],
    'compressed_slot_mapping': ['M21'], 'restore_tp_heads': ['M11', 'M23'],
    'scatter_cache_sk': ['M07', 'M08', 'M18'],
    'set_ascend_forward_context': ['M04', 'M21'],
    'sp_all_gather': ['M23'], 'sp_padding_mask': ['M21', 'M23'],
}
module_ledger = json.loads((O / 'module-profile-ledger-v154.json').read_text())['modules']
groups = []
by_module = collections.defaultdict(list)
for i, row in enumerate(catalog['catalog'], 1):
    a = row['source_anchor']; symbol = a['enclosing_qualified_symbol']
    assert symbol in routes, symbol
    mids = routes[symbol]
    gids = sorted({cid for m in module_ledger if m['id'] in mids for cid in m['candidate_links']})
    gid = 'OG169-' + str(i).zfill(3)
    groups.append(dict(id=gid, source_anchor=a, CPU_operator=row['CPU_operator'], native_kernel_multiset=row['native_kernel_multiset'], public_calls=row['calls'], native_executions=row['native_executions'], cumulative_native_work_ns=row['native_work_ns'], module_routing_hints=mids, existing_candidate_routing_hints=gids, multiple_native_kernels=len(row['native_kernel_multiset']) > 1, ownership_status='EXACT_CALLSITE_WITH_SEMANTIC_MODULE_ROUTING_HINTS_NOT_EXCLUSIVE_NATIVE_PHASE_OWNERSHIP'))
    for mid in mids:
        by_module[mid].append(gid)
multi = [x for x in groups if x['multiple_native_kernels']]
families = [
    {'id': 'PF169-INTEGER-METADATA', 'symbols': ['AscendDSAV41MetadataBuilder.build', 'AscendDSAV41MetadataBuilder.build.build_c2_metadata', 'compressed_slot_mapping'], 'candidates': ['F31'], 'question': 'Can the same row/index domain combine quotient/remainder/comparison and gather conversion without changing slot/page/acceptance generations?', 'gates': ['Exact integer division, overflow and negative/sentinel semantics', 'Shared input row domain and publication readiness', 'All outputs still required by consumers']},
    {'id': 'PF169-SP-MASK-PAD', 'symbols': ['_ascend_sp_padding_mask_impl', '_ascend_sp_shard_impl'], 'candidates': ['F21', 'F31', 'F32'], 'question': 'Analyze the full public padding/rank restriction boundary rather than isolated unknown-address native MemSet deletion.', 'gates': ['Full function semantic reduction reviewed only for Boolean mask function', 'Hidden-state padding has a different dtype/value/consumer contract', 'Storage/lifetime and capture interface guards']},
    {'id': 'PF169-DRAFT-SELECT', 'symbols': ['AscendSpecDecodeBaseProposer._run_merged_draft', 'DSparkMarkovHead.embed'], 'candidates': ['F30', 'F35'], 'question': 'Can row/column selection, cast and argmax or embedding selection be combined within existing demanded-output domains?', 'gates': ['Tie/NaN/dtype and exact selected rows', 'Inplace add and view update aliases are observable', 'Acceptance/rejection/rollback and Markov state consumers']},
    {'id': 'PF169-EMBEDDING-ID-MASK', 'symbols': ['DeepseekV41Model.embed_input_ids', 'DeepseekV41DSparkModel.forward'], 'candidates': ['F02', 'F28'], 'question': 'Reuse existing bounds/mask/local-ID/embedding composition clauses on these exact public-call sites.', 'gates': ['Vocabulary rank ownership', 'Masked out-of-range handling and reduction', 'DSpark state updates remain separate from pure ID arithmetic']},
    {'id': 'PF169-ENGRAM-GATE', 'symbols': ['engram_gate', 'AscendEngram.forward'], 'candidates': ['F24', 'F25'], 'question': 'Reuse the existing gate/projection composition domain; inspect required precision and view-copy boundaries only when they change a conclusion.', 'gates': ['Exact residual/gate numerical order', 'UVA readiness and all consumers', 'Inplace copy/alias and update epoch']},
]
for f in families:
    f['groups'] = [x['id'] for x in groups if x['source_anchor']['enclosing_qualified_symbol'] in f['symbols']]
covered = {g for f in families for g in f['groups']}
write('profile-operator-module-frontier.json', {
    'status': 'EXACT_CALLSITE_FRONTIER_FACTORING_WITH_ALL_MODULES_RETAINED',
    'scope': catalog['scope'],
    'counts': {'callsite_groups': len(groups), 'actual_profile_source_symbols': len(routes), 'native_compute_executions': sum(x['native_executions'] for x in groups), 'public_CPU_calls': sum(x['public_calls'] for x in groups), 'multi_kernel_callsite_groups': len(multi), 'multi_kernel_public_calls': sum(x['public_calls'] for x in multi), 'executions_beyond_one_per_multi_kernel_public_call': sum(x['native_executions']-x['public_calls'] for x in multi)},
    'groups': groups,
    'modules': [{'id': m['id'], 'name': m['name'], 'callsite_group_routing_hints': by_module[m['id']], 'evidence_state': 'STARTUP_EXACT_SOURCE_ANCHORS_WITH_EXPLICIT_ROLE_ALTERNATIVES' if by_module[m['id']] else 'NO_ROUTING_HINT_IN_THIS_STARTUP_NATIVE_CENSUS_NOT_ZERO_WORK_OR_NO_OPPORTUNITY', 'whole_module_closed': False} for m in module_ledger],
    'shared_opportunity_families': families,
    'remaining_multi_kernel_groups': [x['id'] for x in multi if x['id'] not in covered],
    'unassigned_callsite_groups': 0,
    'policy': ['Keep one condition question per semantic opportunity family, not every possible pair.', 'Exact same public CPU operator may emit multiple kernels for necessary implementation reasons. Kernel count/name equality alone never proves duplicate work, legal removal or latency gain.', 'Cumulative native durations are not critical path latency; startup cold operations do not rank formal C8 steady-state bottlenecks.', 'Module routing is an evidence index; alternatives are not duplicate work to be summed. No whole-module utilization ranking or branch closure follows from a source ancestor.'],
    'whole_modules_closed': 0, 'overall_complete': False,
})
print(json.dumps({'graph_joins': native['joins'], 'whole_mask_function_formula_checks': checks, 'pad_calls': len(by_call), 'module_frontier_groups': len(groups), 'multi_kernel_groups': len(multi), 'modules_retained': len(module_ledger)}, ensure_ascii=False))
