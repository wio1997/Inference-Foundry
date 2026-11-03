"""Operator semantic refinements and same-domain logical work models; no kernels."""
import json
import math
from pathlib import Path

H = Path(__file__).resolve().parent
O = H.parent / 'profile-first-20261031'


def read(name):
    return json.loads((H/name).read_text())


def write(name, value):
    (H/name).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


d = read('typed-frontier-observation.json')['review']
assert d['callsite_groups'] == 120 and d['typed_regimes'] == 148
assert d['public_CPU_calls'] == 15520 and d['native_executions'] == 21088


def rows(gid):
    return [r for r in d['regimes'] if r['callsite_group'] == gid]


arg = rows('OG169-030')
assert len(arg) == 1 and arg[0]['calls'] == 160
a = arg[0]
assert [x['opType'] for x in a['native_sequence']] == ['Slice', 'ArgMaxV2', 'Cast']
assert a['native_sequence'][1]['inputDataTypes'] == 'DT_BF16;INT64'
assert a['native_sequence'][1]['outputDataTypes'] == 'INT32'
assert a['native_sequence'][2]['inputDataTypes'] == 'INT32'
assert a['native_sequence'][2]['outputDataTypes'] == 'INT64'
M, V = 4, 129280
reference = 6*M*V + 16*M
necessary = 2*M*V + 8*M
write('draft-argmax-public-boundary-refinement.json', {
    'status': 'REVIEWED_CONDITIONAL_PUBLIC_OPERATOR_COMBINATION_DOMAIN',
    'candidate_reuse': ['F30', 'F35'],
    'actual_profile': a,
    'resolved': [
        'The native ArgMax compares BF16 scores; the final Cast widens INT32 indices to INT64. There is no BF16-to-FLOAT score Cast in this public CPU-call expansion.',
        'All160 exact current public CPUcalls have three distinct native task IDs in one stream.',
        'The public input is a selected BF16 view with per-row physical capacity 646400=5*129280; output is four INT64 IDs.',
    ],
    'whole_boundary_domain': {
        'input': 'The same already corrected score view supplied to this public argmax; M=4,V=129280.',
        'output': 'Same INT64 indices, order, native tie/NaN/error semantics; preserve score input and all surrounding updated-score/raw/confidence consumers.',
        'allowed_graph': 'BF16 selected view -> equivalent BF16 comparison/selection -> INT64 index output, without requiring globally materialized private selected-score and INT32-index planes.',
        'not_required_for_this_domain': 'Death of the public corrected score bank or all raw/logprob/confidence consumers; their storage and producers remain.',
        'guards': ['Preserve the exact selected row/column/stride domain and input readiness.', 'The internal temporaries have no required external storage/alias/error observer beyond the public operator contract.', 'Preserve BF16 comparison and tie/NaN behavior, and exact output dtype. Do not cancel earlier BF16 bias/Add rounding.', 'V<2^31 makes every valid INT32 index exactly representable after widening; replacing dtype storage is not permission to change argmax semantics.'],
    },
    'logical_array_model': {
        'domain': 'Byte-addressed cold external input, fresh materialized output, full-scan dense comparison domain; logical array accesses excluding scalar/slice metadata. Not measured HBM/wire traffic.',
        'reference_minimum_array_accesses': 'Slice read+write 4MV; ArgMax read2MV and write4M; indexCast read4M+write8M =>6MV+16M bytes.',
        'reference_bytes_per_call': reference,
        'combined_required_array_IO_floor': '2MV+8M bytes in the stated full-scan materialized-output domain.',
        'combined_required_bytes_per_call': necessary,
        'private_materialization_interface_budget_bytes': reference-necessary,
        'note': 'The floor does not promise a one-pass implementation, physical bandwidth or attainable runtime. Another separate Slice at the same Python line has160 calls; it is not silently paired or added to this exact public-call budget.',
    },
    'symbolic_ceiling': {
        'state': 'SYMBOLIC_CEILING',
        'lower_bound': 'L=max((2MV+8M)/B_external_upper, M(V-1)/P_pairwise_compare_upper), only in the stated external-array/full-scan binary candidate-comparison domain.',
        'mode_and_pool': 'BF16 comparison/Vector and external storage; INT8/BF16 Cube MAC peaks do not bound this comparison capacity.',
        'measured_native_public_call_median_us': a['native_work_ns']['median']/1000,
        'physical_MFU_or_speedup': None,
        'limits': 'The necessary comparison work is a restricted binary candidate-comparison model; other algorithms/preprocessed summaries/constant inputs change the domain. Actual input values, cache state and capacity upper bounds remain unknown.',
    },
    'new_whole_module_closures': 0, 'overall_complete': False,
})

divs = [r for r in d['regimes'] if r['CPU_operator'] in ['aclnnDivMods', 'aclnnFloorDivides']]
assert len(divs) == 5 and sum(r['calls'] for r in divs) == 960
for r in divs:
    seq = r['native_sequence']
    assert [x['opType'] for x in seq] == ['Cast', 'FloorDiv', 'Cast']
    assert seq[0]['inputDataTypes'] == 'INT32' and seq[0]['outputDataTypes'] == 'INT64'
    assert seq[1]['inputDataTypes'] == 'INT64;INT64' and seq[1]['outputDataTypes'] == 'INT64'
    assert seq[2]['inputDataTypes'] == 'INT64' and seq[2]['outputDataTypes'] == 'INT32'
checks = 0
for x in sorted(set([-(2**31), -(2**31)+1, -2**24-1, -1025, -257, -256, -255, -129, -128, -127, -3, -2, -1, 0, 1, 2, 3, 127, 128, 129, 255, 256, 257, 1025, 2**24+1, 2**31-2, 2**31-1])):
    for divisor in [1, 2, 3, 7, 8, 127, 128, 129, 255, 256, 257, 2**24+1, 2**31-1, 2**31, 2**63-1]:
        q = x // divisor
        assert -(2**31) <= q < 2**31
        trunc_q = abs(x)//divisor * (-1 if x < 0 else 1)
        rem = x-trunc_q*divisor
        assert trunc_q-(1 if rem < 0 else 0) == q
        assert 0 <= x-q*divisor < divisor
        checks += 1
models = []
for r in divs:
    n = int(r['native_sequence'][0]['inputShapes'].strip('"'))
    models.append({'group': r['callsite_group'], 'N': n, 'calls': r['calls'], 'separate_reference_array_bytes': 40*n, 'combined_required_array_IO_bytes': 8*n, 'private_materialization_interface_budget_bytes': 32*n})
rem = [r for r in d['regimes'] if r['CPU_operator'] == 'aclnnRemainderTensorScalar']
write('metadata-integer-public-boundary-refinement.json', {
    'status': 'REVIEWED_CONDITIONAL_PUBLIC_INTEGER_DOMAINS_WITH_ACTUAL_TYPED_EVIDENCE',
    'candidate_reuse': ['F31'],
    'reused_source_contract': str(O/'metadata-field-operator-contract-v68.json'),
    'actual_quotient_calls': 960, 'actual_quotient_typed_regimes': 5,
    'actual_quotient_native_expansion': 'INT32 input -> materialized INT64 -> INT64 FloorDiv -> materialized INT64 quotient -> INT32 output.',
    'semantic_domain': 'For x in signedINT32 and positive signedINT64 divisor d, mathematical floor(x/d) lies in signedINT32. Preserve baseline zero/negative-divisor/error behavior outside this domain or use a separate guarded branch.',
    'proof': 'If x>=0 then0<=floor(x/d)<=x<=INT32_MAX. If x<0 andd>=1 then x<=floor(x/d)<=-1,so quotient>=INT32_MIN. Widening the input and narrowing this result does not change mathematical integer values.',
    'negative_and_overflow_guards': ['Floor differs from truncation for negative nonmultiples: floor(-1/2)=-1,not0.', 'Do not implement abs(INT32_MIN) or q*d in overflowing signedINT32 arithmetic. Keeping sign-extended operands in registers is allowed; no narrower native instruction or lower arithmetic cost has been proved.', 'Preserve INT64 slot/position domains where the actual public input is INT64; fixed max sequence length alone does not bound physical slot IDs.', 'Same shapes do not prove two quotient/remainder calls consume the same values, divisor or generation. Joint q/r composition remains conditional on existing v68 operand/epoch guards.'],
    'whole_public_operator_conclusion': 'Internal global Cast planes and scalar Broadcast planes are not semantic prerequisites for a whole-function replacement. Their individual native deletion/alias safety is not claimed. Actual existing INT64 arithmetic may remain; the supported opportunity is eliminating mandatory global materialization between public operator internals.',
    'logical_array_models': models,
    'reference_array_model': 'WidenCast4N+8N,FloorDiv8N+8N,NarrowCast8N+4N=40N; fresh INT32 output/input interface8N. Scalars/constants and cache/physical transactions are excluded.',
    'remainder_actual_typed_regimes': len(rem),
    'remainder_actual_calls': sum(r['calls'] for r in rem),
    'remainder_semantic_refinement': 'A scalar divisor can remain scalar in a whole public remainder operator; materializing N identical divisor elements is an internal representation choice. Preserve INT32 orINT64 width and negative floor-mod behavior; a narrowed divisor or slot domain is not inferred.',
    'integer_edge_checks': checks,
    'symbolic_ceiling': {'state': 'SYMBOLIC_CEILING', 'quotient_IO_floor': '8N/B_external_upper in the cold byte-addressed materialized-output model, plus any stronger valid typed integer instruction/resource floor; do not apply Cube MAC peaks.', 'physical_integer_capacity': None, 'physical_speedup': None},
    'new_whole_module_closures': 0, 'overall_complete': False,
})

embed = rows('OG169-071')
assert len(embed) == 1 and embed[0]['calls'] == 160
assert [x['opType'] for x in embed[0]['native_sequence']] == ['Slice', 'GatherV2']
write('Markov-embedding-strided-ID-refinement.json', {
    'status': 'REVIEWED_CONDITIONAL_PUBLIC_SELECTED_ID_DOMAIN',
    'candidate_reuse': ['F30'], 'actual_profile': embed[0],
    'actual_geometry': {'B': 4, 'R': 256, 'vocabulary_rows': 129280, 'ID_base_capacity': [4, 6], 'selected_IDs': 4, 'output_BF16_values': 1024},
    'conclusion': 'The whole public embedding call can consume the same strided selected ID view and preserve its BF16 output without a separately materialized ID slice. No preselection of future recurrent IDs or removal of confidence embedding outputs is justified.',
    'logical_array_model': {'reference_bytes_assuming_per_occurrence_row_reads': 4192, 'combined_bytes_assuming_per_occurrence_row_reads': 4128, 'private_ID_materialization_budget_bytes_per_call': 64, 'cold_unique_row_lower_model': '8B+2UR+2BR, where U is number of distinct valid selected IDs and0<=U<=B. Actual U/cache/weight residency unknown.', 'limits': 'The full129280x256 weight shape does not require reading the entire table. A64B ID interface is a small byte opportunity;two kernels alone do not justify high priority or imply latency savings.'},
    'guards': ['Identical selected IDs/order, valid-range/error behavior and strided storage.', 'Selected output remains for all recurrence/confidence consumers.', 'Same W1/table generation and readiness; native-only Slice deletion remains separate from the whole public call.'],
    'symbolic_ceiling': {'state': 'SYMBOLIC_CEILING', 'conditional_IO_floor': '(8B+2UR+2BR)/B_external_upper in the declared cold external-array materialized-output domain', 'physical_speedup': None},
    'whole_modules_closed': 0, 'overall_complete': False,
})

src = read('Engram-hidden-cast-targeted-contract.json')
assert src['sha256'] == '4797b8962264d137cdb142a9988825288283092fb797ddbd334c6ab2223eb58d'
first = [r for r in d['regimes'] if r['callsite_group'] == 'OG169-156']
second = [r for r in d['regimes'] if r['callsite_group'] == 'OG169-091' and r['native_sequence'][0]['inputDataTypes'] == 'DT_BF16']
assert len(first) == len(second) == 1 and first[0]['calls'] == second[0]['calls'] == 32
assert first[0]['source_anchor']['source_sha256'] == second[0]['source_anchor']['source_sha256'] == src['sha256']
assert first[0]['native_sequence'][0]['inputShapes'] == second[0]['native_sequence'][0]['inputShapes'] == '"3,4,5120"'
T, h, hidden = 3, 4, 5120
elements = T*h*hidden
write('Engram-hidden-cast-fork-semantic-cut.json', {
    'status': 'REVIEWED_CONDITIONAL_LOCAL_COMMON_CONVERSION_FORK',
    'id': 'CUT170-ENGRAM-HIDDEN-CAST-FORK',
    'candidate_reuse': ['F25'],
    'reuses_old_gate_graph_and_rounding_contract': str(O/'Engram-hash-UVA-and-gate-costs-v62.json'),
    'targeted_code_question_and_pin': src,
    'actual_profile': {'first_hidden_float': first[0], 'residual_hidden_float': second[0]},
    'semantic_derivation': 'The pure function reads hidden.float() atline62 and again atline72. It never assigns to or writes hidden. Optional rotation rebinds original to a new matrix result; local rstd*= updates only rstd. If inputbytes/conversionsemantics remain unchanged and casts have no required external allocation observer, both casts produce the same FP32 values. A single pre-rotation cast may feed both the rotation/gate branch and the final unrotated residual branch.',
    'candidate_graph': 'Cast_BF16_to_FLOAT(hidden) -> fork{optional_rotation->gate, unchanged_FP32_hidden->ordered_residual_add->BF16_round}',
    'important_non_equivalence': {'input_hidden': [1, 2], 'rotation': [[0, 1], [1, 0]], 'gate_times_value': [0, 0], 'correct_unrotated_residual': [1, 2], 'wrong_reuse_of_rotated_original': [2, 1]},
    'guards': ['No concurrent relevant input mutation and identical conversion semantics across the two original sites.', 'Retain the pre-rotation FP32 value; the rotated value is not a residual substitute.', 'Preserve full norm/dot/special/mask evaluation, FP32 multiply/add order, final BF16 round, NaN/Inf/signed-zero semantics and every output consumer.', 'Preserve any required input/storage/allocator/capture/error observer; actual native allocation and safe new handoff are not proved.', 'Account for extended lifetime of the FP32 activation using the same shared memory budget as other candidates.'],
    'logical_value_and_lifetime_model': {
        'shape': [T, h, hidden], 'elements': elements,
        'one_removed_conversion_interface_bytes': 6*elements,
        'additional_live_FP32_storage_at_most_bytes_per_call': 4*elements,
        'calls_with_each_observed_site': 32,
        'logical_removed_conversion_interface_bytes_over_observed_calls': 32*6*elements,
        'scope': 'One potential 2B-read+4B-write conversion interface;not measuredHBM. Keeping the first FP32 allocation live can increase memory peak before the second cast. Do not sum per-call lifetime bytes into simultaneous capacity or count an ideal-gate/wholefusion credit twice.',
    },
    'what_is_new': 'Actual duplicated BF16->FP32 input-conversion sites and a precise local fork/lifetime condition are bound to the same pinned function. Existing versioned q/k-parameter and rotation-weight preparation are not counted as new discoveries.',
    'symbolic_ceiling': {'state': 'SYMBOLIC_CEILING', 'physical_capacity_or_speedup': None, 'note': 'The fullgate necessary work and resource lower bounds must be recomputed jointly;do not sum independent operator lower bounds or infer gain from one Cast duration.'},
    'new_native_reader_or_alias_proofs': 0, 'whole_modules_closed': 0, 'overall_complete': False,
})
write('typed-operator-question-review.json', {
    'previous_goal_turn': 'PROGRESS',
    'status': 'FOUR_OPERATOR_DOMAINS_REFINED_FROM_ACTUAL_TYPED_PROFILES',
    'profile': {'callsite_groups': 120, 'typed_regimes': 148, 'public_CPU_calls': 15520, 'native_executions': 21088, 'elapsed_CPU_analysis_seconds': read('typed-frontier-observation.json')['status']['elapsed_seconds']},
    'refinements': [
        {'file': 'draft-argmax-public-boundary-refinement.json', 'current_evidence_resolves': 'BF16comparison andindexCast64,actual3kernelpublicboundary;standaloneargmaxdoesnotrequiredeadpublicscoreconsumers.'},
        {'file': 'metadata-integer-public-boundary-refinement.json', 'current_evidence_resolves': '960actualINT32quotientpubliccalls materializeINT64planes;guardedwholepublicsemantics andnegative/INT_MINconditions reviewed.'},
        {'file': 'Markov-embedding-strided-ID-refinement.json', 'current_evidence_resolves': 'Selected4IDs/4weightrows,notfulltable;64BprivateIDinterface rankslowbybytes.'},
        {'file': 'Engram-hidden-cast-fork-semantic-cut.json', 'current_evidence_resolves': '32pairs ofobservedhiddenBF16->FLOATconversions atsame27linepurefunction;pre-rotationforkand245760Bpotentialextendedlifetime.'},
    ],
    'not_claimed': ['Implementation or runnablefusion', 'Physicalspeedup,requestcriticalpathgain orwholemoduleMFU', 'Allreader/nativealias/globalepochproofs', 'Historicaloperand/phasebackfill orwholeoriginalscopecompletion'],
    'whole_modules_closed': 0, 'overall_complete': False,
})
print(json.dumps({'status': 'PROGRESS', 'typed_regimes': 148, 'argmax_reference_bytes': reference, 'argmax_required_bytes_in_model': necessary, 'quotient_calls': 960, 'integer_edge_checks': checks, 'Engram_cast_interface_bytes': 6*elements, 'Engram_extended_lifetime_bytes': 4*elements}))
