"""Normalize existing evidence into explicit P5 comparisons; no device work."""
import hashlib
import json
from fractions import Fraction
from pathlib import Path

H = Path(__file__).resolve().parent
N = H.parent
O = N / 'profile-first-20261031'


def read(p):
    return json.loads(p.read_text())


def pin(p):
    b = p.read_bytes()
    return {'path': str(p), 'bytes': len(b), 'sha256': hashlib.sha256(b).hexdigest()}


def rational(v):
    f = Fraction(v)
    return {'exact_fraction': str(f), 'decimal_for_display': float(f)}


routes_path = N / 'shared-profile-analysis-v172/all26-module-metric-and-P5-routes.json'
routes = read(routes_path)
composition_path = H / 'all42-composition-coverage.json'
composition = read(composition_path)
matrix_path = O / 'module-phase-branch-matrix-v173.json'
matrix = read(matrix_path)
modules = {m['id']: m for m in matrix['modules']}
comparisons = []

for route in routes['modules']:
    mid = route['module']
    comparisons.append({
        'object_id': mid,
        'object_kind': 'WHOLE_FUNCTIONAL_MODULE_CONDITIONAL_INVOCATION',
        'graph_id': 'SOURCE_CONDITIONAL_MODULE_AND_REGISTERED_COMPOSITIONS',
        'case_id': 'case_and_guard_defined_by_branch_matrix',
        'branch_ids': [b['id'] for b in modules[mid]['branches']],
        'implementation_domain': {
            'scope': route['resource_scope'],
            'contract': str(matrix_path) + '#' + mid,
            'original_baseline_and_rewrites_are_distinct': True,
            'necessary_work_must_be_recomputed_after_rewrite': True,
        },
        'scope_match_status': 'SYMBOLIC_MODULE_CONTRACT_NOT_MEASURED_MODULE_LATENCY',
        'effective_work_Q': 1,
        'work_unit': 'required_logical_module_invocation_in_declared_case',
        'measured_time_T': None,
        'measured_time_symbol': 'T_' + mid + '(case,guard)',
        'time_lower_L': {'expression': route['P5_time_lower_bound'], 'numeric_value': None},
        'time_unit': 'second',
        'measured_throughput': {'expression': '1/T_' + mid + '(case,guard)', 'numeric_value': None},
        'throughput_upper_U': {'expression': '1/L_' + mid + '(case,guard) when L>0', 'numeric_value': None},
        'measured_over_ceiling_ratio': {'expression': 'L_' + mid + '/T_' + mid, 'numeric_value': None},
        'time_gap_ratio': {'expression': 'T_' + mid + '/L_' + mid + ' when L>0', 'numeric_value': None},
        'utilization_metric_reference': {
            'path': str(routes_path), 'module': mid,
            'reported_activity': route['primary_reported_activity'],
            'metric_status': route['primary_status'],
            'metric_cohort': 'v164 two formal windows;exclusive functional ownership only',
            'not_attached_to_unmeasured_candidate_invocation': True,
        },
        'dominant_constraint_ids': ['UNKNOWN_UNTIL_RESOURCE_TERMS_AND_CASE_ARE_BOUND'],
        'bound_status': 'SYMBOLIC_CONDITIONAL_WITH_EXPLICIT_MISSING_INPUTS',
        'bound_input_path': str(routes_path),
        'calculation_path': str(Path(__file__).resolve()),
        'source_ids': [str(routes_path), str(matrix_path)],
        'unknown_ids': ['MATCHED_WHOLE_MODULE_LATENCY', 'NECESSARY_WORK_AFTER_REWRITE', 'APPLICABLE_RESOURCE_UPPER_RATES'],
        'missing_evidence': route['P5_required_work_and_capacity_missing'],
        'candidate_only': False,
        'diagnosis': route['diagnostic_scope_note'],
    })
    # A partial direct-dense census has its own exact matched-time denominator.
    # Sum of invocation times is an aggregate service-time comparison, never
    # a whole-module walltime or an all-rank capacity denominator.
    for partial in route['P5_numeric_partial_comparisons']:
        cases = partial.get('FLOAT_restricted_mode_cases') or [partial]
        for case in cases:
            mode = case['mode']
            r = {'BF16': 4096, 'FP16': 4096, 'INT8': 8192,
                 'FLOAT_HF32_OFF': 1024, 'FLOAT_HF32_ON': 2048}[mode]
            q = int(partial['conditional_MAC'])
            t = Fraction(int(partial['matched_kernel_sum_ns']), 10**9)
            cycles = Fraction(q, 24*r)
            ref_l = cycles / 1_800_000_000
            comparisons.append({
                'object_id': mid + '/direct-dense/' + partial['window_cluster'] + '/' + mode,
                'object_kind': 'MATCHED_PARTIAL_INVOCATION_POPULATION',
                'graph_id': 'CURRENT_V164_NATIVE_EXECUTIONS',
                'case_id': partial['window_cluster'],
                'implementation_domain': {'algorithm': 'classical_direct_dense_with_all_counted_MAC_required',
                                          'mode': mode, 'guards': partial['guards']},
                'scope_match_status': 'WORK_AND_CUMULATIVE_KERNEL_TIME_MATCH_SAME_SELECTED_EXECUTIONS;MODE_CONDITIONAL',
                'execution_count': partial['executions'],
                'effective_work_Q': q, 'work_unit': 'MAC',
                'measured_time_T': rational(t), 'time_unit': 'second',
                'time_lower_L': {'expression': f'{q}/(24*{r}*f_upper)', 'numeric_value': None,
                                 'cycle_equivalent': rational(cycles)},
                'measured_throughput': rational(Fraction(q)/t),
                'throughput_upper_U': {'expression': f'24*{r}*f_upper', 'unit': 'MAC_per_second', 'numeric_value': None},
                'measured_over_ceiling_ratio': {'expression': f'{q}/(24*{r}*f_upper*({t}))', 'numeric_value': None},
                'time_gap_ratio': {'expression': f'({t})*24*{r}*f_upper/{q}', 'numeric_value': None},
                'reference1800MHz_only': {
                    'rated_frequency_is_not_physical_maximum': True,
                    'time_floor_seconds': rational(ref_l),
                    'L_over_T': rational(ref_l/t), 'T_over_L': rational(t/ref_l),
                },
                'utilization_metric_reference': {'path': str(routes_path), 'module': mid,
                                                'broader_module_metric_not_this_partial_subset': True},
                'dominant_constraint_ids': ['DIRECT_DENSE_CUBE_COMPUTE_ONLY;OTHER_RESOURCES_NOT_PRICED'],
                'bound_status': 'CONDITIONAL_SYMBOLIC_FREQUENCY_AND_RATED_REFERENCE',
                'bound_input_path': str(routes_path), 'calculation_path': str(Path(__file__).resolve()),
                'source_ids': [str(routes_path)],
                'unknown_ids': ['PHYSICAL_FREQUENCY_UPPER', 'LIVE_ARITHMETIC_MODE', 'OTHER_RESOURCE_FLOORS'],
                'candidate_only': False,
                'aggregation': 'Sum selected invocation times and work;no simultaneous16device walltime claim;FLOAT cases are alternatives,not additive.',
            })

for c in composition['candidate_bindings']:
    cid = c['candidate']
    comparisons.append({
        'object_id': cid,
        'object_kind': 'UNIMPLEMENTED_CONDITIONAL_COMPOSITION',
        'graph_id': c['templates'],
        'case_id': 'candidate_branch_contract_and_same_invocation_bindings',
        'implementation_domain': {'contract': c['composition_contract'], 'template_binding': c},
        'scope_match_status': 'CANDIDATE_ONLY_NO_TRANSFER_OF_BASELINE_TIME_OR_ACTIVITY',
        'effective_work_Q': 1, 'work_unit': 'required_logical_subgraph_evaluation',
        'measured_time_T': None, 'time_unit': 'second',
        'time_lower_L': {'expression': 'max_r(W_necessary_' + cid + '_on_shared_pool_r/C_upper_r, B_necessary_cut_' + cid + '/BW_cut_upper, proved_release_or_recurrence_floors)', 'numeric_value': None},
        'measured_throughput': None,
        'throughput_upper_U': {'expression': '1/L_' + cid + ' when L>0', 'numeric_value': None},
        'measured_over_ceiling_ratio': None, 'time_gap_ratio': None,
        'utilization_metric_reference': None,
        'dominant_constraint_ids': ['UNKNOWN;KEEP_MODE_POOL_CUT_AND_LIFETIME_TERMS_SEPARATE'],
        'bound_status': 'CONDITIONAL_SYMBOLIC_CANDIDATE;NO_MEASURED_CANDIDATE',
        'bound_input_path': c['composition_contract'],
        'calculation_path': str(Path(__file__).resolve()),
        'source_ids': [str(composition_path), c['composition_contract']],
        'unknown_ids': ['SELECTED_LEGAL_REWRITE_AND_GUARDS', 'NON_DUPLICATE_NECESSARY_WORK', 'PHYSICAL_RESOURCE_RATE_UPPERS'],
        'candidate_only': True,
        'not_speedup_or_implementation_proof': True,
    })

required = {'object_id','object_kind','graph_id','case_id','implementation_domain','scope_match_status',
            'effective_work_Q','work_unit','measured_time_T','time_lower_L','time_unit','measured_throughput',
            'throughput_upper_U','measured_over_ceiling_ratio','time_gap_ratio','utilization_metric_reference',
            'dominant_constraint_ids','bound_status','bound_input_path','calculation_path','source_ids','unknown_ids','candidate_only'}
assert all(required <= set(c) for c in comparisons)
assert len({c['object_id'] for c in comparisons}) == len(comparisons)
assert len([c for c in comparisons if c['object_kind'].startswith('WHOLE_')]) == 26
assert len([c for c in comparisons if c['candidate_only']]) == 42
out = {
    'version': 'OPERATOR-CEILING-v2',
    'status': 'EXPLICIT26_MODULE_AND42_COMPOSITION_P5_CONTRACTS_WITH_MATCHED_DENSE_SUBSETS',
    'performance_comparisons': comparisons,
    'source_pins': [pin(p) for p in [routes_path, composition_path, matrix_path]],
    'missing_is_not_zero': True,
    'physical_MFU_or_reachable_speedup_not_required_for_diagnosis': True,
    'coverage': {'modules':26, 'registered_candidates':42,
                 'matched_dense_mode_populations':len(comparisons)-68,
                 'whole_module_numeric_bounds_complete':False,
                 'source_conditional_case_analysis_not_live_branch_exhaustiveness':True},
    'overall_complete': False,
}
(H/'module-analysis.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
(H/'P5-contract-checks.json').write_text(json.dumps({'status':'PASS','comparison_count':len(comparisons),
    'required_fields':sorted(required),'unique_object_ids':True,'all26_and42_represented':True,
    'all_inputs_bound_or_explicitly_symbolic':True,'physical_ceiling_or_whole_scan_not_proved':True},indent=2)+'\n')
print(json.dumps({'status':'PASS','comparisons':len(comparisons),'modules':26,'candidates':42,
                  'matched_partial_mode_populations':len(comparisons)-68}))
