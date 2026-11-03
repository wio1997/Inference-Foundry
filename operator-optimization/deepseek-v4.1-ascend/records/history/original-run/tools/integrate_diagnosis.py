"""Integrate existing evidence; no target code changes or device workload.

Controller-authored interpretation is explicit and separate from arithmetic.
This script does not assert a complete primitive graph or physical ceiling.
"""
from pathlib import Path
from fractions import Fraction
from decimal import Decimal
from collections import Counter
from itertools import combinations
import datetime, csv, hashlib, json, shlex, shutil, subprocess, sys

T = Path(__file__).resolve().parents[1]
K = T.parents[1]
Q = Path('/Users/wio/work/173-sglang-matched-172-benchmark')
R = '/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41'
SSH = '/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh'
C = 'wio-vllm-v41-repro169-stage-20261028'
now = datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(name):
    return json.loads((T / name).read_text())
def write(name, data):
    p = T / name; p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

# Preserve the complete call aggregate locally without emitting it to the model.
p = T / 'captured-layer-attribution/module-call-metrics.json'
if not p.exists():
    code = 'from pathlib import Path;p=Path(' + repr(R + '/captured-layer-attribution/module-call-metrics.json') + ');assert p.stat().st_size<20*1024*1024;print(p.read_text(),end="")'
    r = subprocess.run([SSH, shlex.join(['docker', 'exec', C, 'python3', '-c', code])], capture_output=True, timeout=120)
    assert r.returncode == 0, r.stderr.decode()
    p.write_bytes(r.stdout)
calls = read('captured-layer-attribution/module-call-metrics.json')
assert len(calls) == 4160
audit = read('counter-field-audit/summary.json')
assert audit['status'].startswith('PASS_') and len(audit['ranks']) == 16
interior = [r for r in calls if r['run_in_model'] in [1, 2, 3]]
assert len(interior) == 1920
metric_name = 'CSV_REPORTED_AIC_CYCLE_WEIGHTED_MAC_RATIO'
metric_definition = 'sum(aic_total_cycles * aic_mac_ratio) / sum(aic_total_cycles), over attributed kernel rows. Both inputs are vendor CSV-reported fields; ratios are rounded. This derived counter proxy is not MFU, whole-NPU busy time, or a physical-peak fraction.'
modules = []
for layer in range(40):
    rs = [r for r in interior if r['layer'] == layer]
    assert len(rs) == 48 and len({r['rank'] for r in rs}) == 16
    num = sum((Decimal(r['aic_mac_cycle_proxy_numerator']) for r in rs), Decimal(0))
    den = sum((Decimal(r['aic_cycle_denominator']) for r in rs), Decimal(0))
    vnum = sum((Decimal(r['aiv_vec_cycle_proxy_numerator']) for r in rs), Decimal(0))
    vden = sum((Decimal(r['aiv_cycle_denominator']) for r in rs), Decimal(0))
    durations = [r['window_ns'] for r in rs]
    modules.append({'module_id': f'target.layers.{layer}.forward', 'layer': layer,
                    'case_id': 'CAPTURED_TARGET_DECODE_INTERIOR_RUNS_1_2_3', 'executed': True,
                    'call_count': 48, 'rank_count': 16,
                    'metrics': [{'name': metric_name, 'definition': metric_definition,
                                 'numerator': str(num), 'denominator': str(den), 'value': float(num / den),
                                 'unit': 'dimensionless_reported_counter_proxy', 'evidence_type': 'DERIVED_FROM_MEASURED_REPORT',
                                 'sample_count': 48, 'sampling_resolution': 'Vendor PMU report ratios; rounding and counter semantics retained as limitations',
                                 'device_core_scope': 'AIC report counters for attributed kernels across all16 logical devices',
                                 'aggregation': 'Ratio of summed reported-cycle proxy numerators and denominators; not mean of per-call ratios'},
                                {'name': 'CSV_REPORTED_AIV_CYCLE_WEIGHTED_VEC_RATIO', 'numerator': str(vnum), 'denominator': str(vden),
                                 'value': float(vnum / vden) if vden else None, 'unit': 'dimensionless_reported_counter_proxy'}],
                    'mean_window_us': float(Fraction(sum(durations), 48 * 1000)),
                    'sum_window_ns': sum(durations), 'max_window_us': max(durations) / 1000,
                    'known_task_union_ns_sum': sum(r['known_task_union_ns'] for r in rs),
                    'critical_path_exposed_time': None,
                    'attribution_status': 'SOURCE_SEQUENCE_PLUS_INDEPENDENT_ROLE_INVARIANTS_VALIDATED',
                    'source_ids': ['CAPTURED_TARGET_MAPPING', 'COUNTER_FIELD_AUDIT']})
ordered = sorted(modules, key=lambda r: r['metrics'][0]['value'])
assert ordered[0]['layer'] == 0
robust = read('captured-layer-attribution/interior-robustness.json')
assert robust['lowest_identity_stable']
for row in ordered:
    prior = next(x for x in robust['interior_all_40_ranking'] if x['layer'] == row['layer'])
    assert abs(row['metrics'][0]['value'] - prior['cycle_weighted_reported_mac_ratio']) < 1e-12

contract = read('contract.json')
contract['utilization_contract'].update(primary_metric_name=metric_name, definition_and_source=metric_definition,
    numerator='sum reported aic_total_cycles*aic_mac_ratio', denominator='sum reported aic_total_cycles',
    unit='dimensionless_reported_counter_proxy', timing_window='Source-validated captured target decoder calls, per-rank/model runs1,2,3',
    device_core_scope='16logical devices; AIC report counter normalization retained',
    aggregation='Sum numerator/denominator, all48calls per layer; no cross-rank wall-clock alignment required',
    case_aggregation_weights='Exactly3calls per rank,16ranks,40decoder modules; all-capture104calls/layer separately retained',
    not_whole_device_utilization_or_MFU=True)
contract['measurement_scope_refinement'] = {'comparable_modules': [m['module_id'] for m in modules],
    'missing_scope': ['Eager target prefill', 'Eager3block DSpark', 'Engram CPU lookup and outside-decoder gate', 'Root/head/metadata'],
    'scope_was_not_changed_to_claim_full_coverage': True}
write('contract.json', contract)

fusion = read('fusion-scan.json')
records = read('records.json')
review = read('evidence/source-review-31.json')
existing = [x['check_id'] for x in records['existing_fusions']]
conditional = [x['id'] for x in review['items'] if x['status'] == 'CONDITIONAL']
assert len(conditional) == 27 and len(existing) == 3
fusion.update(status='PAUSED_WITH_GAPS', last_full_pass_id='STRUCTURAL_PASS_2',
    stop_reason={'kind': 'ANALYSIS_PHASE_ONLY_EVIDENCE_GAPS',
                 'finite_structural_pass': '12templates/221nodes, connected2-4node sets and sibling branches:928items; second identical pass adds0. This is structural coverage, not semantic or global exhaustion.',
                 'semantic_review': '31source-level boundary checks:27conditional,3already implemented,1unknown.',
                 'why_not_no_more_fusions': 'Primitive/alias/proposer/metadata graph incomplete;928 boundary combinations do not prove928 possible new kernels.',
                 'overall_task_paused': False, 'continued_to': ['P4', 'P5', 'P6']},
    conditional_candidate_ids=conditional, existing_fusion_ids=existing,
    semantic_unknown_candidate_ids=['F31'], whole_model_exhaustion_proven=False)
write('fusion-scan.json', fusion)
relationships = []
cs = records['fusion_candidates']
for a, b in combinations(cs, 2):
    if a.get('module_ids') == b.get('module_ids') and set(a.get('original_node_ids', [])) & set(b.get('original_node_ids', [])):
        relationships.append({'id': a['id'] + '_' + b['id'], 'from_candidate_id': a['id'], 'to_candidate_id': b['id'],
                              'relation': 'OVERLAPPING_BOUNDARIES_NOT_ADDITIVE',
                              'scope_conditions': 'Same template invocation only. A union requires fresh consumer, precision and capacity review; no savings may be counted twice.',
                              'evidence_ids': [a.get('candidate_graph_path'), b.get('candidate_graph_path')]})
records['candidate_relationships'] = relationships
for c in cs:
    c['case_ids'] = ['FROZEN_CONFIG_SOURCE_BRANCHES']
    c['original_node_mapping_scope'] = 'Local source-template node names, not a complete primitive expansion of G0; candidate maps to named template in its graph file. Full instance alias map remains UNKNOWN.'
    if c['id'] in existing:
        c['new_fusion_opportunity'] = False
        c['status'] = 'OPPORTUNITY'  # Existing package enum; availability and new flag disambiguate.

comparisons = read('evidence/conditional-performance-comparisons.json')['comparisons']
capacity = read('evidence/conditional-nominal-capacity.json')
P8 = Fraction(capacity['capacity_mac_per_second'])
for comp in comparisons:
    obj = comp['object_id']
    rate = P8 * (16 if obj == 'TARGET_SELECTED_INT8_C1' else 1)
    data = {'scope': comp['implementation_domain'], 'time_unit': 'second', 'work_unit': comp['work_unit'],
            'effective_work': str(comp['effective_work_Q']), 'bound_status': 'CONDITIONAL_CEILING',
            'conditions_path': 'evidence/conditional-nominal-capacity.json',
            'terms': [{'id': 'necessary_dense_INT8_MAC', 'kind': 'resource', 'work_lower': str(comp['necessary_MAC']),
                       'rate_upper': str(rate), 'work_unit': 'MAC', 'rate_unit': 'MAC/second',
                       'evidence': 'Conditional nominal capacity and conventional dense algorithm domain; capacity is not independently proven physical maximum'},
                      {'id': 'necessary_HBM', 'kind': 'resource', 'work_lower': None, 'rate_upper': None,
                       'symbolic_expression': 'D_HBM_min(initial_residency, fusion, capacity, reuse)/B_HBM_max'},
                      {'id': 'necessary_interconnect', 'kind': 'resource', 'work_lower': None, 'rate_upper': None,
                       'symbolic_expression': 'D_cut_min(placement, algorithm)/B_cut_max'},
                      {'id': 'unavoidable_causal_chain', 'kind': 'delay', 'time_lower': None,
                       'symbolic_expression': 'Only proven indispensable block/value dependencies; current Python barriers and chosen tiling are not universal'}],
            'layers': [{'id': 'ideal_dense_compute', 'term_ids': ['necessary_dense_INT8_MAC']},
                       {'id': 'partial_required_resources', 'term_ids': ['necessary_dense_INT8_MAC', 'necessary_HBM', 'necessary_interconnect', 'unavoidable_causal_chain']}]}
    inp = 'bounds/' + obj + '.input.json'; output = 'calculations/' + obj + '.json'
    write(inp, data)
    r = subprocess.run([sys.executable, str(K / 'tools/calculate_bounds.py'), str(T / inp)], capture_output=True, text=True, timeout=20)
    assert r.returncode == 0, r.stderr
    result = json.loads(r.stdout)
    last = result['layers'][-1]
    assert Fraction(last['time_lower_from_known_terms']['exact_fraction']) == Fraction(comp['time_lower_L']['exact_fraction'])
    assert Fraction(last['throughput_upper_from_known_terms']['exact_fraction']) == Fraction(comp['throughput_upper_U']['exact_fraction'])
    write(output, result)
    comp.update(bound_input_path=inp, calculation_path=output)
    if obj != 'TARGET_SELECTED_INT8_C1':
        comp['output_contract'] = 'Independent dense kernel with allMoutput rows required; no padding elision or target/draft end-to-end token equivalence is assumed.'
write('bounds-input.json', read('bounds/TARGET_SELECTED_INT8_C1.input.json'))
for m in modules:
    comparisons.append({'object_id': m['module_id'], 'object_kind': 'decoder_layer',
                        'case_id': m['case_id'], 'measured_time_T_seconds': str(Fraction(m['sum_window_ns'], 48 * 10**9)),
                        'timing_scope': 'Mean per-rank inclusive decoder bracket,48calls; not global end-to-end makespan',
                        'effective_work_Q': None, 'time_lower_L': None, 'throughput_upper_U': None,
                        'measured_over_ceiling_ratio': None, 'time_gap_ratio': None, 'bound_status': 'SYMBOLIC_CEILING',
                        'symbolic_model_path': 'evidence/IDEAL-MODEL.md',
                        'utilization_metric_reference': {'name': metric_name, 'value': m['metrics'][0]['value']},
                        'unknown_ids': ['U_VALID_ROUTING_AND_PADDING', 'U_HARDWARE_CAPACITY', 'U_HBM_RESIDENCY', 'U_DEPENDENCY_CAPACITY'],
                        'candidate_only': False})
for cid in conditional:
    comparisons.append({'object_id': cid, 'object_kind': 'unimplemented_fusion_boundary',
                        'candidate_only': True, 'measured_time_T': None, 'time_lower_L': None, 'throughput_upper_U': None,
                        'bound_status': 'INCOMPLETE', 'reason': 'Conditional materialization opportunity is not necessary I/O, a proved fusion implementation, or a measured gain.',
                        'source_ids': ['QR_CONSUMER_REVIEW'] if cid == 'F05' else [], 'unknown_ids': ['U_PRIMITIVE_MAPPING', 'U_HARDWARE_CAPACITY']})

cause = lambda cid, statement, mechanism, support, counter, missing, status: {
    'id': cid, 'module_id': 'target.layers.0.forward', 'observed_graph_id': 'G1_REFINED_MODULE_GRAPH',
    'affected_cases': ['CAPTURED_TARGET_DECODE_INTERIOR_RUNS_1_2_3'], 'observed_symptom': statement,
    'mechanism': mechanism, 'supporting_evidence': support, 'counter_evidence': counter, 'missing_evidence': missing,
    'status': status, 'end_to_end_impact': None, 'end_to_end_impact_limit': 'No causal baseline delta or removable-microsecond claim; profiling overhead unknown.'}
records['module_causes'] = [
    cause('C_ARRIVAL_IMBALANCE', 'Layer0MC2mean942.266us; median of other39layerMC2means303.0448us; MACcounter proxy2.49%.',
          'Earlier arriving ranks spend longer inside fusedMC2 while waiting for peer data/readiness. Current source contains cross-rank readiness polling and cross-core flags.',
          ['mc2-arrival-analysis-v2/summary.json: offset0all400calls overlap, shifts-1/+1none; layer0start-duration correlation~-0.9998',
           'mc2-preceding-analysis/summary.json: interior3pairedcalls already skewed at layer entry; DP0later, DP1earlier; layer-to-MC2elapsed times close',
           'evidence/MC2-known-dependencies.md'],
          ['Explicit internalMC2sequence IDs and inter-device clock error bounds absent; timing consistency is not a causal intervention',
           'Source wait mechanism alone cannot quantify exactly which poll accounts for duration'],
          ['Internal readiness timestamps/sequence IDs', 'Cause of DP0later graph entry', 'Counterfactual unprofiled end-to-end impact'],
          'SUPPORTED_OBSERVATIONAL_MECHANISM_CAUSAL_SHARE_UNPROVEN'),
    cause('C_HBM_OR_SMALL_M', 'Small shape and lowMACproxy coexist with MC2memory traffic.',
          'Small expert matrices, tiles, memory movement, routing load imbalance and current tiling may contribute.',
          ['captured-layer-attribution/operator_metrics.csv', 'evidence/MC2-known-dependencies.md'],
          ['Start spread before layer0and nearly synchronized finish favor an arrival-wait component; lowMACalone cannot identify memory saturation'],
          ['Per-expert valid token counts and active-core distribution', 'HBM/L2physical capacities and measured traffic'], 'HYPOTHESIS'),
    cause('C_OBSERVER_OR_MAPPING', 'Capture-edge calls are longer; model/profiler can perturb scheduling.',
          'Capture boundary, instrumentation and report normalization may change observed durations.',
          ['captured-layer-attribution/interior-robustness.json', 'counter-field-audit/summary.json'],
          ['Layer0remains lowest after runs0andterminalcalls excluded; all16ranks pass independent mapping invariants and PMUfield audit'],
          ['Matched unprofiled per-module timing', 'Clock uncertainty and exact PMUnormalization'], 'HYPOTHESIS')]
records['module_causes'][0]['supporting_evidence'].append('graph-launch-link-inspection/summary.json: exact captured TASKs found for104layer0calls, but no CANNreplay link recovered; capture-definition connection IDs repeat across replays')
records['module_causes'][0]['missing_evidence'].append('No valid captured TASK->per-replay CPUsubmission link; cannot separate host late submission from earlier queued device work')
records['claims'] = [
    {'id': 'CL_LOWEST_CAPTURED', 'type': 'DERIVED', 'statement': 'Layer0is lowest of40source-validated captured target decoder modules under the declared reported-counter metric, both all-capture and interior subsets.', 'source_ids': ['CAPTURED_TARGET_MAPPING', 'COUNTER_FIELD_AUDIT']},
    {'id': 'CL_NOMINAL_CAPACITY', 'type': 'ASSUMED', 'statement': '353.8944TMAC/s per logical device is a conditional nominal ceiling only under applicability and frequency-cap premises.', 'source_ids': ['OFFICIAL_NOMINAL_PEAK']},
    {'id': 'CL_FRAMEWORK_COMPUTE', 'type': 'DERIVED', 'statement': 'Selected necessary target INT8work yields a loose conditional compute-only ceiling; no actual achievable maximum is established.', 'source_ids': ['OFFICIAL_NOMINAL_PEAK']}]
records['models'] = [{'id': 'IDEAL_RELAXED_MODEL', 'scope': contract['implementation_domain'],
    'necessary_work_expressions': 'F8_min, F16_min, Fvec_min, D_HBM_min, D_L2_min, D_link_min, L_causal_min; ideal intermediates not materialized unless proved necessary',
    'capacity_upper_expressions': 'P8_max, P16_max, Pvec_max, B_HBM_max, B_L2_max, B_link_max',
    'real_to_relaxed_coverage_argument': 'Removing scheduling/launch/notification/materialization restrictions only enlarges allowed implementations. Each necessary resource-load lower bound applies to every implementation in the stated algorithm domain. Take max across potentially overlapping resources; sum loads only after shared-resource and nonduplication proof.',
    'derivation_path': 'evidence/IDEAL-MODEL.md'}]
records['constraints'] = [{'id': 'DENSE_INT8_COMPUTE_ONLY', 'scope': 'Declared conventional dense domain and nominal capacity premises',
    'time_lower_expression': 'F8_min/P8_max', 'review_status': 'CONDITIONAL_CEILING',
    'necessity_proof_path': 'evidence/IDEAL-MODEL.md', 'unit': 'second'}]
records['relaxations'] = [{'id': 'IDEAL_FUSION_OVERLAP', 'removed_cost_or_constraint': 'Launch, Python conversion/notification, avoidable intermediate writes and current fixed barriers are relaxed; memory/communication can overlap computation subject only to separately proved limits.', 'why_optimistic': 'Does not exclude a legal implementation and does not force whole-operator serialization.'}]
updates = {
 'U_PRIMITIVE_MAPPING': ('Composite/source-template scan complete structurally; full alias/primitive/proposer/metadata lowering and928semantic reviews incomplete.', ['graphs/G1.json', 'graphs/source-templates.json', 'evidence/source-review-31.json']),
 'U_MODULE_ATTRIBUTION': ('Resolved for40captured target decoder modules across16ranks. Eager prefill, DSpark, root/heads/metadata and outside-decoder Engram remain unattributed.', ['captured-layer-attribution/source-mapping-validation.json', 'counter-field-audit/summary.json']),
 'U_HARDWARE_CAPACITY': ('Official nominal INT8formula located, but physical applicability/frequency cap and memory/link hard capacity limits unproven.', ['sources/primary/msprof-analyze/chip_peak_flops.py', 'evidence/profile-device-capacity-info.json']),
 'U_PROFILE_BOUNDARY': ('Interior subset preserves lowest identity; observer effect and internal cross-rank operation-ID/clock proof remain unknown.', ['captured-layer-attribution/interior-robustness.json', 'mc2-arrival-analysis-v2/summary.json'])}
for u in records['unknowns']:
    statement, attempted = updates[u['id']]; u.update(statement=statement, attempted_sources=attempted, failure_reason='Explicit unresolved evidence, not a zero-cost assertion')
for uid, statement in [('U_VALID_ROUTING_AND_PADDING', 'Per-layer valid unpadded expert assignments are not recovered; padded physical rows are not returned tokens.'),
                       ('U_HBM_RESIDENCY', '40MiBWq_b fits within192MiBL2in size; no forced cold HBMread each invocation is established.'),
                       ('U_DEPENDENCY_CAPACITY', 'Source ordering does not establish shortest physical block dependency latency or unavoidable full-stage barriers.'),
                       ('U_CROSS_RANK_CLOCK', 'No independent bound on cross-device timestamp error or internalMC2operation-IDmapping.')]:
    if not any(u['id'] == uid for u in records['unknowns']):
        records['unknowns'].append({'id': uid, 'statement': statement, 'next_entry': 'Read existing lower-level trace/SDK/source if it can resolve this; no invented rates or timings.'})
records['reviews'] = [{'id': 'REVIEW_CHECKPOINT_' + now, 'reviewer': 'GPT primary controller', 'review_type': 'SOURCE_AND_EVIDENCE_REVIEW_NOT_FORMAL_PROOF',
    'status': 'PARTIAL_WITH_EXPLICIT_CONDITIONS', 'reason': 'All-capture/interior module identity agrees; exact arithmetic separately reproduced; physical ceiling and global primitive coverage remain unproven.',
    'date': now}]
write('records.json', records)

analysis = read('module-analysis.json')
analysis.update(status='PARTIAL', modules=modules,
    source_version=contract['graph_scope']['source_version'], trace_source_ids=['RETAINED_16RANK_KERNEL_EXPORTS'],
    module_kernel_mapping_path='captured-layer-attribution/source-mapping-validation.json',
    rankings=[{'case_ids': ['CAPTURED_TARGET_DECODE_INTERIOR_RUNS_1_2_3'], 'metric_name': metric_name,
               'comparison_contract': metric_definition, 'ordered_module_groups': [[m['module_id']] for m in ordered],
               'lowest_module_ids': ['target.layers.0.forward'], 'coverage_statement': '40of40captured target decoder modules; not entire inference framework',
               'status': 'VALID_FOR_STATED_REPORTED_COUNTER_METRIC', 'global_full_framework_minimum_known': False}],
    coverage={'measured_module_ids': [m['module_id'] for m in modules], 'target_decoder_modules_measured': 40,
              'missing_module_ids': contract['measurement_scope_refinement']['missing_scope'],
              'global_full_framework_coverage_fraction': None, 'complete_model_claimed': False},
    cause_ids=[r['id'] for r in records['module_causes']], performance_comparisons=comparisons)
write('module-analysis.json', analysis)

profile = read('profile-summary.json')
profile.update(status='CAPTURED_TARGET_SUBSET_DIAGNOSED', observed_graph_id='G1_REFINED_MODULE_GRAPH',
    module_attribution_status='VALIDATED_FOR_CAPTURED_TARGET_ONLY', total_module_count=None, comparable_module_count=40,
    metric_definitions=[{'id': metric_name, 'definition': metric_definition}],
    lowest_utilization_groups=[{'module': m['module_id'], 'counter_proxy': m['metrics'][0]['value'], 'calls': m['call_count'], 'mean_window_us': m['mean_window_us']} for m in ordered[:5]],
    long_wait_or_idle_groups=[{'id': 'C_ARRIVAL_IMBALANCE', 'state': 'OBSERVATIONAL_SUPPORT_NOT_QUANTIFIED_CAUSAL_TIME', 'evidence': 'mc2-preceding-analysis/summary.json'}],
    full_aggregate_table_refs=['evidence/TRACE-SUMMARY.json', 'captured-layer-attribution/module_metrics.csv', 'captured-layer-attribution/module-call-metrics.json', 'captured-layer-attribution/operator_metrics.csv'],
    scope_and_coverage={'rank_count': 16, 'kernel_csv_rows': 481548, 'captured_calls_all': 4160, 'interior_calls': 1920, 'target_decoder_modules': 40, 'full_framework_coverage': 'PARTIAL'})
assert len(json.dumps(profile).encode()) < 32768
write('profile-summary.json', profile)
index = read('profile-index.json')
index['status'] = 'CAPTURED_TARGET_AGGREGATES_AND_TIMELINE_INDEXED'
index['processing_steps'] = [str(p.relative_to(T)) for p in sorted((T / 'tools').glob('*.py')) if p.name not in ['integrate_diagnosis.py']]
for n, count, keys in [('captured-layer-attribution/module_metrics.csv', 640, ['rank', 'layer']),
                       ('captured-layer-attribution/module-call-metrics.json', 4160, ['rank', 'model', 'run', 'layer']),
                       ('captured-layer-attribution/operator_metrics.csv', None, ['bucket', 'layer', 'task_type']),
                       ('mc2-arrival-analysis-v2/rank_target_calls.csv', 6720, ['rank', 'target_call_ordinal'])]:
    if not any(x.get('path_or_table') == n for x in index['aggregate_tables']):
        index['aggregate_tables'].append({'id': Path(n).stem.upper(), 'path_or_table': n, 'row_count': count, 'grouping_keys': keys,
                                          'sha256': sha(T / n), 'full_scope_coverage': 'Declared captured/target subset only'})
index['coverage'].update(module_ids=[m['module_id'] for m in modules], missing_items=contract['measurement_scope_refinement']['missing_scope'])
index['evidence_slices'] = [{'id': 'LAYER0_PRECEDING', 'path': 'mc2-preceding-analysis/bounded_local_slices.csv',
                            'read_policy': 'Filter rank and forward ordinal; at most200rows/32KiBper model view; full retained slice table is not discarded.'},
                           {'id': 'COUNTER_AUDIT', 'path': 'counter-field-audit/summary.json'}]
write('profile-index.json', index)

for p in sorted(Q.glob('operator-ceiling169-v2-*.verified.json')):
    for suffix in ['.verified.json', '.exit-code', '.task.txt', '.local.sh', '.output.txt', '.evidence.txt']:
        src = p.with_name(p.name.removesuffix('.verified.json') + suffix)
        if src.exists():
            shutil.copyfile(src, T / 'zcode' / src.name)
receipts = []
for p in sorted((T / 'zcode').glob('*.verified.json')):
    d = json.loads(p.read_text())
    assert d['model_verified'] and all('deepseek' in s.lower() for s in d['observed_models'])
    exitp = p.with_name(p.name.removesuffix('.verified.json') + '.exit-code')
    receipts.append({'stage': p.name.removesuffix('.verified.json'), 'session_id': d['session_id'],
                     'observed_models': d['observed_models'], 'exit_code': int(exitp.read_text()) if exitp.exists() else None,
                     'verification_receipt_sha256': sha(p)})
write('evidence/ZCODE-EXECUTION-INDEX.json', {'stages': receipts, 'controller': 'GPT', 'mechanical_executor': 'Local Mac ZCode DeepSeek',
    'failure_kept': 'v2-007equal-count guard rejected400vs440; v2-008preserved counts and analyzed candidate shifts; no forced match.'})

state = read('STATE.json')
state.update(state='ACTIVE', current_phase='P6', last_update=now, fusion_scan_status='PAUSED_WITH_GAPS',
    fusion_stop_reason=fusion['stop_reason'], module_diagnosis_status='PARTIAL_CAPTURED_TARGET_DIAGNOSED',
    performance_comparison_status='7CONDITIONAL_NUMERIC_40SYMBOLIC_MODULE_27UNIMPLEMENTED_CANDIDATE_RECORDS',
    goal_tool_state='blocked', goal_tool_reason=None, user_continuation_authorized=True,
    overall_work_paused=False, next_read_only_action='Review graph-launch connection evidence, finalize diagnosis report and sync all records to authorized169main directory.')
write('STATE.json', state)
numeric = [r for r in comparisons if r.get('bound_status') == 'CONDITIONAL_CEILING']
table = '\n'.join('| ' + r['object_id'] + ' | ' + str(round(float(Fraction(r['measured_time_T']['exact_fraction'])) * 10**6, 6)) + ' | ' + str(round(float(Fraction(r['time_lower_L']['exact_fraction'])) * 10**6, 6)) + ' | ' + str(round(float(Fraction(r['time_gap_ratio']['exact_fraction'])), 3)) + ' |' for r in numeric if r['object_id'] != 'TARGET_SELECTED_INT8_C1')
low_table = '\n'.join(f"| 第{m['layer']}层 | {100*m['metrics'][0]['value']:.4f}% | {m['mean_window_us']:.3f} | 48 |" for m in ordered[:5])
report = f'''# 169 vLLM-Ascend：融合扫描、低活动模块和条件理论界

更新：{now}。结论范围是已有证据的发现与定位；没有修改模型、框架或 kernel，没有实施融合或调参。本轮仅 CPU 离线分析既有 16 rank trace，GPT 主控，本机 ZCode DeepSeek 执行机械任务。

## 当前发现

已验证归属的 40 个 target 解码层中，第 0 层的 CSV 报告 AIC 周期加权 MAC 活动指标最低。去掉采样边缘后为 **3.6402%**；全采样口径为 3.0315%。该指标是报告字段构成的代理量，**不是整卡利用率、MFU 或相对物理峰值的算力效率**。

第 0 层内部 MC2 的 48 次调用平均 942.266 微秒，其他 39 层对应均值的中位数为 303.045 微秒。现有时间线支持跨 DP 到达不齐、早到 rank 在 MC2 中等待的机制；尚未证明其可消除时间或端到端影响。不能据此认定底层 INT8 计算慢。

## 基线与数据范围

无 profiler 基线保持为 32K 输入、1K 输出、并发 8、实际前缀命中 89.990234375%；两轮聚合 **519.8440965 输出 token/s**。基线与有 profiler 的短样本分开，不用采样耗时替代基线。源版本、镜像、数值语义见 contract.json；原始导出 CSV 哈希见 evidence/TRACE-EXPORT-SHA256.json，原始大 trace 保留在 169 本任务原目录。

全量聚合覆盖 16 rank、481548 行 kernel CSV。通过 mHC 次序、每层 1 MC2/1 SparseFlashMla、压缩 KV 与 index source 角色等独立不变量，从捕获图恢复 4160 次 target decoder 调用；16 rank 全部通过。主排名使用每 rank/model 的 runs1、2、3，共 1920 次调用、每层 48 次，先固定采样规则再比较。全采样与内部样本的最低层身份一致。

| 已测模块 | MAC 报告代理指标 | 平均模块窗口/微秒 | 样本 |
|---|---:|---:|---:|
{low_table}

分子为 `sum(aic_total_cycles*aic_mac_ratio)`，分母为 `sum(aic_total_cycles)`，仅纳入已归属 kernel；不是百分比简单平均。PMU 字段审计未发现正周期但比例缺失/非法的行。缺周期的任务没有被当成具有已知 PMU 的零活动任务。完整分子/分母、各 rank 结果和全部 40 层排序在 module-analysis.json、captured-layer-attribution/ 中。

覆盖缺口：eager prefill、3 个 eager DSpark block、decoder 外的 Engram CPU lookup/投影门控、root/head/metadata 未形成同口径模块排名。这里只能称“已测 40 个 target decoder 中最低”，不能称整个推理框架最低。模块窗口的跨 rank 平均值不是全局关键路径时间。

## 融合扫描：做了什么，停在哪里

冻结 G0，G1 补齐延后消费的 mHC 系数跨层依赖。12 个源码模板、221 个模板节点，扫描两两、三个、四个相连节点、共享输入分支以及已知复合阶段的拆分重组/跨模块边界，枚举 928 个结构项；第二轮同图同规则新增 0。928 是结构项数量，不是已证明可实现或能提速的融合数量。

对 31 个源码边界检查项复核：27 个有条件候选，3 个已实现的融合（mHC pre、RMSNormCast、SparseFlashMla），1 个 metadata 路径仍未知。候选图在 candidates/，原始检查在 evidence/FUSION-SWEEP.md，覆盖/停止在 fusion-scan.json，重叠关系在 records.json。已实现融合不重复算收益；相互重叠的候选不能把节省量相加。

具体例子 F05：Q 归一化输出 qr 在 8 个 index source 层还供给 indexer，融合 norm+quant 不能直接删除 qr 写出。其余 32 层没有该数值消费者时，qr 的一写一读最多为 5120*M 字节的有条件移除机会。BF16 舍入、scale 归约、两条分支及原有多流重叠必须保留。详见 evidence/QR-consumers-and-fusion-review.md。

当前融合阶段标记 `PAUSED_WITH_GAPS`，指完整 primitive/alias/proposer/metadata 图和语义证明仍有缺口；**不表示整个任务暂停，也不表示已经证明再无可融合项**。不能把有限结构复扫包装成全模型融合穷尽。

## 理想上界与实际对照

先允许理想融合和重叠，再用必要工作/容量约束：`T>=max(F8/P8, F16/P16, D_HBM/B_HBM, D_link/B_link, L_causal,...)`，`U=Q/L`。共享资源只在不重复计数时累计工作；当前同步、Python 转换和所选 tile 的开销不自动进入不可避免下界。详见 evidence/IDEAL-MODEL.md。

官方标称公式与实际设备信息给出条件分母 353.8944 TMAC/s/逻辑设备；9382 适用性、1800 MHz 频率上限及标称公式作为容量上限仍是前提。以下独立 Wq_b 是所有 M 行输出需保留的标准 dense 算法域，M 不等于返回 token 数。

| 对象 | 实测中位数/微秒 | 条件计算下界/微秒 | T/L（非可达提速） |
|---|---:|---:|---:|
{table}

对选定 target INT8 必要工作，用 16 设备容量得到的 **116294.745 输出 token/s** 只是极松的条件计算上界：忽略 attention、BF16/FP32、draft/reject、访存、通信和调度。不能把它当“本机实际极限预测”，不能声称基线存在 223 倍可提速空间。

本次已保存 7 个独立数值对象的精确输入与计算器原始输出；40 层模块和 27 个未实现候选也有对照记录，缺失项保留符号/null。HBM/L2/互连容量上限、必要字节和有效路由计数未证，**本机无条件、不可超越的数值吞吐上界仍未知**。没有用最快实测速率代替硬上限。

## 第 0 层为什么低：支持到哪里

MC2 源码已融合 dispatch、GMM1、SwiGLU/量化、GMM2、combine，并有逐 expert/跨核心 flag 与跨 rank readiness 轮询；它不是一个纯 INT8 GEMM。源码见 evidence/MC2-known-dependencies.md。

DP0 记录 400 次 target MC2，DP1 记录 440 次。首次等次数校验主动拒绝配对，失败记录保留；后续保留不等次数并检验相对偏移 -1/0/+1 轮。offset0 的 400 组调用均有 16 rank 公共执行区间，错开一轮均没有；这支持该配对，但没有内部 operation ID 与时钟误差上限，仍作为有条件的时间线关联。

在 3 次两组 DP 都处于内部采样窗口的第 0 层调用中，层入口的开始时间差中位数 1347.468 微秒，MC2 开始时间差中位数 1344.596 微秒，MC2 结束时间差中位数 32.524 微秒。每次都是 DP0 晚进入该层，DP1 早进入；各组层入口到 MC2 的耗时约 377–395 微秒，差异小得多。时间差在层入口前已存在，不能归咎于本层 attention 本身多算了同等时间。

这些证据与早到 rank 在 MC2 等对端一致。当前 TASK 连接到的是重复的捕获定义，104 次第 0 层精确 TASK 查找均成功，却未找到对应的每次 replay CANN 提交连接；不能靠 CPU 时间窗包含关系补造归属。因此，DP0 为什么更晚进入图——host 提交、前序设备队列、其他工作或 profiler 扰动——仍未区分。

小 M/专家负载不均/访存与当前分块策略可同时贡献；未拿到专家有效 token 分布、内部发布时间与实际容量，不能精确分摊。去掉边缘后异常仍在，削弱“纯采样首尾效应”的解释，但不排除 profiler 对调度的扰动。基线端到端可消除时间为 UNKNOWN。

## 复核与恢复

先读 STATE.json、contract.json、profile-summary.json 和本报告；再按索引读取最多 200 行/32 KiB 的局部证据。完整聚合保留，不把整个大 trace 塞入模型。tools/ 保存 CPU 分析脚本，bounds/ 与 calculations/ 保存精确数值复算；calculator 仅检查算术，前提由主控审阅。ZCode 模型验证和执行失败/成功凭据在 zcode/ 与 evidence/ZCODE-EXECUTION-INDEX.json。

本轮分析记录同步到 `/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028`。不操作 172，不修改目标框架/模型，不发送新推理负载。

目标工具仍返回无原因的 blocked 且时间戳未更新；用户已经授权恢复，任务文件保持 ACTIVE。该工具状态与本轮实际执行进度分开记录，不能声称自动续跑已恢复。
'''
(T / 'REPORT.md').write_text(report)
print(json.dumps({'state': state['state'], 'phase': state['current_phase'], 'module_count': len(modules), 'lowest': ordered[0]['module_id'],
                  'lowest_reported_counter_proxy': ordered[0]['metrics'][0]['value'], 'performance_comparison_records': len(comparisons),
                  'numeric_calculations': 7, 'fusion_source_checks': 31, 'new_fusion_implementations': 0, 'zcode_receipts': len(receipts)}, indent=2))
