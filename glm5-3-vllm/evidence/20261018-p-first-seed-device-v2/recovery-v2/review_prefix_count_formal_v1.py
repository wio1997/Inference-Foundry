from pathlib import Path
import hashlib, json, subprocess, sys

r = Path(__file__).resolve().parent
subprocess.run([sys.executable, str(r/'analyze_prefix_count_formal_v1.py')], check=True, capture_output=True, text=True)
x = json.loads((r/'prefix-count-formal-verdict-v1.json').read_text())
c = r/'prefix-count-formal-contract-v1.json'; contract = json.loads(c.read_text())
quality = json.loads((r/'prefix-count-quality-reviewed-v1.json').read_text())
assert contract['formal_design']['C4_formal_requests'] == 384 and quality['proceed_finite_formal']
ratios = {k: x['C4_pooled']['adaptive'][k]['p95_linear']/x['C4_pooled']['native'][k]['p95_linear'] for k in ['TTFT_ms', 'last_token_ms', 'mean_TPOT_ms']}
gates = dict(C4_aggregate_throughput_improved=x['C4_candidate_over_native'] > 1,
             all3_C4_cycles_improved=all(z['candidate_over_native'] > 1 for z in x['C4_cycles']),
             C4_all_pooled_p95_nonregressing=all(z <= 1 for z in ratios.values()),
             C1_TTFT_and_TPOT_nonregressing=all(x['C1_equal_case_ratios'][k] <= 1 for k in ['TTFT_ms', 'mean_TPOT_ms']),
             cancel_and_peer_reuse_complete=x['cancel']['completed'] and x['cancel']['client_close_requests'] == 10 and x['cancel']['full_reuse_and_peer_responses'] == 16)
assert gates['cancel_and_peer_reuse_complete']
x['decision'] = 'ACCEPT_FINITE_JOINT_POINT_PREFIX_COUNT' if all(gates.values()) else 'MIXED_DEFAULT_OFF_PREFIX_COUNT_NO_JOINT_GAIN' if x['C4_candidate_over_native'] > 1 else 'REJECT_DEFAULT_OFF_PREFIX_COUNT'
x.update(frozen_contract_sha256=hashlib.sha256(c.read_bytes()).hexdigest(), predeclared_joint_gates=gates,
         C4_p95_candidate_over_native=ratios, finite_quality_review=quality,
         adoption_scope='Only this finite same-load point if all gates pass. Default helper env remains off. No generic quality/capacity, global optimum, single-cause or cross-window combined-gain claim.')
(r/'prefix-count-reviewed-verdict-v1.json').write_text(json.dumps(x, indent=2)+'\n')
print(json.dumps({k: x[k] for k in ['decision', 'predeclared_joint_gates', 'C1_equal_case_ratios', 'C4_candidate_over_native', 'C4_cycles', 'C4_p95_candidate_over_native', 'cancel']}, indent=2))
