"""Finite sequence evidence only; no blanket rounding/quality conclusion."""
from pathlib import Path
import hashlib, json

r = Path(__file__).resolve().parent
outcome = json.loads((r/'prefix-count-correctness-outcome-v1.json').read_text())
assert outcome['completed'] and outcome['all16_paths'] and outcome['actual_prefix_GMM1_and_count_GMM2']
p = next((r/'prefix-count-quality-v1').glob('*/metrics.json'))
quality = json.loads(p.read_text())
assert quality['completed'] and quality['numeric_all_pass'] and len(quality['rows']) == 52
assert len(quality['sequence_comparisons']) == 26 and all(z['functional_pass'] and z['numeric_pass'] and len(z['token_ids']) == 64 for z in quality['rows'])
unequal = {k for k, v in quality['sequence_comparisons'].items() if not v['all_same']}
assert unequal == set(outcome['unequal_full64_cases'])
review = dict(proceed_finite_formal=False, full64_equal_cases=26-len(unequal), unequal_cases=sorted(unequal), quality_metrics_sha256=hashlib.sha256(p.read_bytes()).hexdigest(), scope='Only finite numeric/full64 paired evidence and bounded same-instance controls. No general model QA, harmless rounding or CANN cause claim.')
if not unequal:
    review['proceed_finite_formal'] = True
else:
    files = list((r/'prefix-count-quality-control-v1').glob('*/metrics.json'))
    if not files:
        print(json.dumps(dict(review, next='Run one bounded native-A/replica/candidate/native-B control for these actual unequal cases, then primary review.')))
        raise SystemExit(2)
    assert len(files) == 1
    controls = json.loads(files[0].read_text())
    assert controls['completed'] and controls['candidate_and_diagnostic_markers_removed']
    assert len(controls['rows']) == 4*len(unequal) and {z['case_id'] for z in controls['rows']} == unequal
    assert all(z['numeric_pass'] and z['full_budget_pass'] and len(z['token_ids']) == 64 for z in controls['rows'])
    comparisons = []
    for case in sorted(unequal):
        rows = [z for z in controls['rows'] if z['case_id'] == case]
        assert sorted(z['mode'] for z in rows) == ['native-A', 'native-B', 'native-replica', 'target-prefix-count']
        native = {tuple(z['token_ids']) for z in rows if z['mode'] != 'target-prefix-count'}
        initial = next(z['token_ids'] for z in quality['rows'] if z['case_id'] == case and z['mode'] == 'adaptive')
        repeat = next(z['token_ids'] for z in rows if z['mode'] == 'target-prefix-count')
        comparisons.append(dict(case=case, original_candidate_in_native_set=tuple(initial) in native, current_candidate_in_native_set=tuple(repeat) in native, observed_native_sequences=len(native)))
    review.update(controls=comparisons, control_rows=len(controls['rows']), control_metrics_sha256=hashlib.sha256(files[0].read_bytes()).hexdigest())
    review['proceed_finite_formal'] = all(z['original_candidate_in_native_set'] and z['current_candidate_in_native_set'] for z in comparisons)
path = r/'prefix-count-quality-reviewed-v1.json'; assert not path.exists()
path.write_text(json.dumps(review, indent=2)+'\n'); print(json.dumps(review, indent=2))
