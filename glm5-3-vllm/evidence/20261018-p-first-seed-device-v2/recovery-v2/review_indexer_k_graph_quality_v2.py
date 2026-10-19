"""Preserve the frozen full64 gate and execute only necessary finite controls."""
from pathlib import Path
import datetime
import hashlib
import importlib.util
import json

r=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('transport',r/'run_prefix_count_service_v1.py')
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
outcome=json.loads((r/'indexer-k-graph-correctness-outcome-v2.json').read_text())
assert outcome['completed'] and outcome['all16_actual_paths']
quality_path=next((r/'indexer-k-graph-quality-v2').glob('*/metrics.json'))
quality=json.loads(quality_path.read_text())
assert quality['completed'] and quality['numeric_all_pass']
unequal=[k for k,v in quality['sequence_comparisons'].items() if not v['all_same']]
control=None
control_path=None
if unequal:
    control=m.collect(r/'control_indexer_k_graph_quality_v2.py','indexer_k_graph_quality_controls_v2','indexer-k-graph-quality-controls-v2',timeout=650)
    assert control['completed'] and all(z['numeric_pass'] for z in control['rows'])
    assert set(control['membership'])==set(unequal)
    control_path=next((r/'indexer-k-graph-quality-controls-v2').glob('*/metrics.json'))
proceed=not unequal or all(v['both_initial_and_repeat_candidate_in_observed_native_set'] for v in control['membership'].values())
review=dict(reviewed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            proceed_finite_hardware_and_formal_review=proceed,formal_executed=False,
            initial_full64_equal_cases=26-len(unequal),cases=26,unequal_cases=unequal,
            finite_controls=control['membership'] if control else None,
            quality_source=str(quality_path.relative_to(r)),quality_sha256=hashlib.sha256(quality_path.read_bytes()).hexdigest(),
            control_source=str(control_path.relative_to(r)) if control_path else None,
            control_sha256=hashlib.sha256(control_path.read_bytes()).hexdigest() if control_path else None,
            P_CID=outcome['P_CID'],D_CID=outcome['D_CID'],
            scope='Same loaded P/D for initial and current native controls. Initial and repeat candidate variants must both belong to observed native full64 sets. Finite quality gate only, no harmless-rounding, general QA or service gain claim. All16 actual copy SDMA proof and primary review still precede formal.')
(r/'indexer-k-graph-quality-reviewed-v2.json').write_text(json.dumps(review,indent=2)+'\n')
print(json.dumps(review),flush=True)
