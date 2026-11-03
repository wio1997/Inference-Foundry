#!/usr/bin/env python3
"""Check arithmetic consistency of saved PMU ratios, not exporter binding."""
import hashlib, json, pathlib
p = pathlib.Path(__file__).with_name('prepare-pmu-32-selected.json')
b = p.read_bytes()
samples = json.loads(b)['samples']
assert len(samples) == 32
fields = ('aiv_scalar_ratio', 'aiv_vec_ratio', 'aiv_mte2_ratio', 'aiv_mte3_ratio')
result = {'input_sha256': hashlib.sha256(b).hexdigest(), 'samples': len(samples), 'interpretation': 'ratio*cycles integer consistency only; no measured numerator or provenance binding', 'metrics': {}}
for name in fields:
    products = [x['pmu'][name] * x['pmu']['aiv_total_cycles'] for x in samples]
    result['metrics'][name] = {'max_abs_residual_to_integer': max(abs(v-round(v)) for v in products), 'inferred_integer_min': min(round(v) for v in products), 'inferred_integer_max': max(round(v) for v in products)}
x = samples[0]
result['example'] = {'ep': x['ep'], 'layer': x['layer'], 'scalar_ratio': x['pmu']['aiv_scalar_ratio'], 'total_cycles': x['pmu']['aiv_total_cycles'], 'inferred_product': x['pmu']['aiv_scalar_ratio'] * x['pmu']['aiv_total_cycles']}
print(json.dumps(result, indent=2, ensure_ascii=False))
