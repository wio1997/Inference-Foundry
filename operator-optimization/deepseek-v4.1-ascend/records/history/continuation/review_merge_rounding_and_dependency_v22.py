"""Finite arithmetic/DAG review; stdlib only, no torch, model, or device."""
from pathlib import Path
from datetime import datetime, timezone
import json
import struct

OUT = Path(__file__).parent / 'profile-first-20261031/merge-rounding-and-dependency-review-v22.json'

def f32(x):
    return struct.unpack('>f', struct.pack('>f', x))[0]

def bf16(x):
    # Explicit finite FP32 -> BF16 round-to-nearest, ties-to-even model.
    u = struct.unpack('>I', struct.pack('>f', f32(x)))[0]
    assert u & 0x7f800000 != 0x7f800000
    u = (u + 0x7fff + ((u >> 16) & 1)) & 0xffff0000
    return struct.unpack('>f', struct.pack('>I', u))[0]

def cyclic(edges):
    nodes = {n for e in edges for n in e}
    adj = {n: [] for n in nodes}
    for a, b in edges:
        adj[a].append(b)
    state = {}
    def visit(n):
        if state.get(n) == 1:
            return True
        if state.get(n) == 2:
            return False
        state[n] = 1
        if any(visit(k) for k in adj[n]):
            return True
        state[n] = 2
        return False
    return any(visit(n) for n in nodes)

def main():
    assert not OUT.exists(), 'refuse overwrite'
    s, r, p = 1.0, 3 / 1024, 1.5
    assert bf16(s) == s and bf16(r) == r
    x = bf16(f32(s + r))
    original_a = bf16(f32(x * p))
    bypass_a = bf16(f32(f32(s + r) * p))
    assert x == 1.0 and original_a == 1.5 and bypass_a == 1.5078125
    s2, r2, gamma = -2.0, 171 / 128, 1.5
    assert bf16(s2) == s2 and bf16(r2) == r2
    product = f32(r2 * gamma)
    scaled_r = bf16(product)
    original_b = bf16(f32(s2 + scaled_r))
    bypass_b = bf16(f32(s2 + product))
    assert product == 2.00390625 and scaled_r == 2.0
    assert original_b == 0.0 and bypass_b == 0.00390625
    order = ['MC2_COMPLETE', 'SHARED_DEQUANT', 'SHARED_GATE_UP',
             'SHARED_ACT_QUANT', 'SHARED_DOWN', 'MERGE', 'HC_POST']
    baseline = list(zip(order[:-1], order[1:]))
    assert not cyclic(baseline)
    upstream_only = baseline + [('SHARED_DOWN', 'MC2_COMPLETE')]
    assert cyclic(upstream_only)
    result = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        scope='Finite arithmetic examples and conditional DAG analysis only.',
        numeric_model='Finite FP32 arithmetic rounded after each listed operation; BF16 RNE storage boundaries. No claim that native HC uses this exact precision/order.',
        counterexamples=[dict(
            id='N22-A', proposition='Unconditionally delete the BF16 merge-result rounding before subsequent FP32 coefficient multiplication.',
            inputs=dict(shared=s, postscaled_routed=r, post=p, other_HC_contribution=0),
            merge_before_bf16=f32(s+r), merge_bf16=x,
            original_output=original_a, rewritten_output=bypass_a,
            conclusion='Identity fails in the explicitly stated finite arithmetic model. The BF16 merge boundary must be preserved unless the accepted numerical contract allows a change.',
            limit='Not an observed frozen-model input or validation of the loaded native HC binary.'), dict(
            id='N22-B', proposition='Replace a separate BF16 routed scaling followed by plain addition with a single higher-precision scaled addition.',
            inputs=dict(shared=s2, routed=r2, conditional_gamma=gamma),
            product_before_bf16=product, scaled_routed_bf16=scaled_r,
            original_output=original_b, rewritten_output=bypass_b,
            conclusion='Identity fails in the explicitly stated finite arithmetic model.',
            limit='Conditional source branch only; does not claim the captured runner effective gamma equals 1.5 or differs from 1.')],
        graph_review=dict(
            baseline_edges=baseline, baseline_acyclic=True,
            proposed_extra_edge=['SHARED_DOWN', 'MC2_COMPLETE'], proposed_graph_cyclic=True,
            excluded_domain='Move the shared+routed merge only into the upstream MC2 epilogue, make MC2 completion wait for SharedDown, and retain all shown baseline order edges.',
            conclusion='This epilogue-only change creates a dependency cycle in the stated domain.',
            limit='Not a baseline deadlock, not a theorem excluding a redesigned kernel, changed producer order, or a different stream schedule. Hardware feasibility/gain is not proved.'),
        no_model_import_or_device_or_service_work=True,
        overall_complete=False)
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(output=str(OUT), arithmetic_examples=2,
                          baseline_acyclic=True, conditional_epilogue_graph_cyclic=True,
                          overall_complete=False)))

if __name__ == '__main__':
    main()
