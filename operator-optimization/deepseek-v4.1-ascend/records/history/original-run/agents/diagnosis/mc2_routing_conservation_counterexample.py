#!/usr/bin/env python3
"""Conditional conservation test; synthetic routes are NOT historical samples."""
import json
from pathlib import Path

root = Path(__file__).resolve().parents[2]
data = json.loads((root / 'agents/diagnosis/mc2-l14-l39-matched-tasks.json').read_text())
mc2 = [x for x in data['results'] if x['name'] == 'DispatchFFNCombine']
assert len(mc2) == 1
pairs = mc2[0]['pairs']
assert sorted(x['ep'] for x in pairs) == list(range(8,16))
q = min(x['ratio'] for x in pairs)
assert q > 1

# 160 synthetic tokens, top-6, 24 distinct experts per rank, 16 ranks.
# Each token selects 6 distinct experts on ONE rank; top-k need not be spread.
# L39 sends 8 tokens to target ranks, 152 tokens to the rest, balanced within each group.
# L14 sends all 160 tokens to target ranks, evenly (20 tokens per rank).
routes = {'L14': [], 'L39': []}
for i in range(160):
    rank14 = 8 + i % 8
    rank39 = 8 + i if i < 8 else (i - 8) % 8
    for layer, rank in [('L14', rank14), ('L39', rank39)]:
        experts = [rank * 24 + j for j in range(6)]
        assert len(experts) == len(set(experts)) == 6
        assert all(0 <= e < 384 for e in experts)
        routes[layer].append(experts)

counts = {}
for layer, route in routes.items():
    counts[layer] = [0] * 16
    for experts in route:
        for expert in experts:
            counts[layer][expert // 24] += 1
    assert sum(counts[layer]) == 160 * 6
    assert max(counts[layer]) <= 120  # illustrative cap, no truncation
assert counts['L39'] == [114]*8 + [6]*8
assert counts['L14'] == [0]*8 + [120]*8
assert all(counts['L14'][i] / counts['L39'][i] == 20 for i in range(8, 16))
# An equal non-negative additive per-rank intercept compresses the ratio.
for intercept in (0, 1, 100):
    assert all((counts['L14'][i]+intercept)/(counts['L39'][i]+intercept) <= 20 for i in range(8,16))
result = {
    'observed_min_selected_duration_ratio': q,
    'conditional_short_layer_selected_count_share_upper_bound': 1/q,
    'conditional_basis': 'same total R in both layers; each selected rank has same nonnegative affine time model in both layers',
    'synthetic_only': {'tokens': 160, 'topk': 6, 'experts_per_rank': 24, 'rank_count': 16,
                       'L14_rank_counts': counts['L14'], 'L39_rank_counts': counts['L39'],
                       'L14_selected_count': sum(counts['L14'][8:]),
                       'L39_selected_count': sum(counts['L39'][8:]),
                       'total_per_layer': 960, 'target_ratio': 20,
                       'target_short_share': sum(counts['L39'][8:])/960},
    'not_observed': ['actual expert counts', 'source group membership', 'active mask', 'topk truncation', 'capacity', 'duration affine model'],
}
print(json.dumps(result, ensure_ascii=False, indent=2))
