#!/usr/bin/env python3
"""Logical counterexample, not a reconstruction of the historical run."""
import json

capacity = 48
vectors = {'concentrated': [48] + [0] * 23, 'distributed': [2] * 24}

def trim(counts, cap):
    used = 0
    result = []
    for count in counts:
        keep = min(count, max(0, cap - used))
        result.append(keep)
        used += keep
    return result

out = {}
for name, counts in vectors.items():
    effective = trim(counts, capacity)
    out[name] = {'input_sum': sum(counts), 'effective_sum': sum(effective),
                 'nonzero_experts': sum(x > 0 for x in effective),
                 'effective_vector': effective}
assert all(x['effective_sum'] == 48 for x in out.values())
assert [out[x]['nonzero_experts'] for x in vectors] == [1, 24]
print(json.dumps({'capacity_assumption': capacity, 'cases': out}, indent=2))
