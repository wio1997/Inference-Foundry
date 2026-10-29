#!/usr/bin/env python3
"""Check published summary arithmetic; this does not reproduce measurements."""
import json
from pathlib import Path

summary = json.loads((Path(__file__).parent / 'evidence-summary.json').read_text())
baseline = summary['formal_baseline']
rounds = baseline['rounds']
assert len(rounds) == 2
for row in rounds:
    assert row['requests'] == 8
    assert row['input_tokens'] == 8 * baseline['input_tokens_per_request']
    assert row['output_tokens'] == 8 * baseline['output_tokens_per_request']
    assert row['prefix_queries'] == 262144 and row['prefix_hits'] == 235904
    assert row['preemptions'] == 0
    assert abs(row['output_tps'] - row['output_tokens'] / row['duration_seconds']) < 1e-9
aggregate = sum(x['output_tokens'] for x in rounds) / sum(x['duration_seconds'] for x in rounds)
assert abs(aggregate - baseline['aggregate_output_tps']) < 1e-9
profile = summary['profiled_sample']
assert profile['target_scope_count'] == 1280
assert profile['aic_valid_groups'] <= profile['complete_8rank_groups'] <= profile['physical_signature_groups']
f08 = profile['f08_by_reported_rows']
assert sum(x['scope_count'] for x in f08.values()) == 1280
assert sum(x['scatter_longest_count'] for x in f08.values()) == 1207
assert all(0 <= x['scatter_fraction_median'] <= 1 for x in f08.values())
print(f'Formal output throughput: {aggregate:.4f} tokens/s')
print(f"Prefix-hit rate: {rounds[0]['prefix_hits'] / rounds[0]['prefix_queries']:.4%}")
print('F08: scatter individually longest in 1207/1280 selected profiled triples')
print('Arithmetic check only: source measurements and hardware bounds are not independently verified.')
