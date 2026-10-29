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
total_window = sum(x['duration_seconds'] for x in rounds)
input_total = sum(x['input_tokens'] for x in rounds)
query_total = sum(x['prefix_queries'] for x in rounds)
hit_total = sum(x['prefix_hits'] for x in rounds)
assert query_total == input_total and 0 <= hit_total <= query_total
print(f'Input tokens / sum of two AISBench windows: {input_total / total_window:.2f} tokens/s')
print(f'Prefix queries minus hits / sum of two AISBench windows: {(query_total - hit_total) / total_window:.2f} tokens/s')
print('These rates exclude warmup, inter-round time and full HTTP completion; they are not standalone prefill performance.')
nominal_macs = 3 * 1280 * 32768
nominal_macs_per_second = 24 * 1800 * 10**6 * 8192
print(f'Conditional compute-only model: {nominal_macs / nominal_macs_per_second * 10**6:.6f} us')
row_limited_capacity = nominal_macs_per_second * 3 / 16
print(f'Additional 3-of-16-row model: {nominal_macs / row_limited_capacity * 10**6:.6f} us')
assert 3 * 1280 + 1280 * 32768 == 41946880
print('Conditional cold-input example: 41946880 distinct raw bytes')
print('Conditional model assumptions and observed task identity are not independently verified.')
print('F08: scatter individually longest in 1207/1280 selected profiled triples')

# Published distributions contain anonymous marginal counts, not raw events.
def histogram_statistics(histogram):
    assert histogram and all(isinstance(value, int) and isinstance(count, int) and count > 0
                             for value, count in histogram)
    assert all(histogram[i][0] < histogram[i + 1][0] for i in range(len(histogram) - 1))
    total = sum(count for _, count in histogram)
    def value_at(position):
        seen = 0
        for value, count in histogram:
            seen += count
            if position < seen:
                return value
        raise ValueError('histogram position outside total')
    median_ns = (value_at((total - 1) // 2) + value_at(total // 2)) / 2
    return total, histogram[0][0], median_ns, histogram[-1][0]

pair = profile['q_scatter_rope_aggregate']
overlap = histogram_statistics(pair['overlap_ns_histogram'])
gap = histogram_statistics(pair['scatter_end_to_rope_start_ns_histogram'])
assert overlap == (1920, 7120, 22829.5, 30000)
assert gap == (1920, 520, 620, 1020)
print(f'Selected Q/scatter overlap median: {overlap[2] / 1000:.4f} us')
print(f'Selected scatter-to-RoPE gap median: {gap[2] / 1000:.3f} us')
print('Separate anonymous histograms verify these marginal distributions, not event identity or causality.')
# The following checks paired anonymous reported durations, not equal effective work.
from statistics import median
pairs = profile['dynamic_quant_crossshape_pairs']['durations_us_anonymous_pairs']
rows = profile['dynamic_quant_crossshape_pairs']['rows_per_pair']
assert len(pairs) == 32 and rows == [595, 1023]
assert all(len(p) == 2 and p[0] > 0 and p[1] > 0 for p in pairs)
ratios = [p[1] / p[0] for p in pairs]
per_row = [x * rows[0] / rows[1] for x in ratios]
assert round(min(ratios), 3) == 1.030 and round(median(ratios), 3) == 1.242 and round(max(ratios), 3) == 1.388
assert round(median(per_row), 3) == 0.722
print(f'DynamicQuant 1023/595 duration ratio (min/median/max): {min(ratios):.3f}/{median(ratios):.3f}/{max(ratios):.3f}')
print(f'DynamicQuant per-physical-row ratio median: {median(per_row):.3f}; different profiled batches, not equal-work comparison.')
print('Arithmetic check only: source measurements and hardware bounds are not independently verified.')
