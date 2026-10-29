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
# Counter deltas are service aggregate, not per-request timing or accuracy.
for row, expected in zip(rounds, ((1751,8755,6447),(1746,8730,6456))):
    c = row['draft_counter_delta']
    assert (c['drafts'],c['draft_tokens'],c['accepted_tokens']) == expected
    assert sum(c['accepted_per_position']) == c['accepted_tokens']
    print(f"Draft token counter acceptance: {100*c['accepted_tokens']/c['draft_tokens']:.2f}%; accepted per draft: {c['accepted_tokens']/c['drafts']:.3f}")
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
# Paired summaries retain no request, tensor or device identifiers.
gate = profile['mc2_gate_l14_l39']['anonymous_paired_durations_us']
assert len(gate) == 8 and all(len(p) == 4 and min(p) > 0 for p in gate)
assert sum(p[0] > p[1] for p in gate) == 4 and sum(p[0] < p[1] for p in gate) == 4
assert round(median([p[0] for p in gate]), 3) == 41.579
assert round(median([p[1] for p in gate]), 3) == 38.449
print('Eight gate pair comparisons: four L14 longer, four L14 shorter; no causal claim.')
chain = profile['ffn_crossshape_four_tasks']; records = chain['anonymous_paired_durations_us']
assert len(records) == 32 and all(len(p) == 4 and all(len(q) == 2 and min(q) > 0 for q in p) for p in records)
medians = [median(p[i][1] / p[i][0] for p in records) for i in range(4)]
assert [round(v, 3) for v in medians] == [1.242, 1.369, 1.422, 1.293]
print('Four selected task 1023/595 duration ratio medians: ' + '/'.join(f'{x:.3f}' for x in medians))
# Five task timings are within one profiled pair of layer scopes.
five = profile['selected_layer36_five_task_pairs']['durations_us']
assert len(five) == 5 and all(len(p) == 3 and p[1] > 0 and p[2] > 0 for p in five)
assert [round(p[2] / p[1], 3) for p in five] == [1.346, 1.130, 1.006, 0.990, 0.942]
print('Selected five task duration ratios: ' + '/'.join(f'{p[2]/p[1]:.3f}' for p in five))
# The following model is conditional, not independently established hardware capacity.
capacity = 24 * 1_800_000_000 * 8192
for i, k, n in ((1, 5120, 4608), (3, 2304, 5120)):
    for j, m in enumerate((595, 1023)):
        model_us = m*k*n/capacity*1e6
        observed_us = median(p[i][j] for p in records)
        print(f'Conditional INT8 model M={m} K={k} N={n}: {model_us:.3f} us; observed median {observed_us:.2f} us')
q = profile['q_two_projection_shapes']
macs = [q['m'] * q['k'] * n for n in q['outputs']]
assert macs == [42907729920, 5363466240] and macs[0] == 8 * macs[1]
assert [round(m/t/1e6,3) for m,t in zip(macs,q['duration_us'])] == [139.347, 105.455]
for n, nz in zip(q['outputs'], q['weight_nz_shapes']):
    assert [(n+31)//32, (q['k']+15)//16, 16, 32] == nz
r=profile['f32_selected_projection_rows']; assert r['q_rows_histogram'] == {'1023':7, '1019':1} and r['kv_rows_histogram'] == {'8180':8}
assert 7*1023+1019 == 8180 and 1023*512 != 1019*512
print('Q pair conditional MACs: 42.908G / 5.363G; 9.1 non-transpose shapes are compatible, historical parameters unbound.')
print('Selected Q rows: seven 1023, one 1019; same communication group and valid rows unproven.')
c=profile['client_first_data_comparison']
assert c['recapture_requests'] == c['recapture_first_data_nonempty_count'] == 8
assert c['baseline_first_data_ms']['max'] < c['recapture_first_data_ms']['min']
print('Selected baseline and diagnostic client first-data ranges do not overlap; timing semantics differ.')
print('Arithmetic check only: source measurements and hardware bounds are not independently verified.')
