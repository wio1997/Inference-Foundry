#!/usr/bin/env python3
"""Check published summary arithmetic; this does not reproduce measurements."""
import json
from pathlib import Path

summary = json.loads((Path(__file__).parent / 'evidence-summary.json').read_text())
stage = summary.get('stage_consolidation', {})
if 'indexer_ledger' in stage:
    ledger = stage['indexer_ledger']
    example = ledger['example']
    positions = example['assumed_query_rows'] * example['assumed_keys_per_query']
    assert positions == example['position_count']
    assert ledger['INT8_QK_MAC_per_position'] == 32 * 128
    assert ledger['FP16_weighted_head_MAC_per_position'] == 32
    assert example['logical_score_roundtrip_bytes'] == 8 * positions
    p8 = 24 * 1_800_000_000 * 8192
    p16 = 24 * 1_800_000_000 * 4096
    cube_us = (4096 / p8 + 32 / p16) * positions * 1e6
    assert abs(cube_us - example['nominal_cube_only_us']) < 1e-10
    tasks = ledger['selected_three_task_us']
    assert abs(tasks['prepare'] / sum(tasks.values()) - ledger['prepare_fraction_of_three_task_sum']) < 1e-12
    print(f'Conditional QLI example: Cube only {cube_us:.6f} us; logical score GM bytes {8 * positions}')
    print('This excludes sorting, key traffic, padding and waits; it is not complete QLI efficiency.')
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
matched = profile['same_stream_l14_l39_matched']['groups']
assert len(matched) == 18 and all(len(g['ratios_l14_over_l39']) == 8 and min(g['ratios_l14_over_l39']) > 0 for g in matched)
mc2 = [g for g in matched if g['name'] == 'DispatchFFNCombine']
assert len(mc2) == 1
other = [x for g in matched if g['name'] != 'DispatchFFNCombine' for x in g['ratios_l14_over_l39']]
assert len(other) == 136
assert [round(min(mc2[0]['ratios_l14_over_l39']), 3), round(max(mc2[0]['ratios_l14_over_l39']), 3)] == [19.344, 19.516]
assert [round(min(other), 3), round(max(other), 3)] == [0.794, 1.286]
print('Selected same-stream L14/L39 ratios: MC2 19.344–19.516; 17 other matched signatures 0.794–1.286. Not equal work or causal evidence.')
cross = profile['mc2_cross_group_layer_medians']
assert [c['physical_input_rows'] for c in cross] == [1023, 643]
assert all(len(c['ratios']) == 40 and min(c['ratios']) > 0 for c in cross)
assert [sum(v > 1 for v in c['ratios']) for c in cross] == [33, 40]
assert [[round(min(c['ratios']), 3), round(median(c['ratios']), 3), round(max(c['ratios']), 3)] for c in cross] == [[0.990, 1.531, 8.528], [1.174, 1.456, 10.331]]
assert histogram_statistics(profile['q_scatter_tail_minus_rope_ns_histogram']) == (1920, 4899, 7360.0, 14940)
print('Selected Q scatter-tail minus RoPE duration: 1920 samples, min/median/max 4.899/7.360/14.940 us; not proof of executable overlap.')
print('Selected MC2 groups: 40 layer medians per shape; different groups, not matched workloads.')
gaps = profile['selected_mc2_gap_coverage_fractions']
assert len(gaps) == 32 and all(len(x) == 3 and 0 <= x[0] <= 1 and 0 <= x[1] <= x[2] <= 1 for x in gaps)
assert [[round(min(r[i] for r in gaps)*100,2), round(max(r[i] for r in gaps)*100,2)] for i in (1,2)] == [[8.52,68.56],[11.34,97.05]]
print('Selected inter-MC2 gap coverage: other same-stream 8.52-68.56%; all selected streams 11.34-97.05%; uncovered is not idle.')
client_gaps = profile['baseline_sse_line_gaps']
assert [v['request_count'] for v in client_gaps] == [8,8]
assert [v['sse_gap_count'] for v in client_gaps] == [1759,1754]
assert [v['gaps_over_100ms_count'] for v in client_gaps] == [11,11]
assert [[round(v['pooled_quantiles_ms'][k],2) for k in ('p50','p99')] for v in client_gaps] == [[65.33,73.76],[65.13,78.07]]
print('Baseline SSE-line gap summaries consistent; not per-token TPOT and quantiles are not recomputed from raw intervals here.')
hc = profile['hc_repeated_scope_ratios']
assert len(hc) == 8 and all(len(g['ratios']) == 320 and min(g['ratios']) > 0 for g in hc)
post = [g for g in hc if g['name'] == 'HcPost']
pre = [g for g in hc if g['name'] == 'HcPre']
assert len(post) == len(pre) == 4
assert sorted(round(median(g['ratios']),3) for g in post if g['physical_rows'] == 1023) == [0.856,0.863]
assert sorted(round(median(g['ratios']),3) for g in post if g['physical_rows'] != 1023) == [0.974,0.998]
assert [round(min(median(g['ratios']) for g in pre),3), round(max(median(g['ratios']) for g in pre),3)] == [0.990,1.029]
dot = profile['int8_exact_dot_abs_bounds']
assert [k*dot['signed_input_abs_max']**2 for k in dot['k_values']] == [83886080,37748736]
assert max(k*dot['signed_input_abs_max']**2 for k in dot['k_values']) < dot['int32_max']
print('Selected HC paired ratio medians verified; equal signatures do not prove equal workloads.')
print('Conditional signed INT8 exact dot absolute bounds verified; historical operator contract remains unbound.')
tail_cases = [(595,5120,4608),(1023,5120,4608),(595,2304,5120),(1023,2304,5120)]
padded_us = [((m+15)//16*16)*k*n/capacity*1e6 for m,k,n in tail_cases]
assert [round(t,2) for t in padded_us] == [40.53,68.27,20.27,34.13]
print('Conditional 16-row padding example verified; actual tiling and capacity unbound.')
outstanding = profile['baseline_client_outstanding']
assert len(outstanding)==2
for item in outstanding:
    durations=item['outstanding_duration_ms_by_count']; window=item['window_ms']; assert len(durations)==9 and abs(sum(durations)-window)<1e-6
    assert item['request_count']==8 and item['tokens']==8192
assert [round(x['outstanding_duration_ms_by_count'][8]/x['window_ms']*100,2) for x in outstanding] == [96.35,92.13]
assert [round(sum(i*t for i,t in enumerate(x['outstanding_duration_ms_by_count']))/x['window_ms'],3) for x in outstanding] == [7.860,7.755]
assert [round(x['tokens']*1000/x['window_ms'],2) for x in outstanding] == [522.37,517.35]
print('Client POST-to-last-SSE overlap summaries verified; not service-side concurrency.')
qli = profile['qli_layer_role_ratios']
assert len(qli) == 28 and min(qli)>0
assert [round(min(qli),3), round(median(qli),3), round(max(qli),3)] == [2.537,2.588,2.676]
print('Selected QLI later-layer / layer20 ratios verified; no tensor identity or causal claim.')
print('Arithmetic check only: source measurements and hardware bounds are not independently verified.')
