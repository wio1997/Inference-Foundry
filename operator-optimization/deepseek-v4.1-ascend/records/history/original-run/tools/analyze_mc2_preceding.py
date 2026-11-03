"""Read-only, CPU-only decomposition of the already captured layer-zero arrivals.

Cross-rank pairing is the time-consistent offset-zero hypothesis, not an op-ID
or cross-device clock proof. No waiting interval is promoted to a lower bound.
"""
from pathlib import Path
from decimal import Decimal
from collections import defaultdict
import bisect, csv, json, statistics

T = Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41')
ROOT = Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-20261028/diagnostic/runtime/diagnostic-traces')
O = T / 'mc2-preceding-analysis'
O.mkdir(exist_ok=False)
mapping_path = T / 'captured-layer-attribution/module-call-metrics.json'
assert mapping_path.stat().st_size < 20 * 1024 * 1024
mapping = json.loads(mapping_path.read_text())
target = defaultdict(list)
for row in csv.DictReader((T / 'mc2-arrival-analysis-v2/rank_target_calls.csv').open()):
    row['start_ns'] = int(row['start_ns'])
    row['end_ns'] = int(row['end_ns'])
    row['target_call_ordinal'] = int(row['target_call_ordinal'])
    target[row['rank']].append(row)
for rs in target.values():
    rs.sort(key=lambda r: r['target_call_ordinal'])
detail, slices = [], []
for rank, calls in sorted(target.items()):
    mods = sorted([m for m in mapping if m['rank'] == rank], key=lambda m: m['window_start_ns'])
    starts = [m['window_start_ns'] for m in mods]
    kernels = []
    for row in csv.DictReader((ROOT / rank / 'ASCEND_PROFILER_OUTPUT/kernel_details.csv').open()):
        row['_s'] = int(Decimal(row['Start Time(us)'].strip()) * 1000)
        row['_e'] = row['_s'] + int(Decimal(row['Duration(us)']) * 1000)
        kernels.append(row)
    kernels.sort(key=lambda r: r['_s'])
    kstarts = [r['_s'] for r in kernels]
    for call in calls[::40]:
        if call['model_id'] == '4294967295' or int(call['physical_input_rows']) != 3:
            continue
        mi = bisect.bisect_right(starts, call['start_ns']) - 1
        mod = mods[mi]
        assert mod['layer'] == 0 and call['end_ns'] <= mod['window_end_ns']
        a = bisect.bisect_left(kstarts, mod['window_start_ns'])
        b = bisect.bisect_right(kstarts, mod['window_end_ns'])
        kr = [k for k in kernels[a:b] if k['Model ID'] == mod['model_id'] and k['_e'] <= mod['window_end_ns']]
        marker = [k for k in kr if k['Task ID'] == mod['marker_task_ids'][2] and k['Stream ID'] == mod['main_stream_id'] and k['Type'] == 'HcPre']
        attn = [k for k in kr if k['Type'] == 'SparseFlashMla']
        assert len(marker) == len(attn) == 1
        row = {'rank': rank, 'dp': rank[:3], 'target_forward_ordinal': call['target_call_ordinal'] // 40,
               'captured_run_in_model': mod['run_in_model'], 'interior_runs_1_2_3': mod['run_in_model'] in [1, 2, 3],
               'layer_start_ns': mod['window_start_ns'], 'attention_start_ns': attn[0]['_s'], 'attention_end_ns': attn[0]['_e'],
               'ffn_pre_start_ns': marker[0]['_s'], 'mc2_start_ns': call['start_ns'], 'mc2_end_ns': call['end_ns'],
               'layer_to_mc2_us': (call['start_ns'] - mod['window_start_ns']) / 1000,
               'attention_duration_us': (attn[0]['_e'] - attn[0]['_s']) / 1000,
               'ffn_pre_to_mc2_us': (call['start_ns'] - marker[0]['_s']) / 1000,
               'mc2_duration_us': (call['end_ns'] - call['start_ns']) / 1000}
        detail.append(row)
        # A bounded local evidence slice: previous six launches plus this layer's
        # four mHC markers and attention/MC2. Names alone do not assign ownership.
        selected = kernels[max(0, a - 6):a] + [k for k in kr if k['Type'] in ['HcPre', 'HcPost', 'SparseFlashMla', 'DispatchFFNCombine']]
        for k in selected:
            slices.append({'rank': rank, 'target_forward_ordinal': row['target_forward_ordinal'],
                           'relative_to_layer_start_us': (k['_s'] - mod['window_start_ns']) / 1000,
                           'duration_us': float(Decimal(k['Duration(us)'])), 'type': k['Type'],
                           'model_id': k['Model ID'], 'stream_id': k['Stream ID'], 'task_id': k['Task ID'],
                           'previous_tasks_not_semantically_attributed': k['_s'] < mod['window_start_ns']})

groups = defaultdict(list)
for row in detail:
    groups[row['target_forward_ordinal']].append(row)
paired = []
for ordinal, rows in sorted(groups.items()):
    if len(rows) != 16:
        continue
    origin = min(r['layer_start_ns'] for r in rows)
    spread = lambda key: (max(r[key] for r in rows) - min(r[key] for r in rows)) / 1000
    rec = {'target_forward_ordinal': ordinal, 'all_interior_runs_1_2_3': all(r['interior_runs_1_2_3'] for r in rows),
           'layer_start_spread_us': spread('layer_start_ns'), 'attention_start_spread_us': spread('attention_start_ns'),
           'attention_end_spread_us': spread('attention_end_ns'), 'ffn_pre_start_spread_us': spread('ffn_pre_start_ns'),
           'mc2_start_spread_us': spread('mc2_start_ns'), 'mc2_end_spread_us': spread('mc2_end_ns')}
    for dp in ['dp0', 'dp1']:
        items = [r for r in rows if r['dp'] == dp]
        for key in ['layer_start_ns', 'attention_start_ns', 'attention_end_ns', 'ffn_pre_start_ns', 'mc2_start_ns', 'mc2_end_ns']:
            rec[dp + '_' + key.replace('_ns', '_mean_offset_us')] = statistics.mean((r[key] - origin) / 1000 for r in items)
        rec[dp + '_layer_to_mc2_mean_us'] = statistics.mean(r['layer_to_mc2_us'] for r in items)
        rec[dp + '_mc2_mean_duration_us'] = statistics.mean(r['mc2_duration_us'] for r in items)
    paired.append(rec)

for name, rows in [('rank_layer0_timeline.csv', detail), ('paired_timeline.csv', paired), ('bounded_local_slices.csv', slices)]:
    assert rows
    with (O / name).open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

summary = {'status': 'OBSERVED_TIMELINE_UNDER_OFFSET_ZERO_PAIRING_HYPOTHESIS',
           'clock_error_bound_us': None, 'internal_mc2_sequence_id_known': False,
           'paired_layer0_calls': len(paired), 'unpaired_captured_rows_retained': len(detail) - 16 * len(paired),
           'paired_timeline': paired, 'no_device_work': True,
           'interpretation_limit': 'Pre-MC2 elapsed time and timestamp spreads do not establish causality, unavoidable delay, or removable time. Earlier task names do not prove ownership.'}
assert len(json.dumps(summary).encode()) < 32768
(O / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2))
