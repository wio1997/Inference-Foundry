#!/usr/bin/env python3
"""Fixed-observed-cost local Scheduling relaxation from admitted Run592 rows."""
from __future__ import annotations

import hashlib
import json
import statistics
from decimal import Decimal
from pathlib import Path

ROOT = Path('/data/wio/Inference_Foundry')
SOURCE = ROOT / 'evidence/20260928_loop081_bound/run592/summary.json'
SOURCE_SHA = '20c314fdf00dc66e45e9e0491f7ec3a67f8e549681d608fe4863f0aa4107f172'
OUT = ROOT / 'evidence/20260928_loop081_bound/run639/local_fixed_cost.json'


def need(ok, why):
    if not ok:
        raise AssertionError(why)


def stat(values):
    vals = [float(v) for v in values]
    return {'min': min(vals), 'median': statistics.median(vals),
            'max': max(vals)}


def main():
    need(hashlib.sha256(SOURCE.read_bytes()).hexdigest() == SOURCE_SHA,
         'Run592 source drift')
    src = json.loads(SOURCE.read_text())
    need(src['status'] == 'historical_instrumented_current_slice_only' and
         src['rank_count'] == 8 and src['occurrences_per_rank'] == 2 and
         src['fixed_W0_equivalence_to_Run99'] is False and len(src['rows']) == 16,
         'Run592 admission scope')
    rows = []
    for row in src['rows']:
        times = row['task_times_us']
        duration = {role: Decimal(times[role]['duration'])
                    for role in ('partial','rs','copy')}
        current = Decimal(row['intervals_us']['partial_start_to_copy_end'])
        cost = sum(duration.values())
        need(cost > 0 and current >= cost, 'same-row cost/envelope ordering')
        gap = current - cost
        # This is conditional on treating measured RS duration, possibly
        # including peer wait, as frozen service. It is not a universal floor.
        rows.append({'rank': row['rank'], 'occurrence': row['occurrence'],
                     'observed_primitive_duration_sum_us': str(cost),
                     'current_instrumented_envelope_us': str(current),
                     'conditional_max_gap_if_serial_cost_frozen_us': str(gap),
                     'rs_observed_duration_us': str(duration['rs'])})
    need({r['rank'] for r in rows} == set(range(8)) and
         all(sum(r['rank']==rank for r in rows)==2 for rank in range(8)),
         'all8 two-occurrence identity')
    out = {
        'status': 'local_historical_fixed_observed_cost_relaxation_only',
        'scope': 'Run246 Level1, 16 local MatMul->RS->copy selected chains, Run592 cross-run semantic mapping conditional',
        'source_path': str(SOURCE.relative_to(ROOT)),
        'source_sha256': SOURCE_SHA,
        'cost_policy': 'Observed MatMul, RS, copy durations kept fixed. RS may contain peer wait; this local cost assumption can overstate necessary service. No other tasks, rank synchronization, Product or W0 transfer.',
        'graph_policy': 'serial MatMul->RS->copy only; actual envelope includes gaps/events. Removing its conditional gap requires proving other waits/consumers do not constrain the schedule.',
        'statistics_us': {
            'conditional_fixed_observed_cost': stat(Decimal(r['observed_primitive_duration_sum_us']) for r in rows),
            'current_instrumented_envelope': stat(Decimal(r['current_instrumented_envelope_us']) for r in rows),
            'conditional_max_gap': stat(Decimal(r['conditional_max_gap_if_serial_cost_frozen_us']) for r in rows),
        },
        'rows': rows,
        'formal_current_tps': 571.681,
        'same_formal_W0': False,
        'whole_product_framework_tps_interval': None,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps({'status':out['status'], 'statistics_us':out['statistics_us']}))


if __name__ == '__main__':
    main()
