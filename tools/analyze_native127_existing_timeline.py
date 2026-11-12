"""Decompose existing AIS request timestamps; never runs model arithmetic or NPU work."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import hashlib
import json
import math
import argparse

W = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--evidence-dir', type=Path, default=W / 'evidence/migration103-20261112')
E = parser.parse_args().evidence_dir.resolve()
inputs = {}


def read(path):
    inputs[str(path.relative_to(E))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return json.loads(path.read_text())


def decompose(rows, concurrency):
    assert len(rows) == 32 and len({r['data_id'] for r in rows}) == 32
    events = []
    for r in rows:
        assert r['success'] and r['input_tokens'] == 1024 and r['output_tokens'] == 256
        assert r['request_start'] <= r['first_token'] <= r['request_end']
        events += [(r['request_start'], 1), (r['request_end'], -1)]
    active = maximum = 0
    for _, change in sorted(events):
        active += change
        maximum = max(active, maximum)
        assert 0 <= active <= concurrency
    assert active == 0
    first = min(r['request_start'] for r in rows)
    duration = max(r['request_end'] for r in rows) - first
    occupied = sum(r['request_end'] - r['request_start'] for r in rows) / concurrency
    ttft = sum(r['first_token'] - r['request_start'] for r in rows) / concurrency
    post = sum(r['request_end'] - r['first_token'] for r in rows) / concurrency
    idle = duration - occupied
    assert idle >= -1e-8 and math.isclose(ttft + post + idle, duration, abs_tol=1e-8)
    groups = []
    ordered = sorted(rows, key=lambda r: r['data_id'])
    for offset in range(0, len(ordered), concurrency):
        subset = ordered[offset:offset + concurrency]
        groups.append({'data_ids': [r['data_id'] for r in subset],
            'start_offset_s': min(r['request_start'] for r in subset) - first,
            'end_offset_s': max(r['request_end'] for r in subset) - first,
            'mean_request_duration_s': sum(r['request_end'] - r['request_start'] for r in subset) / len(subset),
            'mean_ttft_s': sum(r['first_token'] - r['request_start'] for r in subset) / len(subset),
            'mean_post_first_s': sum(r['request_end'] - r['first_token'] for r in subset) / len(subset)})
    return {'formal_requests': len(rows), 'max_active_requests': maximum,
        'formal_start_monotonic_s': first, 'makespan_s': duration, 'output_tps': 8192 / duration,
        'occupied_request_time_per_slot_s': occupied, 'ttft_area_per_slot_s': ttft,
        'post_first_area_per_slot_s': post, 'unoccupied_area_per_slot_s': idle,
        'data_id_groups_not_engine_batches': groups}


reproductions = {g: read(E / f'native{g}-reproduction.json') for g in (114, 127)}
reference = read(E / 'reference117-reproduction.json')
arms = {str(g): {f'C{c}': decompose(q['raw_timing_endpoints'][f'C{c}']['rows'], c)
                for c in (4, 8)} for g, q in reproductions.items()}
deltas = {}
for c in (4, 8):
    phase = f'C{c}'
    a, b = arms['114'][phase], arms['127'][phase]
    fields = ('makespan_s', 'ttft_area_per_slot_s', 'post_first_area_per_slot_s',
              'unoccupied_area_per_slot_s')
    deltas[phase] = {k: b[k] - a[k] for k in fields}
    assert math.isclose(deltas[phase]['makespan_s'], sum(deltas[phase][k] for k in fields[1:]), abs_tol=1e-8)

B = E / 'native-g127-complete/native-run'
ranks = [read(B / f'communication/native-tp8/rank{rank}/native-prefill-state.json') for rank in range(8)]
assert all(len(q['events']) == len(ranks[0]['events']) for q in ranks)
anomalies = []
host_summary = {}
for c in (4, 8):
    phase = f'C{c}'
    receipt = read(B / f'aisbench-c{c}-result.json')
    first = arms['127'][phase]['formal_start_monotonic_s']
    last = first + arms['127'][phase]['makespan_s']
    bucket = {'formal': [], 'existing_warmups_or_other_phase_overhead': []}
    for i, event in enumerate(ranks[0]['events']):
        ns = event['started_monotonic_ns']
        if not receipt['started_monotonic_ns'] <= ns <= receipt['finished_monotonic_ns']:
            continue
        formal = first <= ns / 1e9 <= last
        duration = (event['host_return_monotonic_ns'] - ns) / 1e6
        bucket['formal' if formal else 'existing_warmups_or_other_phase_overhead'].append((event['hit'], duration))
        assert all(q['events'][i]['spans'] == event['spans'] and q['events'][i]['hit'] == event['hit'] for q in ranks)
        if duration <= 500:
            continue
        peers = [q['events'][i] for q in ranks]
        wall = datetime.fromisoformat(receipt['started_at']) + timedelta(seconds=(ns - receipt['started_monotonic_ns']) / 1e9)
        anomalies.append({'phase': phase, 'in_formal_request_interval': formal, 'event_index': i,
            'estimated_wall_time_from_same_host_receipt': wall.isoformat(), 'formal_start_offset_s': ns / 1e9 - first,
            'spans': event['spans'], 'hit': event['hit'],
            'rank_host_intervals_ms': [(p['host_return_monotonic_ns'] - p['started_monotonic_ns']) / 1e6 for p in peers],
            'rank_start_spread_ms': (max(p['started_monotonic_ns'] for p in peers) - min(p['started_monotonic_ns'] for p in peers)) / 1e6,
            'saved_entries_delta': event['counts'].get('saved_entries', 0) - ranks[0]['events'][i-1]['counts'].get('saved_entries', 0)})
    host_summary[phase] = {scope: {'calls': len(values), 'hits': sum(hit for hit, _ in values),
        'miss_host_intervals_sum_ms': sum(ms for hit, ms in values if not hit),
        'hit_host_intervals_sum_ms': sum(ms for hit, ms in values if hit)} for scope, values in bucket.items()}

ref_c4 = decompose(reference['raw_timing_endpoints']['C4']['rows'], 4)
threshold = ref_c4['makespan_s'] / 1.3
formal_c4 = [e for e in anomalies if e['phase'] == 'C4' and e['in_formal_request_interval']]
assert len(formal_c4) == 1
longest_rank_interval_s = max(formal_c4[0]['rank_host_intervals_ms']) / 1000
rank0_interval_s = formal_c4[0]['rank_host_intervals_ms'][0] / 1000
bound = {'reference117_c4_30pct_time_limit_s': threshold,
    'candidate_c4_time_to_save_s': arms['127']['C4']['makespan_s'] - threshold,
    'rank0_anomaly_host_interval_s': rank0_interval_s,
    'max_rank_anomaly_host_interval_s': longest_rank_interval_s,
    'c4_time_if_entire_max_rank_host_interval_removed_s': arms['127']['C4']['makespan_s'] - longest_rank_interval_s,
    'remaining_to_threshold_even_in_that_accounting_scenario': arms['127']['C4']['makespan_s'] - longest_rank_interval_s - threshold,
    'scope': 'fixed-rest accounting only;not a causal bound when scheduling or MTP trajectory changes',
    'decision': 'removing only this observed interval does not account for enough savings;no anomaly-only new model run'}
assert bound['remaining_to_threshold_even_in_that_accounting_scenario'] > 0
result = {'created_at_utc': datetime.now(timezone.utc).isoformat(),
    'analysis_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'candidate_source': reproductions[127]['source_identity']['commit'],
    'baseline_source': reproductions[114]['source_identity']['commit'],
    'analysis_kind': 'offline existing metadata only', 'new_model_loads': 0, 'remote_calls': 0,
    'arms': arms, 'candidate_minus_baseline_seconds': deltas,
    'host_event_scope': 'host planning/consensus/original model enqueue/snapshot interval,not device kernel time',
    'host_prefill_intervals': host_summary, 'host_intervals_over_500ms': anomalies,
    'single_interval_accounting': bound,
    'interpretation_limits': [
        'post-first request time includes contention,other-request prefill,scheduling and transport;not pure decode',
        'unoccupied client slots do not identify the server scheduler as the cause',
        'data-ID groups are descriptive cohorts,not verified engine batches',
        'distinct historical arms;no causal attribution of total difference to snapshot copies',
        'no new quality or performance acceptance;default remains114;closed stage budget not reopened'],
    'inputs_sha256': inputs}
path = E / 'native127-existing-timeline-analysis.json'
path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'output': path.name, 'deltas_s': deltas,
                  'accounting': bound, 'host_counts': host_summary}, ensure_ascii=False))
