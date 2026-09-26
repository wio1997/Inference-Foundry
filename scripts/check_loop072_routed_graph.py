"""Aggregate all-rank private complete-MoE Graph numerical and endpoint evidence."""

import argparse
import json
import statistics
from pathlib import Path


ARMS = ('A', 'A_repeat', 'B', 'A2')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    root = parser.parse_args().run_dir
    bench = json.loads((root / 'bench12.json').read_text())
    if bench['summary']['success'] != 12 or any(
            item['output_tokens'] != 1024 for item in bench['requests']):
        raise RuntimeError('12x1024 eager diagnostic carrier incomplete')
    runtime = [json.loads(path.read_text()) for path in
               sorted((root / 'runtime').glob('rank*_cohort*.json'))]
    if len(runtime) != 8 or {x['rank'] for x in runtime} != set(range(8)) or not all(
            x['pass'] and x['target_graph_mode'] == 'NONE' for x in runtime):
        raise RuntimeError('8-rank Runtime carrier failed')
    reports = []
    for rank in range(8):
        paths = sorted((root / 'fixture').glob(f'rank{rank}_cohort*.json'))
        if len(paths) != 1:
            raise RuntimeError(f'rank{rank} expected one report, got {paths}')
        item = json.loads(paths[0].read_text())
        if (item.get('stage') != 'complete' or item.get('pass') is not True or
                item.get('final_sync_error') or item.get('capture_state_error') or
                item.get('capture_still_active') is not False or item['rank'] != rank):
            raise RuntimeError(f'rank{rank} Graph fixture failed: {item.get("error", item)}')
        if len(item['samples']) != 52:
            raise RuntimeError(f'rank{rank} expected 52 Graph arm samples')
        reports.append(item)
    if len({(x['cohort'], x['cycle'], tuple(x['active_mask'])) for x in reports}) != 1:
        raise RuntimeError('rank cycle/mask mismatch')
    if sum(x['local_active_rows'] for x in reports) != 48:
        raise RuntimeError('active output coverage is not 48 rows')

    control_envelopes = []
    all_router_ids_exact = True
    all_router_a2_ids_exact = True
    all_router_weights_exact = True
    eager_a_comparisons = []
    for item in reports:
        controls = []
        candidates = []
        for sample in item['samples']:
            if sample['arm'] != 'A2' or sample['warmup']:
                continue
            comp = sample['comparisons']
            controls.extend((comp['A_A_repeat'], comp['A_A2']))
            candidates.append(comp['B_active_vs_A'])
            eager_a_comparisons.append(comp['A_eager'])
            all_router_ids_exact &= comp['router_ids_exact']
            all_router_a2_ids_exact &= comp['router_ids_A_vs_A2_exact']
            all_router_weights_exact &= comp['router_weights_A_vs_B']['exact']
        if len(controls) != 20 or len(candidates) != 10:
            raise RuntimeError('numerical comparison samples incomplete')
        finite = all(x['finite'] for x in controls + candidates)
        control_max = max(x['max_abs'] for x in controls)
        control_rms = max(x['rms'] for x in controls)
        control_signed = max(abs(x['mean_signed']) for x in controls)
        candidate_max = max(x['max_abs'] for x in candidates)
        candidate_rms = max(x['rms'] for x in candidates)
        candidate_signed = max(abs(x['mean_signed']) for x in candidates)
        control_envelopes.append({
            'rank': item['rank'], 'active_rows': item['local_active_rows'],
            'finite': finite,
            'control_max_abs': control_max, 'B_max_abs': candidate_max,
            'control_max_rms': control_rms, 'B_max_rms': candidate_rms,
            'control_max_abs_mean_signed': control_signed,
            'B_max_abs_mean_signed': candidate_signed,
            'B_within_control_envelope': item['local_active_rows'] == 0 or (
                finite and candidate_max <= control_max and
                candidate_rms <= control_rms and candidate_signed <= control_signed),
        })
    if not all_router_ids_exact:
        raise RuntimeError('active routing expert IDs changed in B')
    if not all_router_a2_ids_exact:
        raise RuntimeError('active routing expert IDs changed in independent A2 capture')
    if not all(x['finite'] for x in control_envelopes):
        raise RuntimeError('nonfinite active Graph output')

    device_by_arm = {arm: [] for arm in ARMS}
    host_by_arm = {arm: [] for arm in ARMS}
    for triplet in range(3, 13):
        for arm in ARMS:
            matching = [next(sample for sample in item['samples']
                             if sample['triplet'] == triplet and sample['arm'] == arm)
                        for item in reports]
            device_by_arm[arm].append(max(x['device_ms'] for x in matching))
            host_by_arm[arm].append(
                (max(x['host_end_ns'] for x in matching) -
                 min(x['host_start_ns'] for x in matching)) / 1e6)
    strict_device_wins = sum(
        device_by_arm['B'][i] < min(device_by_arm['A'][i],
                                    device_by_arm['A_repeat'][i],
                                    device_by_arm['A2'][i])
        for i in range(10))
    strict_host_wins = sum(
        host_by_arm['B'][i] < min(host_by_arm['A'][i],
                                  host_by_arm['A_repeat'][i],
                                  host_by_arm['A2'][i])
        for i in range(10))
    print(json.dumps({
        'status': 'private_graph_screen_complete',
        'scope': 'one layer4 full MoE Graph endpoint; no live Target or Product E2E claim',
        'cycle': reports[0]['cycle'],
        'active_mask': reports[0]['active_mask'],
        'local_active_rows': [x['local_active_rows'] for x in reports],
        'routing_ids_exact_all_measured': all_router_ids_exact,
        'routing_A_vs_independent_A2_ids_exact_all_measured': all_router_a2_ids_exact,
        'routing_weights_exact_all_measured': all_router_weights_exact,
        'eager_vs_graph_A_max_abs': max(x['max_abs'] for x in eager_a_comparisons),
        'eager_vs_graph_A_rms_max': max(x['rms'] for x in eager_a_comparisons),
        'numerical_control_envelopes': control_envelopes,
        'nonempty_ranks_within_control_envelope': sum(
            x['B_within_control_envelope'] for x in control_envelopes if x['active_rows']),
        'nonempty_rank_count': sum(bool(x['active_rows']) for x in control_envelopes),
        'max_rank_device_ms_median_by_arm': {arm: statistics.median(values)
                                            for arm, values in device_by_arm.items()},
        'rank_rendezvous_host_ms_median_by_arm': {arm: statistics.median(values)
                                                 for arm, values in host_by_arm.items()},
        'strict_B_faster_than_all_A_device_triplets': strict_device_wins,
        'strict_B_faster_than_all_A_host_triplets': strict_host_wins,
        'device_ms_by_arm': device_by_arm,
        'host_ms_by_arm': host_by_arm,
    }, indent=2))


if __name__ == '__main__':
    main()
