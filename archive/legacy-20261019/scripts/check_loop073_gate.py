"""Aggregate private all-active gate placement Graph evidence."""

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
                item.get('capture_still_active') is not False or
                item.get('gate_weight_exact_to_rank0') is not True or item['rank'] != rank):
            raise RuntimeError(f'rank{rank} gate fixture failed: {item.get("error", item)}')
        if len(item['samples']) != 52:
            raise RuntimeError(f'rank{rank} expected 52 Graph arm samples')
        reports.append(item)
    if len({(x['cohort'], x['cycle'], tuple(x['active_mask'])) for x in reports}) != 1:
        raise RuntimeError('rank cycle/mask mismatch')
    if sum(reports[0]['active_mask']) != 12:
        raise RuntimeError('not a full-active decode cycle')

    comparison_summary = []
    router_ids_exact = True
    for item in reports:
        controls, candidates, gate_controls, gate_candidates = [], [], [], []
        for sample in item['samples']:
            if sample['arm'] != 'A2' or sample['warmup']:
                continue
            comp = sample['comparisons']
            controls.extend((comp['A_A_repeat_output'], comp['A_A2_output']))
            candidates.append(comp['B_A_output'])
            gate_controls.append(comp['A_A2_logits'])
            gate_candidates.append(comp['B_A_logits'])
            router_ids_exact &= comp['A_A2_router_ids_exact'] and comp['B_A_router_ids_exact']
        if len(controls) != 20 or len(candidates) != 10:
            raise RuntimeError('numerical samples incomplete')
        if not all(x['finite'] for x in controls + candidates + gate_controls + gate_candidates):
            raise RuntimeError('nonfinite Graph values')
        control_max = max(x['max_abs'] for x in controls)
        control_rms = max(x['rms'] for x in controls)
        control_bias = max(abs(x['mean_signed']) for x in controls)
        b_max = max(x['max_abs'] for x in candidates)
        b_rms = max(x['rms'] for x in candidates)
        b_bias = max(abs(x['mean_signed']) for x in candidates)
        comparison_summary.append({
            'rank': item['rank'], 'gate_logits_A2_control_max_abs': max(
                x['max_abs'] for x in gate_controls),
            'gate_logits_B_max_abs': max(x['max_abs'] for x in gate_candidates),
            'gate_logits_B_rms_max': max(x['rms'] for x in gate_candidates),
            'output_control_max_abs': control_max, 'output_B_max_abs': b_max,
            'output_control_max_rms': control_rms, 'output_B_max_rms': b_rms,
            'output_control_max_abs_mean_signed': control_bias,
            'output_B_max_abs_mean_signed': b_bias,
            'B_output_within_control_envelope': b_max <= control_max and
                b_rms <= control_rms and b_bias <= control_bias,
        })

    device = {arm: [] for arm in ARMS}
    host = {arm: [] for arm in ARMS}
    for triplet in range(3, 13):
        for arm in ARMS:
            matching = [next(sample for sample in item['samples']
                             if sample['triplet'] == triplet and sample['arm'] == arm)
                        for item in reports]
            device[arm].append(max(x['device_ms'] for x in matching))
            host[arm].append((max(x['host_end_ns'] for x in matching) -
                              min(x['host_start_ns'] for x in matching)) / 1e6)
    paired_b_delta_us = [(device['B'][i] -
                          (device['A'][i] + device['A2'][i]) / 2) * 1000
                         for i in range(10)]
    print(json.dumps({
        'status': 'private_graph_screen_complete' if router_ids_exact else
                  'numerical_gate_failed_routing',
        'scope': 'one all-active layer4 full-MoE Graph endpoint; no live Target or Product E2E claim',
        'cycle': reports[0]['cycle'], 'active_mask': reports[0]['active_mask'],
        'gate_weight_exact_all_ranks': True,
        'router_ids_exact_all_measured': router_ids_exact,
        'routing_gate_pass': router_ids_exact,
        'comparison_summary': comparison_summary,
        'ranks_output_within_control_envelope': sum(
            x['B_output_within_control_envelope'] for x in comparison_summary),
        'max_rank_device_ms_median_by_arm': {
            arm: statistics.median(device[arm]) for arm in ARMS},
        'rank_rendezvous_host_ms_median_by_arm': {
            arm: statistics.median(host[arm]) for arm in ARMS},
        'strict_B_faster_than_all_A_device_triplets': sum(
            device['B'][i] < min(device['A'][i], device['A_repeat'][i], device['A2'][i])
            for i in range(10)),
        'strict_B_faster_than_all_A_host_triplets': sum(
            host['B'][i] < min(host['A'][i], host['A_repeat'][i], host['A2'][i])
            for i in range(10)),
        'paired_B_minus_mean_A_A2_device_us': paired_b_delta_us,
        'device_ms_by_arm': device, 'host_ms_by_arm': host,
    }, indent=2))
    if not router_ids_exact:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
