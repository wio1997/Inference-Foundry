#!/usr/bin/env python3
"""Paired fixed-layer hidden gather scheduling trace screen; device intervals only."""
import argparse
import glob
import json
import re
import statistics
from pathlib import Path


def selected_traces(root):
    found = glob.glob(str(Path(root) / 'rank*_ascend_pt/ASCEND_PROFILER_OUTPUT/trace_view.json'))
    by_rank = {}
    for path in found:
        match = re.search(r'rank(\d+)_', path)
        if match:
            rank = int(match.group(1))
            if rank not in by_rank or path > by_rank[rank]:
                by_rank[rank] = path
    if sorted(by_rank) != list(range(8)):
        raise RuntimeError(f'expected all8 ranks, got {sorted(by_rank)}')
    return by_rank


def sample(path, mode):
    events = json.loads(Path(path).read_text())
    device = sorted((e for e in events if e.get('ph') == 'X'
                    and 'Physic Stream Id' in e.get('args', {})),
                    key=lambda e: float(e['ts']))
    compressors = [e for e in device if e.get('name') == 'Compressor']
    if len(compressors) != 124:
        raise RuntimeError(f'{path}: expected 124 Compressor tasks, got {len(compressors)}')
    rows = []
    for cycle, ci in enumerate((0, 62)):
        comp = compressors[ci]
        ct = float(comp['ts'])
        gathers = [e for e in device if e['name'].startswith('hcom_allGather')
                   and ct - 300 < float(e['ts']) < ct]
        if len(gathers) != 1:
            raise RuntimeError(f'{path}: cycle {cycle}: expected one preceding hidden gather, got {len(gathers)}')
        gather = gathers[0]
        start = float(gather['ts'])
        end = start + float(gather['dur'])
        pre = [e for e in device if e['name'] == 'RmsNorm'
               and 0 <= start - (float(e['ts']) + float(e['dur'])) < 30]
        if len(pre) != 1:
            raise RuntimeError(f'{path}: cycle {cycle}: expected one common predecessor RmsNorm, got {len(pre)}')
        pre_end = float(pre[0]['ts']) + float(pre[0]['dur'])
        main = [e for e in device if str(e['args']['Physic Stream Id']) == '1'
                and e['name'] not in ('EVENT_WAIT', 'EVENT_RECORD')]
        overlapping = [e for e in main if float(e['ts']) < end
                       and float(e['ts']) + float(e['dur']) > start]
        overlap_us = sum(max(0.0, min(end, float(e['ts']) + float(e['dur']))
                             - max(start, float(e['ts']))) for e in overlapping)
        next_a2a = next((e for e in device if e['name'].startswith('hcom_alltoall')
                         and float(e['ts']) > ct), None)
        first_after_gather = next((e for e in main if float(e['ts']) >= end), None)
        q_rope = next((e for e in main if e['name'] == 'InplacePartialRotaryMul'
                       and start < float(e['ts']) < ct), None)
        if q_rope is None:
            raise RuntimeError(f'{path}: cycle {cycle}: missing local Q RoPE')
        q_rope_end = float(q_rope['ts']) + float(q_rope['dur'])
        wkv_input = next((e for e in main if 'DynamicQuant' in e['name']
                          and q_rope_end <= float(e['ts']) < ct), None)
        if wkv_input is None:
            raise RuntimeError(f'{path}: cycle {cycle}: missing post-join WKV input quant')
        rows.append({
            'mode': mode, 'cycle_sample': cycle, 'trace': path,
            'gather_start_us': start, 'gather_duration_us': float(gather['dur']),
            'gather_end_to_first_compressor_us': ct-end,
            'gather_start_to_first_compressor_us': ct-start,
            'common_predecessor_end_to_gather_start_us': start-pre_end,
            'common_predecessor_end_to_q_rope_end_us': q_rope_end-pre_end,
            'common_predecessor_end_to_wkv_input_us': float(wkv_input['ts'])-pre_end,
            'common_predecessor_end_to_first_compressor_us': ct-pre_end,
            'common_predecessor_end_to_next_alltoall_us': float(next_a2a['ts'])-pre_end if next_a2a else None,
            'common_predecessor_end_to_next_alltoall_end_us': float(next_a2a['ts'])+float(next_a2a['dur'])-pre_end if next_a2a else None,
            'gather_end_to_q_rope_end_us': q_rope_end-end,
            'q_rope_end_to_wkv_input_us': float(wkv_input['ts'])-q_rope_end,
            'gather_start_to_next_alltoall_us': float(next_a2a['ts'])-start if next_a2a else None,
            'gather_end_to_next_main_kernel_us': float(first_after_gather['ts'])-end if first_after_gather else None,
            'device_overlap_us': overlap_us,
            'overlap_kernel_names': [e['name'] for e in overlapping],
            'gather_stream': str(gather['args']['Physic Stream Id']),
            'compressor_stream': str(comp['args']['Physic Stream Id']),
        })
    return rows


def median(rows, key):
    vals = [r[key] for r in rows if r[key] is not None]
    return statistics.median(vals) if vals else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--delayed-root', required=True)
    ap.add_argument('--immediate-root')
    ap.add_argument('--output', required=True)
    a = ap.parse_args()
    by_mode = {'delayed': [row for rank, path in selected_traces(a.delayed_root).items()
                           for row in sample(path, 'delayed')]}
    if a.immediate_root:
        by_mode['immediate'] = [row for rank, path in selected_traces(a.immediate_root).items()
                                for row in sample(path, 'immediate')]
    summary = {}
    for mode, rows in by_mode.items():
        summary[mode] = {
            'samples': len(rows),
            'all_gather_and_main_compute_overlap_samples': sum(r['device_overlap_us'] > 0 for r in rows),
            'median_gather_duration_us': median(rows, 'gather_duration_us'),
            'median_device_overlap_us': median(rows, 'device_overlap_us'),
            'median_gather_start_to_first_compressor_us': median(rows, 'gather_start_to_first_compressor_us'),
            'median_gather_start_to_next_alltoall_us': median(rows, 'gather_start_to_next_alltoall_us'),
            'median_common_predecessor_end_to_wkv_input_us': median(rows, 'common_predecessor_end_to_wkv_input_us'),
            'median_common_predecessor_end_to_first_compressor_us': median(rows, 'common_predecessor_end_to_first_compressor_us'),
            'median_common_predecessor_end_to_next_alltoall_us': median(rows, 'common_predecessor_end_to_next_alltoall_us'),
            'median_common_predecessor_end_to_next_alltoall_end_us': median(rows, 'common_predecessor_end_to_next_alltoall_end_us'),
            'median_gather_end_to_q_rope_end_us': median(rows, 'gather_end_to_q_rope_end_us'),
        }
    if 'immediate' in by_mode:
        summary['delayed_minus_immediate_medians_us'] = {
            key: summary['delayed'][key] - summary['immediate'][key]
            for key in ('median_gather_start_to_first_compressor_us',
                        'median_gather_start_to_next_alltoall_us',
                        'median_common_predecessor_end_to_wkv_input_us',
                        'median_common_predecessor_end_to_first_compressor_us',
                        'median_common_predecessor_end_to_next_alltoall_us',
                        'median_common_predecessor_end_to_next_alltoall_end_us')
        }
    result = {'summary': summary, 'rows': by_mode,
              'scope': 'device interval screen at first Compressor in each of two sampled cycles per rank; no same-state numerical proof, full-cycle attribution or formal TPS claim'}
    Path(a.output).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    main()
