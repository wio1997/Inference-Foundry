"""Validate all-rank ownership census on a complete eager diagnostic carrier."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    root = parser.parse_args().run_dir
    bench = json.loads((root / 'bench12.json').read_text())
    if bench['summary']['success'] != 12 or any(
            item['output_tokens'] != 1024 for item in bench['requests']):
        raise RuntimeError('12x1024 diagnostic carrier incomplete')
    reports = []
    for rank in range(8):
        paths = sorted((root / 'fixture').glob(f'rank{rank}_cohort*.json'))
        if len(paths) != 1:
            raise RuntimeError(f'rank{rank} expected one census report, got {paths}')
        item = json.loads(paths[0].read_text())
        if item.get('stage') != 'complete' or item['rank'] != rank:
            raise RuntimeError(f'rank{rank} census failed: {item}')
        reports.append(item)
    if len({(x['cycle'], tuple(x['active_mask'])) for x in reports}) != 1:
        raise RuntimeError('ranks disagree on cycle or active mask')
    if any(x['context_num_tokens'] != 96 or x['hidden_shape'] != [12, 4096]
           for x in reports):
        raise RuntimeError('Target context and local shape are not 96/12')
    rows = [row for item in reports for row in item['global_rows']]
    if rows != list(range(96)):
        raise RuntimeError('TP8 contiguous row ownership invalid')
    mask = reports[0]['active_mask']
    expected = [bool(mask[row // 8]) for row in rows]
    got = [flag for item in reports for flag in item['local_active_flags']]
    if got != expected or sum(got) != 48:
        raise RuntimeError('active row ownership invalid')
    print(json.dumps({
        'status': 'read_only_ownership_census_pass',
        'scope': 'eager diagnostic, no candidate or Product E2E result',
        'cycle': reports[0]['cycle'], 'active_mask': mask,
        'global_tokens': 96, 'active_global_tokens': 48,
        'local_active_rows': [item['local_active_rows'] for item in reports],
        'module_is_sequence_parallel': [item['module_is_sequence_parallel'] for item in reports],
        'moe_comm_type': [item['context_moe_comm_type'] for item in reports],
        'flash_comm_v1': [item['context_flash_comm_v1_enabled'] for item in reports],
    }, indent=2))


if __name__ == '__main__':
    main()
