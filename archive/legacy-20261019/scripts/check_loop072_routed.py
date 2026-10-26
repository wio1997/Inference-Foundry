"""Fail-closed carrier and A/A controls for one-layer post-gather routed screen."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    root = parser.parse_args().run_dir
    bench = json.loads((root / 'bench12.json').read_text())
    if bench['summary']['success'] != 12 or any(
            request['output_tokens'] != 1024 for request in bench['requests']):
        raise RuntimeError('12x1024 eager diagnostic carrier incomplete')
    runtime_paths = sorted((root / 'runtime').glob('rank*_cohort*.json'))
    if len(runtime_paths) != 8:
        raise RuntimeError(f'expected 8 Runtime reports, got {len(runtime_paths)}')
    runtime = [json.loads(path.read_text()) for path in runtime_paths]
    if {x['rank'] for x in runtime} != set(range(8)) or not all(
            x['pass'] and x['target_graph_mode'] == 'NONE' for x in runtime):
        raise RuntimeError('8-rank eager Runtime carrier failed')
    reports = []
    for rank in range(8):
        paths = sorted((root / 'fixture').glob(f'rank{rank}_cohort*.json'))
        if len(paths) != 1:
            raise RuntimeError(f'rank{rank} expected one fixture, got {paths}')
        item = json.loads(paths[0].read_text())
        if item.get('stage') != 'complete' or item['rank'] != rank:
            raise RuntimeError(f'rank{rank} fixture failed: {item}')
        if (not item['A_A_repeat']['exact'] or
                not item['A_A2']['exact'] or
                not item['A_A_repeat']['finite_reference'] or
                not item['B_active_vs_A']['finite_candidate'] or
                not item['B_all_finite'] or
                not item['outer_context_restored'] or
                item['apply']['full_input'] != [96, 4096] or
                item['apply']['compact_input'] != [48, 4096] or
                item['apply']['routed_restored'] != [96, 4096] or
                not item['tid2eid_is_none']):
            raise RuntimeError(f'rank{rank} controls or compact apply gate failed')
        reports.append(item)
    if len({(x['cohort'], x['cycle'], tuple(x['active_mask'])) for x in reports}) != 1:
        raise RuntimeError('rank cycle/active mask mismatch')
    if sum(x['B_active_vs_A']['rows'] for x in reports) != 48:
        raise RuntimeError('active local output row total is not 48')
    if [x['B_active_vs_A']['rows'] for x in reports] != [
            sum(x['active_mask'][i // 8] for i in range(rank * 12, (rank + 1) * 12))
            for rank, x in enumerate(reports)]:
        raise RuntimeError('local active row mapping mismatch')
    print(json.dumps({
        'status': 'private_routed_semantic_screen_complete',
        'scope': 'eager same-prestate layer4, no Graph or Product TPS claim',
        'cycle': reports[0]['cycle'],
        'active_mask': reports[0]['active_mask'],
        'local_active_rows': [x['B_active_vs_A']['rows'] for x in reports],
        'B_nonempty_ranks': sum(x['B_active_vs_A']['rows'] > 0 for x in reports),
        'B_exact_nonempty_ranks': sum(x['B_active_vs_A']['exact']
                                      for x in reports if x['B_active_vs_A']['rows'] > 0),
        'B_close_1e-3_ranks': sum(x['B_active_vs_A']['allclose_atol1e-3_rtol1e-3']
                                 for x in reports),
        'B_close_1e-2_ranks': sum(x['B_active_vs_A']['allclose_atol1e-2_rtol1e-2']
                                 for x in reports),
        'B_max_abs_by_rank': [x['B_active_vs_A']['max_abs'] for x in reports],
    }, indent=2))


if __name__ == '__main__':
    main()
