#!/usr/bin/env python3
"""Reuse three historical single-cohort full-cycle Event profiles as cost priors.

No Product or fixed-primitive Bound transfer is made by this reducer.
"""
import hashlib
import json
import math
import statistics
from pathlib import Path

ROOT = Path('/data/wio/Inference_Foundry')
OUT = ROOT / 'evidence/20260928_loop081_bound/run653/fullcycle_stage_prior.json'
PIN_PATH = ROOT / 'evidence/20260928_loop081_bound/run653/input_pins.json'
CASES = {
    87: ROOT / 'evidence/20260924_loop035_diagnostic/run87',
    98: ROOT / 'evidence/20260924_loop036_metadata/run98',
    367: ROOT / 'evidence/20260927_loop076_bound/run367',
}
WINDOWS = ((0, 8), (8, 64), (64, 128), (128, 192),
           (192, 256), (256, 1025))

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def dist(values):
    assert values
    s = sorted(values)
    return {'n': len(s), 'min_ms': s[0], 'p10_ms': s[int(.1*(len(s)-1))],
            'median_ms': statistics.median(s),
            'p90_ms': s[int(.9*(len(s)-1))], 'max_ms': s[-1]}

def one(run, base):
    bench = base / 'bench.json'
    client = json.loads(bench.read_text())['summary']
    assert client['n'] == client['success'] == client['concurrency'] == 12
    assert client['fail'] == 0 and client['max_tokens'] == 1024
    dags = []
    pin = {str(bench.relative_to(ROOT)): sha(bench)}
    for rank in range(8):
        path = base / f'dag/rank{rank}.json'
        x = json.loads(path.read_text())
        assert x['rank'] == rank
        assert len(x['runtime_stage_ms']) == len(x['dspark_stage_ms']) == x['cycles']
        assert all(set(v) == {'prepare_target', 'derived_target_metadata',
                              'target', 'acceptance', 'state_advance',
                              'proposer', 'draft_commit'} for v in x['runtime_stage_ms'])
        assert all(set(v) == {'host_mirror', 'refresh_common', 'prepare_inputs',
                              'pack_hidden', 'model'} for v in x['dspark_stage_ms'])
        assert all(math.isfinite(value) and value >= 0
                   for row in x['runtime_stage_ms'] + x['dspark_stage_ms']
                   for value in row.values())
        pin[str(path.relative_to(ROOT))] = sha(path)
        dags.append(x)
    assert len({x['cycles'] for x in dags}) == 1
    ncycle = dags[0]['cycles']
    windows = []
    for lo, hi in WINDOWS:
        hi = min(hi, ncycle)
        if lo >= hi:
            continue
        rows = [(rank, cycle, x['runtime_stage_ms'][cycle])
                for rank, x in enumerate(dags) for cycle in range(lo, hi)]
        # Run367 has an additional native profiler and explicit sync at
        # nominal cycles64–66. Keep this as a separate excluded stratum.
        excluded = [r for r in rows if run == 367 and 64 <= r[1] <= 66]
        rows = [r for r in rows if r not in excluded]
        windows.append({
            'cycle_window': [lo, hi], 'excluded_profiler_rank_cycles': len(excluded),
            'target_current_stream': dist([r[2]['target'] for r in rows]),
            'proposer_current_stream': dist([r[2]['proposer'] for r in rows]),
            'paired_target_plus_proposer': dist(
                [r[2]['target'] + r[2]['proposer'] for r in rows]),
        })
    normal = [w for w in windows if w['cycle_window'][0] >= 8]
    target_medians = [w['target_current_stream']['median_ms'] for w in normal]
    proposer_medians = [w['proposer_current_stream']['median_ms'] for w in normal]
    per_rank = []
    for rank, x in enumerate(dags):
        active = [row for cycle, row in enumerate(x['runtime_stage_ms'])
                  if cycle >= 8 and not (run == 367 and 64 <= cycle <= 66)]
        per_rank.append({'rank': rank,
                         'target_median_ms': statistics.median(row['target'] for row in active),
                         'proposer_median_ms': statistics.median(row['proposer'] for row in active)})
    conservative_exclusion_delta = None
    if run == 367:
        standard = [row['target'] for x in dags for cycle, row in enumerate(x['runtime_stage_ms'])
                    if cycle >= 8 and not 64 <= cycle <= 66]
        wide = [row['target'] for x in dags for cycle, row in enumerate(x['runtime_stage_ms'])
                if cycle >= 8 and not 63 <= cycle <= 67]
        conservative_exclusion_delta = statistics.median(wide) - statistics.median(standard)
    controllers = {
        87: ROOT / 'scripts/run_loop035_decode_dag_profile_run87.sh',
        98: ROOT / 'scripts/run_loop036_metadata_profile.sh',
        367: ROOT / 'scripts/run_loop076_joint_profile.sh',
    }
    controller = controllers[run]
    return {
        'run': run, 'cycles': ncycle,
        'client_diagnostic_wall_s': client['duration_s'],
        'client_diagnostic_tps': client['output_tps'],
        'input_sha256': pin,
        'controller_source': str(controller.relative_to(ROOT)),
        'controller_source_sha256': sha(controller),
        'windows': windows,
        'normal_per_rank': per_rank,
        'run367_target_median_change_excluding_63_to_67_instead_of_64_to_66_ms':
            conservative_exclusion_delta,
        'normal_window_target_median_range_ms': [min(target_medians), max(target_medians)],
        'normal_window_proposer_median_range_ms': [min(proposer_medians), max(proposer_medians)],
        'limitation': 'Single 12-request historical diagnostic W0 with full-cycle Events; current-stream stage time includes queue/wait/observer and is not fixed intrinsic primitive cost or Run606 Product transfer.',
    }

def main():
    pinned = json.loads(PIN_PATH.read_text())
    for rel, expected in pinned.items():
        assert sha(ROOT / rel) == expected, rel
    rows = [one(run, base) for run, base in CASES.items()]
    for row in rows:
        assert all(pinned[rel] == digest for rel, digest in row['input_sha256'].items())
        assert pinned[row['controller_source']] == row['controller_source_sha256']
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        'status': 'historical_fullcycle_stage_prior_only',
        'cases': rows,
        'scope': 'Three separate single-cohort W0s, not combined or extrapolated to Run606/Run99. Run367 nominal profiler cycles64–66 excluded from window summaries.',
        'framework_only_product_tps_bound': None,
    }, indent=2) + '\n')
    print(json.dumps([{'run': x['run'], 'cycles': x['cycles'],
                       'target_normal_window_medians_ms': x['normal_window_target_median_range_ms'],
                       'proposer_normal_window_medians_ms': x['normal_window_proposer_median_range_ms']}
                      for x in rows], indent=2))

if __name__ == '__main__':
    main()
