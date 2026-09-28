#!/usr/bin/env python3
"""Run634: join Run606 Product calls with admitted Target dispatch modes.

This is current Host interval ownership, not Target device service or a Bound.
"""
from __future__ import annotations

import hashlib
import json
import statistics
from pathlib import Path
from loop081_dispatch_validate_run606 import validate_frontier

ROOT = Path('/data/wio/Inference_Foundry')
BASE = ROOT / 'evidence/20260928_loop081_bound/run606/live/b'
OUT = ROOT / 'evidence/20260928_loop081_bound/run634/prep_mode_coverage.json'


def read(p):
    return json.loads(p.read_text())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def summarize(values):
    if not values:
        return None
    return {'min_s': min(values), 'median_s': statistics.median(values),
            'max_s': max(values)}


def main():
    pa_path = BASE / 'product_admission.json'
    da_path = BASE / 'dispatch_admission.json'
    prep_path = ROOT / 'evidence/20260928_loop081_bound/run633/prep_host_coverage.json'
    pa, da, prep = read(pa_path), read(da_path), read(prep_path)
    if da['product_admission_sha256'] != sha(pa_path):
        raise AssertionError('dispatch/Product admission mismatch')
    if da['status'] != 'scoped_dispatch_identity_pass' or pa['status'] != 'product_identity_pass':
        raise AssertionError('admission status')
    if da['run_ts'] != pa['run_ts'] or len(da['cohorts']) != 4:
        raise AssertionError('run/cohort ownership')
    if len(prep['rows']) != 32:
        raise AssertionError('Run633 all8 prep coverage')
    if prep['source_sha256']['product_admission'] != sha(pa_path):
        raise AssertionError('Run633 Product admission source')
    if {c['cohort'] for c in da['cohorts']} != set(range(5, 9)):
        raise AssertionError('unique dispatch cohorts')
    prep_by_key = {(r['rank'], r['cohort']): r for r in prep['rows']}
    if len(prep_by_key) != 32 or set(prep_by_key) != {
        (rank, cohort) for rank in range(8) for cohort in range(5, 9)
    }:
        raise AssertionError('unique Run633 rank/cohort rows')
    owner = {r['req_id']: r['cohort'] for r in pa['join_rows']}
    if len(owner) != 48:
        raise AssertionError('request join')
    rows = []
    for c in da['cohorts']:
        cohort = c['cohort']
        if cohort not in range(5, 9) or len(c['ranks']) != 8:
            raise AssertionError('dispatch cohort/rank')
        for rank, d in enumerate(c['ranks']):
            path = BASE / f'product/rank{rank}_cohort{cohort}.json'
            if sha(path) != pa['raw_sha256'].get(str(path)) or sha(path) != da['raw_sha256'].get(str(path)):
                raise AssertionError('raw Product SHA')
            raw = read(path)
            if raw['rank'] != rank or raw['cohort'] != cohort or raw['run_ts'] != pa['run_ts']:
                raise AssertionError('raw owner')
            # Rebuild the complete semantic signature from raw frontier records.
            # Checking admission's mode histogram alone is not a source check.
            if json.loads(json.dumps(validate_frontier(raw))) != d:
                raise AssertionError('raw frontier/dispatch admission mismatch')
            marks = raw['marks']
            sig = d['semantic_signature']['forward']
            if len(sig) != d['ordinary_pairs']:
                raise AssertionError('forward count')
            by_id = {}
            for z in sig:
                count, mode, padded, graph_size, metadata = z
                if count in by_id or mode not in ('NONE', 'FULL') or not isinstance(padded, int):
                    raise AssertionError('dispatch schema')
                by_id[count] = z
            if {mode: sum(z[1] == mode for z in sig) for mode in ('NONE', 'FULL')} != d['forward_modes']:
                raise AssertionError('mode histogram')
            observed = []
            for call in raw['calls']:
                if call['kind'] != 'target_forward':
                    continue
                count = call['execute_mark_count']
                if count not in by_id or not 1 <= count < len(marks):
                    raise AssertionError('target/dispatch join')
                if any(m['kind'] != 'execute_entry' for m in marks[:count]):
                    raise AssertionError('count is not all-execute prefix')
                source = marks[count - 1]
                ids = [q['req_id'] for q in source['new'] + source['cached']]
                labels = {owner.get(rid) for rid in ids}
                if not ids or None in labels or labels - {cohort - 1, cohort}:
                    raise AssertionError('request owner')
                label = 'own' if labels == {cohort} else ('prior' if labels == {cohort - 1} else 'mixed')
                a, b = call['host_start_ns'], call['host_end_ns']
                if not source['t_ns'] <= a <= b <= marks[count]['t_ns']:
                    raise AssertionError('call interval')
                mode = by_id[count][1]
                if call['num_tokens_padded'] != by_id[count][2]:
                    raise AssertionError('padded shape')
                observed.append({'execute_mark_count': count, 'ownership': label,
                                 'mode': mode, 'padded': call['num_tokens_padded'],
                                 'host_duration_s': (b - a) / 1e9})
            observed_counts = [z['execute_mark_count'] for z in observed]
            if len(observed_counts) != len(set(observed_counts)) or set(observed_counts) != set(by_id):
                raise AssertionError('incomplete target calls')
            own_none = [z['host_duration_s'] for z in observed if z['ownership'] == 'own' and z['mode'] == 'NONE']
            own_full = [z['host_duration_s'] for z in observed if z['ownership'] == 'own' and z['mode'] == 'FULL']
            prior = [z for z in observed if z['ownership'] == 'prior']
            if cohort == 5 and prior or cohort > 5 and len(prior) != 1:
                raise AssertionError('prior carryover target count')
            prep_row = prep_by_key[(rank, cohort)]
            target_union = prep_row['exclusive_host_interval_occupancy_ns'].get('target_forward:own', 0) / 1e9
            if abs(target_union - (sum(own_none) + sum(own_full))) > 1e-7:
                raise AssertionError('Run633 union mismatch')
            rows.append({'rank': rank, 'cohort': cohort,
                         'own_NONE_count': len(own_none), 'own_FULL_count': len(own_full),
                         'own_NONE_target_forward_host_s': sum(own_none),
                         'own_FULL_target_forward_host_s': sum(own_full),
                         'prior_FULL_target_count': sum(z['mode'] == 'FULL' for z in prior),
                         'target_forward_own_host_total_s': target_union,
                         'calls': observed})
    if len(rows) != 32:
        raise AssertionError('all8 rows')
    summary = []
    for cohort in range(5, 9):
        selected = [r for r in rows if r['cohort'] == cohort]
        summary.append({
            'cohort': cohort,
            'all8_own_NONE_count': [r['own_NONE_count'] for r in selected],
            'all8_own_FULL_count': [r['own_FULL_count'] for r in selected],
            'per_rank_own_NONE_host_sum_s': summarize([r['own_NONE_target_forward_host_s'] for r in selected]),
            'per_rank_own_FULL_host_sum_s': summarize([r['own_FULL_target_forward_host_s'] for r in selected]),
        })
    result = {
        'status': 'run606_same_W0_current_Target_Host_dispatch_mode_join_only',
        'source_sha256': {'product_admission': sha(pa_path), 'dispatch_admission': sha(da_path),
                          'dispatch_validator': sha(ROOT / 'scripts/loop081_dispatch_validate_run606.py'),
                          'run633_prep': sha(prep_path)},
        'contract': 'fixed DSpark7 acceptance/cycles/output/model work; 8x910B3 DP1TP8 48x32K->1024 c12',
        'rows': rows, 'summary': summary,
        'scope': 'Host Python target_forward call interval; NONE/FULL source dispatch identity under Run606 observer. FULL host call may include existing Graph pre-replay synchronize; none of these durations certifies device completion, HCCL, exposed critical path, compulsory work or removable wall. Distinct Run99 formal W0 not transferable.',
        'strict_resource_floor_s': None, 'strict_scheduling_floor_s': None,
        'product_e2e_ceiling_tps': None, 'numeric_current_to_credible_limit_gap': None,
        'next': 'Measure same-W0 Target prefill producer completion, Graph/HCCL side-stream readiness and first actual consumer at broad phase boundaries with low-overhead marker transfer; keep DSpark and KV links conditional.',
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'status': result['status'], 'rows': len(rows), 'sha256': sha(OUT)}))


if __name__ == '__main__':
    main()
