#!/usr/bin/env python3
"""Rehash Run606 admissions and reduce scoped all-rank dispatch observations."""
import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from loop081_dispatch_validate_run606 import validate_frontier


def load(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_hashes(rows, root):
    for path, expected in rows.items():
        item = Path(path)
        assert digest(item if item.is_absolute() else root / item) == expected, path


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument('--arm-dir', required=True, type=Path)
    cli.add_argument('--output', required=True, type=Path)
    args = cli.parse_args()
    arm = args.arm_dir.resolve()
    live = arm.parent
    repo = arm.parents[4]
    basis = load(arm / 'basis_admission.json')
    product = load(arm / 'product_admission.json')
    dispatch = load(arm / 'dispatch_admission.json')
    assert basis['status'] == 'diagnostic_compact_basis_all8_admitted'
    assert product['status'] == 'product_identity_pass'
    assert dispatch['status'] == 'scoped_dispatch_identity_pass'
    assert {basis['run_ts'], product['run_ts'], dispatch['run_ts']} == {'LOOP081-RUN606-B'}
    assert len(product['raw_sha256']) == 281
    assert len(dispatch['raw_sha256']) == 64
    assert len(basis['source_sha256']) == 97
    check_hashes(basis['source_sha256'], repo)
    check_hashes(product['raw_sha256'], repo)
    check_hashes(dispatch['raw_sha256'], repo)
    assert dispatch['product_admission_sha256'] == digest(arm / 'product_admission.json')
    assert (live / 'source_before.sha256').read_bytes() == (live / 'source_after.sha256').read_bytes()
    assert (live / 'scripts_before.sha256').read_bytes() == (live / 'scripts_after.sha256').read_bytes()
    cleanup = dict(line.split('=', 1) for line in (live / 'cleanup_status.txt').read_text().splitlines())
    assert set(cleanup) == {'run_exit', 'stop_exit', 'stop_verify_exit',
                            'restore_exit', 'source_sha_exit',
                            'source_compare_exit', 'script_sha_exit',
                            'script_compare_exit', 'final_exit'}
    assert set(cleanup.values()) == {'0'}
    assert (arm / 'server_post_count.txt').read_text().strip() == '96'
    assert (arm / 'server_post_count_after_stop.txt').read_text().strip() == '96'

    per_rank = []
    per_cohort = []
    for cohort in range(5, 9):
        cohort_rows = []
        for rank in range(8):
            worker = load(arm / 'product' / f'rank{rank}_cohort{cohort}.json')
            assert (worker['rank'], worker['cohort'], worker['run_ts']) == (
                rank, cohort, 'LOOP081-RUN606-B')
            stats = validate_frontier(worker)
            frontier = worker['frontier']
            replay = [g for g in frontier['graphs'] if g['branch'] == 'replay']
            sync_ms = sum((g['sync_return_ns'] - g['sync_enter_ns']) / 1e6
                          for g in replay if g['sync_taken'])
            last = frontier['context_stores'][-1]
            same_stream = (last['stream'] == frontier['first_target_stream'] ==
                           frontier['first_draft_stream'])
            assert same_stream
            assert all(math.isfinite(x) and x >= 0 for x in
                       (last['to_first_target_ms'], last['to_first_draft_ms']))
            classes = Counter()
            for fwd in frontier['forward']:
                classes[(fwd['mode'], fwd['attention'][0]['num_prefills'] > 0)] += 1
                assert len(fwd['attention']) == 8
                assert all(row == fwd['attention'][0] for row in fwd['attention'])
            cohort_rows.append({
                'rank': rank, 'ordinary_pairs': stats['ordinary_pairs'],
                'modes': stats['forward_modes'], 'graph': stats['graph_branches'],
                'existing_sync_count': stats['existing_replay_sync_count'],
                'existing_sync_host_ms': sync_ms,
                'metadata_classes': {f'{mode}:{"prefill" if pref else "decode"}': count
                                     for (mode, pref), count in classes.items()},
                'context_rows': stats['draft_context_rows'],
                'query_rows': stats['draft_query_rows'],
                'last_context_to_target_entry_ms': last['to_first_target_ms'],
                'last_context_to_draft_entry_ms': last['to_first_draft_ms'],
                'same_current_stream': same_stream,
                'semantic_signature': stats['semantic_signature'],
            })
        assert all(r['semantic_signature'] == cohort_rows[0]['semantic_signature']
                   for r in cohort_rows)
        for row in cohort_rows:
            del row['semantic_signature']
        per_rank.extend(cohort_rows)
        per_cohort.append({'cohort': cohort, 'allrank': cohort_rows})
    rank0 = [r for r in per_rank if r['rank'] == 0]
    result = {
        'status': 'scoped_run606_reduction_pass',
        'scope': 'Observed same-W0 ordinary dispatch/current-stream markers only; no compulsory work, side-stream readiness, perturbation control or finite Bound.',
        'admission_sha256': {name: digest(arm / f'{name}_admission.json')
                             for name in ('basis', 'product', 'dispatch')},
        'source_sha256_manifest': digest(live / 'source_before.sha256'),
        'script_sha256_manifest': digest(live / 'scripts_before.sha256'),
        'cleanup': cleanup,
        'cohorts': per_cohort,
        'rank0_totals': {
            'ordinary_pairs': sum(r['ordinary_pairs'] for r in rank0),
            'context_rows': sum(r['context_rows'] for r in rank0),
            'query_rows': sum(r['query_rows'] for r in rank0),
            'existing_full_replay_syncs': sum(r['existing_sync_count'] for r in rank0),
            'existing_sync_host_ms': sum(r['existing_sync_host_ms'] for r in rank0),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'status': result['status'], 'rank0_totals': result['rank0_totals']}))


if __name__ == '__main__':
    main()
