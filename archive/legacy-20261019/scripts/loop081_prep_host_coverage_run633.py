#!/usr/bin/env python3
"""Run633: classify current Host call interval coverage in Run606 preparation.

No cross-W0 timing transfer, physical device occupancy or Bound promotion.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path('/data/wio/Inference_Foundry')
BASE = ROOT / 'evidence/20260928_loop081_bound/run606/live/b'
OUT = ROOT / 'evidence/20260928_loop081_bound/run633/prep_host_coverage.json'


def read(p):
    return json.loads(p.read_text())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def seconds(ns):
    return ns / 1e9


def classify(mark, owner, current):
    ids = [row['req_id'] for row in mark['new'] + mark['cached']]
    if not ids:
        return 'no_request_operand'
    labels = {owner.get(rid) for rid in ids}
    if None in labels or labels - {current - 1, current}:
        raise AssertionError(('unknown request', labels))
    if labels == {current}:
        return 'own'
    if labels == {current - 1}:
        return 'prior'
    return 'mixed_prior_own'


def rank_cohort(path, expected_sha, run_ts, time_namespace, owner, cohort, expected_rank):
    if sha(path) != expected_sha:
        raise AssertionError('raw SHA')
    x = read(path)
    if (x['run_ts'] != run_ts or x['cohort'] != cohort
            or x['rank'] != expected_rank
            or x['time_namespace'] != time_namespace):
        raise AssertionError('run/cohort')
    marks = x['marks']
    if any(m.get('rank', x['rank']) != x['rank'] for m in marks):
        raise AssertionError('marker rank')
    execs = [m for m in marks if m['kind'] == 'execute_entry']
    built = [m for m in marks if m['kind'] == 'runtime_built_host']
    if not execs or len(built) != 1:
        raise AssertionError('markers')
    first_record = min(m['t_ns'] for m in execs)
    first_own = min(m['t_ns'] for m in execs if classify(m, owner, cohort) == 'own')
    last_built = built[0]['t_ns']
    if not first_record <= first_own < last_built:
        raise AssertionError('boundary order')
    calls = []
    for row in x['calls']:
        count = row['execute_mark_count']
        if not 1 <= count < len(marks):
            raise AssertionError('call owner index')
        # The patch stores len(all marks), not len(filtered execute marks).
        # This raw packet happens to have only execute marks before each call.
        if any(m['kind'] != 'execute_entry' for m in marks[:count]):
            raise AssertionError('non-execute marker before call')
        source_mark = marks[count - 1]
        label = classify(source_mark, owner, cohort)
        if row['kind'] not in ('target_forward', 'propose_and_optional_copy'):
            raise AssertionError('call kind')
        a, b = row['host_start_ns'], row['host_end_ns']
        if not source_mark['t_ns'] <= a <= b <= marks[count]['t_ns']:
            raise AssertionError('call outside its execute marker')
        if not first_record <= a <= b <= last_built:
            raise AssertionError('call interval outside packet')
        if row['kind'] == 'propose_and_optional_copy':
            raw_ids = row['req_ids']
            mark_ids = [q['req_id'] for q in source_mark['new'] + source_mark['cached']]
            if Counter(raw_ids) != Counter(mark_ids):
                raise AssertionError('proposal request ownership')
        calls.append((a, b, row['kind'], label))
    points = sorted({first_own, last_built} | {
        max(first_own, min(last_built, v)) for a, b, _, _ in calls for v in (a, b)})
    covered = 0
    uncovered = 0
    overlap = 0
    by_class = {}
    for a, b in zip(points, points[1:]):
        active = [(kind, label) for ca, cb, kind, label in calls if ca <= a and b <= cb and cb > ca]
        dur = b - a
        if active:
            covered += dur
            if len(active) > 1:
                overlap += dur
            key = '|'.join(sorted({f'{k}:{l}' for k, l in active}))
        else:
            uncovered += dur
            key = 'uncovered_by_target_propose_host_calls'
        by_class[key] = by_class.get(key, 0) + dur
    if covered + uncovered != last_built - first_own:
        raise AssertionError('coverage conservation')
    return {
        'rank': x['rank'], 'cohort': cohort,
        'execute_marker_counts': {z: sum(classify(m, owner, cohort) == z for m in execs)
                                  for z in ('prior', 'no_request_operand', 'own', 'mixed_prior_own')},
        'call_counts': {f'{kind}:{label}': sum(k == kind and l == label for _, _, k, l in calls)
                        for kind in ('target_forward', 'propose_and_optional_copy')
                        for label in ('prior', 'no_request_operand', 'own', 'mixed_prior_own')},
        'own_first_to_built_host_s': seconds(last_built - first_own),
        'first_recorded_to_own_host_s': seconds(first_own - first_record),
        'known_target_propose_host_union_within_own_to_built_s': seconds(covered),
        'uncovered_host_interval_s': seconds(uncovered),
        'target_propose_host_overlap_s': seconds(overlap),
        'exclusive_host_interval_occupancy_ns': by_class,
        'call_count': len(calls),
        'scope': 'Host Python call intervals only; current-stream event elapsed omitted because it is not an absolute completion or disjoint service sum',
    }


def main():
    admission = read(BASE / 'product_admission.json')
    client = BASE / 'client_admission.json'
    if admission['status'] != 'product_identity_pass':
        raise AssertionError('product admission')
    if sha(client) != admission['raw_sha256'].get(str(client)):
        raise AssertionError('client SHA')
    client_namespace = read(client)['measured_summary']['clock']['time_namespace']
    owner = {r['req_id']: r['cohort'] for r in admission['join_rows']}
    if len(owner) != 48:
        raise AssertionError('request join cardinality')
    rows = []
    for cohort in range(5, 9):
        for rank in range(8):
            path = BASE / f'product/rank{rank}_cohort{cohort}.json'
            rows.append(rank_cohort(path, admission['raw_sha256'][str(path)],
                                    admission['run_ts'], client_namespace, owner,
                                    cohort, rank))
    if len(rows) != 32:
        raise AssertionError('all8 four cohorts')
    result = {
        'status': 'run606_same_W0_preparation_host_call_coverage_only',
        'contract': 'fixed DSpark7 acceptance/cycles/output/model work; 8x910B3 DP1TP8 48x32K->1024 c12',
        'source_sha256': {
            'product_admission': sha(BASE / 'product_admission.json'),
            'client_admission': sha(client),
            'run631_coverage': sha(ROOT / 'evidence/20260928_loop081_bound/run631/product_coverage.json'),
        },
        'rows': rows,
        'interpretation': [
            'Current Host target_forward/propose call intervals are tagged by exact same-run request ownership, including prior-cohort and no-request markers.',
            'Within each rank/cohort, intervals are unioned rather than summing calls. The complement is unclassified Host envelope, not idle time or recoverable wall.',
            'Different ranks execute concurrently; no sum across ranks or cohorts is a Product critical path. Host call time is not device completion, HCCL service, or compulsory model work.',
            'Run606 observer is perturbed and differs from Run99 formal W0; no timing transfer or numerical Bound follows.',
        ],
        'strict_resource_floor_s': None,
        'strict_scheduling_floor_s': None,
        'product_e2e_ceiling_tps': None,
        'numeric_current_to_credible_limit_gap': None,
        'next': 'Use the uncovered/overlap distributions to choose a bounded same-W0 producer-completion and existing-wait packet, rather than timing a single KV row or interpreting current-stream events as full device readiness.',
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'status': result['status'], 'rows': len(rows), 'sha256': sha(OUT)}))


if __name__ == '__main__':
    main()
