#!/usr/bin/env python3
"""Admit one 48-request ON diagnostic and reduce bounded prior Event queries."""
from __future__ import annotations
import argparse
import json
import statistics
from pathlib import Path

from scripts.loop081_runtime_observer_validate_run638 import arm, need, read


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args()
    base = arm(a.root / 'on', 'LOOP081-RUN660-ON', True)
    counts = {'cycle_begin': 0, 'prepare_target': 0, 'target_before': 0}
    lag_ms = []
    query_us = []
    for path in sorted((a.root / 'on/packet').glob('rank*_cohort*.json')):
        packet = read(path)
        rows = packet['query_rows']
        need(len(rows) == 12, 'exact two-transition query cardinality')
        events = {(e['cycle'], e['label']): e for e in packet['events']
                  if e['stream'] == 'current'}
        by = {(r['cycle'], r['point'], r['source']): r for r in rows}
        need(len(by) == 12, 'unique query identities')
        for cycle in (64, 65):
            for point in counts:
                for source in ('proposer_after', 'draft_commit_after'):
                    row = by[cycle, point, source]
                    need(row['source_cycle'] == cycle - 1 and
                         row['query_end_ns'] >= row['query_begin_ns'] >=
                         events[cycle, point]['host_submit_ns'],
                         'query generation/Host order')
                    query_us.append((row['query_end_ns']-row['query_begin_ns'])/1000)
                ready = by[cycle, point, 'draft_commit_after']
                if ready['completed']:
                    counts[point] += 1
                    next_issue = events[cycle, 'target_before']['host_submit_ns']
                    if point == 'cycle_begin':
                        need(next_issue >= ready['query_end_ns'], 'negative Host issue lag')
                        lag_ms.append((next_issue-ready['query_end_ns'])/1e6)
    need(len(base['packet_summary']) == 8*4*3 and len(query_us) == 8*4*12,
         'all-rank packet completeness')
    result = {
        'status': 'one_arm_rank_local_query_diagnostic_admitted',
        'source_run': 'run660', 'run_ts': base['run_ts'],
        'all8_cohorts': 4, 'rank_cycle_transitions': 8*4*2,
        'prior_draft_commit_completed_by_point_count': counts,
        'cycle_begin_completed_to_target_issue_host_lag_ms_median':
            statistics.median(lag_ms) if lag_ms else None,
        'query_duration_us_median': statistics.median(query_us),
        'query_duration_us_max': max(query_us),
        'limits': 'True means prior current-stream marker completed by query end. '
                  'No side-stream/HCCL/resource readiness, observer-free wall, '
                  'removable interval, or numeric Framework Bound is certified.',
    }
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
