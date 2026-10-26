#!/usr/bin/env python3
"""Reduce admitted Run661 same-W0 prior-event queries without wall transfer."""
from __future__ import annotations
import argparse
import hashlib
import json
import statistics
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    admitted = json.loads((a.root / 'admission.json').read_text())
    assert admitted['status'] == 'one_arm_rank_local_query_diagnostic_admitted'
    paths = sorted((a.root / 'on/packet').glob('rank*_cohort*.json'))
    assert len(paths) == 32
    transition = {}
    hashes = {}
    query_durations = []
    host_lags = []
    event_lags = []
    for path in paths:
        x = json.loads(path.read_text())
        hashes[str(path)] = sha(path)
        ident = x['identity']
        rank, cohort = ident['rank'], ident['cohort']
        assert x['sample_cycles'] == [63,64,65]
        assert len(x['query_rows']) == 12 and len(x['typed_rows']) >= 24
        events = {(e['cycle'], e['label']): e for e in x['events']
                  if e['stream'] == 'current'}
        queries = {(q['cycle'], q['point'], q['source']): q
                   for q in x['query_rows']}
        assert len(queries) == 12
        for q in x['query_rows']:
            query_durations.append((q['query_end_ns']-q['query_begin_ns'])/1000)
        for cycle in (64,65):
            checks = {point: queries[cycle, point, 'draft_commit_after']
                      for point in ('cycle_begin','prepare_target','target_before')}
            assert not checks['cycle_begin']['completed'] or checks['prepare_target']['completed']
            assert not checks['prepare_target']['completed'] or checks['target_before']['completed']
            before = events[cycle,'target_before']
            prior = events[cycle-1,'proposer_after']
            host_lag = (before['host_submit_ns']-prior['host_submit_ns'])/1e6
            event_lag = (before['elapsed_ms_from_first_same_stream']-
                         prior['elapsed_ms_from_first_same_stream'])
            host_lags.append(host_lag)
            event_lags.append(event_lag)
            transition.setdefault((cohort,cycle),[]).append({
                'rank': rank, 'target_before_host_ns': before['host_submit_ns'],
                'ready_at_begin': checks['cycle_begin']['completed'],
                'ready_at_prepare': checks['prepare_target']['completed'],
                'ready_at_target_issue': checks['target_before']['completed'],
                'host_proposer_to_target_ms': host_lag,
                'current_stream_event_proposer_to_target_ms': event_lag,
            })
    assert len(transition) == 8 and all(len(v)==8 for v in transition.values())
    latest=[]
    for (cohort, cycle), rows in sorted(transition.items()):
        latest.append({'cohort':cohort,'cycle':cycle,
                       **max(rows,key=lambda r:r['target_before_host_ns']),
                       'all8_ready_at_target_issue':sum(r['ready_at_target_issue'] for r in rows)})
    result={
        'status':'admitted_same_W0_current_stream_query_only',
        'raw_packet_sha256':hashes,
        'rank_cycle_transitions':64,
        'ready_count':{point:sum(r[f'ready_at_{point}'] for rows in transition.values() for r in rows)
                       for point in ('begin','prepare','target_issue')},
        'latest_host_rank_by_transition':latest,
        'latest_host_rank_ready_at_target_issue':sum(r['ready_at_target_issue'] for r in latest),
        'host_proposer_to_target_ms_median':statistics.median(host_lags),
        'current_stream_event_proposer_to_target_ms_median':statistics.median(event_lags),
        'query_duration_us_median':statistics.median(query_durations),
        'query_duration_us_max':max(query_durations),
        'limits':'No Event completion time, other-stream or native resource readiness, observer-free '
                 'wall, removable time or numeric Framework bound follows from these queries.'
    }
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('status','ready_count','latest_host_rank_ready_at_target_issue',
                                          'host_proposer_to_target_ms_median',
                                          'current_stream_event_proposer_to_target_ms_median')}))


if __name__=='__main__':
    main()
