#!/usr/bin/env python3
"""Rejoin six frozen communication samples to strict draft windows, locally only."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
paths = {
    'distribution': ROOT / 'agents/diagnosis/16rank-draft-alltoall-distribution.json',
    'identity': ROOT / 'agents/diagnosis/24-draft-alltoall-identity.json',
    'strict_mc2': ROOT / 'draft-mc2-analysis/strict_draft_calls.csv',
    'strict_calls': ROOT / 'eager-draft-mainstream-attribution-v2/draft_call_metrics.json',
}
digests = {k: hashlib.sha256(p.read_bytes()).hexdigest() for k, p in paths.items()}
dist = json.loads(paths['distribution'].read_text())
identities = json.loads(paths['identity'].read_text())
calls = json.loads(paths['strict_calls'].read_text())
with paths['strict_mc2'].open(newline='') as f:
    mc2 = list(csv.DictReader(f))

output = []
targets = [x for x in identities['samples'] if '_ep0_' in x['rank'] or '_ep1_' in x['rank']]
assert len(targets) == 6
for x in targets:
    bracket_hits = [(b, h) for b in dist['brackets'] if b['rank'] == x['rank']
                    for h in b['alltoall'] if len(h['exact_task_matches']) == 1
                    and h['exact_task_matches'][0]['globalTaskId'] == x['selected_task_global_id']]
    assert len(bracket_hits) == 1
    b, h = bracket_hits[0]
    m = [r for r in mc2 if r['rank'] == x['rank'] and int(r['bracket_start_ns']) == b['bracket_start_ns']]
    c = [r for r in calls if r['rank'] == x['rank'] and r['start_ns'] == b['bracket_start_ns']]
    assert len(m) == len(c) == 1
    m, c = m[0], c[0]
    s, e = int(m['bracket_start_ns']), int(m['bracket_end_ns'])
    cs, ce = h['alltoall_start_ns'], h['alltoall_end_ns']
    ms, me = int(m['mc2_start_ns']), int(m['mc2_end_ns'])
    assert c['end_ns'] == e and c['main_stream_id'] == b['main_stream']
    assert s <= cs < ce <= ms < me <= e
    assert b['bracket_start_ns'] <= b['attention_end_ns'] <= e
    segments = [cs-s, ce-cs, ms-ce, me-ms, e-me]
    assert sum(segments) == e-s
    output.append({'rank': x['rank'], 'local_op_id': x['opId_local'],
                   'rank_local_window_start_ns': s, 'rank_local_window_end_ns': e,
                   'comm_start_ns': cs, 'comm_end_ns': ce, 'mc2_start_ns': ms, 'mc2_end_ns': me,
                   'main_stream': c['main_stream_id'], 'mc2_task_id': c['mc2_task_id'],
                   'window_us': (e-s)/1000,
                   'start_to_comm_start_us': segments[0]/1000, 'comm_us': segments[1]/1000,
                   'comm_end_to_mc2_start_us': segments[2]/1000, 'mc2_us': segments[3]/1000,
                   'mc2_end_to_window_end_us': segments[4]/1000,
                   'mc2_aic_ratio_pct': 100*float(m['mc2_aic_num'])/float(m['mc2_aic_den'])})

assert sorted((x['rank'].split('_ep')[1].split('_')[0],x['local_op_id']) for x in output) == [('0',402),('0',405),('0',408),('1',402),('1',405),('1',408)]
print(json.dumps({'inputs_sha256': digests, 'query_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  'scope': 'six previously selected same-rank exported intervals, no cross-rank clock comparison',
                  'units': 'integer ns on each rank local timeline, displayed in us',
                  'samples': output,
                  'limitations': ['Intervals outside the two named tasks are not idle time and may include tasks and gaps.',
                                  'Same-rank containment and ordering do not prove a data or synchronization dependency.',
                                  'Across ranks, same operation name is not proof of the same request or collective.',
                                  'Elapsed task time is not pure network or pure compute time.',
                                  'MC2 counters are exporter proxies; actual routed work and counter semantics are unknown.']}, ensure_ascii=False, indent=2))
