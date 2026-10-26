#!/usr/bin/env python3
"""Replay rank-local observed task intervals in Run611's selected W0 window."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from decimal import Decimal
from pathlib import Path

ROOT = Path('/data/wio/Inference_Foundry')
BASE = ROOT / 'evidence/20260928_loop081_bound/run611'
MANIFEST = BASE / 'offline_parse/manifest.json'
REPLAY = BASE / 'native_graph_replay.json'
MANIFEST_SHA = 'b51eac01dcdc9f183544d51d20040f371f5ffb23a686ded7372b2cb366c84e83'
OUT = ROOT / 'evidence/20260928_loop081_bound/run640/rank_local_current_timeline.json'
PHYSICAL = {'KERNEL_AICORE','KERNEL_MIX_AIC','KERNEL_AIVEC',
            'KERNEL_MIX_AIV','KERNEL_AICPU','MEMCPY_ASYNC'}
WAIT = {'EVENT_WAIT','NOTIFY_WAIT'}


def need(ok, why):
    if not ok:
        raise AssertionError(why)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def d(value):
    return Decimal(str(value).strip())


def union(intervals):
    ordered = sorted(intervals)
    if not ordered:
        return Decimal(0)
    first, last = ordered[0]
    total = Decimal(0)
    for begin, end in ordered[1:]:
        if begin > last:
            total += last-first
            first,last = begin,end
        elif end > last:
            last = end
    return total+last-first


def main():
    need(digest(MANIFEST) == MANIFEST_SHA, 'Run611 parse manifest drift')
    m = json.loads(MANIFEST.read_text())
    r = json.loads(REPLAY.read_text())
    need(m['status'] == 'all8_salvaged_raw_copy_offline_parse_pass' and
         r['status'] == 'all8_three_native_model45_replay_occurrences_pass' and
         r['manifest_sha256'] == MANIFEST_SHA and len(r['rows']) == 24,
         'Run611 identity')
    rows=[]
    for rank in range(8):
        source = m['parsed'][rank]
        need(source['rank']==rank,'rank source order')
        path = ROOT/source['task_time_csv']
        need(digest(path)==source['task_time_sha256'],'task_time raw drift')
        graph=sorted((x for x in r['rows'] if x['rank']==rank),
                     key=lambda x:x['ordinal_in_profile'])
        need([x['ordinal_in_profile'] for x in graph]==[0,1,2],
             'three graph occurrences')
        begin=d(graph[0]['first_task_start_us'])
        end=d(graph[2]['last_task_end_us'])
        need(begin<end,'current window')
        intervals={}
        counts=Counter()
        clipped=Counter()
        all_count=0
        with path.open(newline='') as file:
            for task in csv.DictReader(file):
                a,b=d(task['task_start(us)']),d(task['task_stop(us)'])
                need(a<=b,'negative native task')
                kind=task['kernel_type']
                all_count+=1
                if b<=begin or a>=end:
                    continue
                counts[kind]+=1
                aa,bb=max(a,begin),min(b,end)
                clipped[kind]+=int(a<begin or b>end)
                intervals.setdefault(kind,[]).append((aa,bb))
        physical=[v for kind,items in intervals.items() if kind in PHYSICAL for v in items]
        waits=[v for kind,items in intervals.items() if kind in WAIT for v in items]
        span=end-begin
        busy=union(physical)
        need(Decimal(0)<busy<=span and sum(counts.values())>3*5412,
             'partial window native coverage')
        rows.append({
            'rank':rank, 'window_first_graph_start_us':str(begin),
            'window_third_graph_last_end_us':str(end),
            'window_span_us':str(span),
            'physical_kernel_or_copy_union_us':str(busy),
            'window_without_physical_kernel_or_copy_us':str(span-busy),
            'event_wait_union_us_overlapping_window':str(union(waits)),
            'task_counts_in_window':dict(counts),
            'clipped_at_window_boundary':dict(clipped),
            'total_task_time_rows':all_count,
            'source_task_time_path':source['task_time_csv'],
            'source_task_time_sha256':source['task_time_sha256'],
            'scope':'Rank-local Level0 instrumented Current; no cross-rank clock join or removable-time claim.'})
    result={'status':'all8_rank_local_current_interval_replay_only',
            'manifest_sha256':MANIFEST_SHA,
            'native_graph_replay_sha256':digest(REPLAY),
            'scope':'Three Target FULL Graph occurrences and intervening work in one Run610/611 diagnostic W0. Task_time physical interval union describes Current activity, not necessary resource service, full cycle, legal overlap or observer-free cost.',
            'ranks':rows,
            'framework_only_whole_product_interval':None,
            'formal_run99_cost_transfer':False}
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':result['status'],
                      'rank_count':len(rows),
                      'physical_union_us':[x['physical_kernel_or_copy_union_us']
                                           for x in rows]}))


if __name__ == '__main__':
    main()
