"""Offline same-host queue-edge decomposition, never imports torch."""
import bisect
import collections
import gzip
import json
from pathlib import Path


def read(path):
    return json.loads(gzip.decompress(path.read_bytes()))


def split(rank, taskfile, queuefile):
    r, q = read(taskfile), read(queuefile)
    flow = collections.defaultdict(set)
    for group, connection in q['connection_ids']:
        flow[group].add(connection)
    enqueue = {}
    for row in q['queue']:
        if row[2].startswith('Enqueue@'):
            for connection in flow[row[3]]:
                enqueue[connection] = row
    deqs = sorted([row for row in q['queue'] if row[2].startswith('Dequeue@')], key=lambda row: int(row[0]))
    ds = [int(row[0]) for row in deqs]
    tasks = sorted(r['tasks'], key=lambda x: x[0])
    embeddings = [x for x in tasks if 'Embedding' in (x[6] or '')]
    copies = [x for x in tasks if x[2] == 45 and x[9] == 'aclrtMemcpyAsyncWithCondition']
    assert len(embeddings) == 10 and len(copies) == 5
    lo, hi = embeddings[0][0], copies[-1][1]
    main = [x for x in tasks if x[2] == 47]
    rows = []
    totals = collections.Counter()
    by_name = collections.Counter()
    for prev, task in zip(main, main[1:]):
        if task[7] is None:
            continue
        start, end = max(lo, prev[1]), min(hi, task[0], task[7])
        if end <= start:
            continue
        total = end-start; totals['host_launch_late_ns'] += total
        i = bisect.bisect_right(ds, task[7])-1
        dequeue = deqs[i] if i >= 0 else None
        matched = bool(dequeue and int(dequeue[0]) <= task[7] <= int(dequeue[1]) and dequeue[4] == task[10])
        if not matched:
            totals['unmatched_queue_ns'] += total
            continue
        es = [enqueue[f] for f in flow[dequeue[3]] if f in enqueue]
        assert len(es) == 1
        enq = es[0]
        assert enq[2].split('@', 1)[1] == dequeue[2].split('@', 1)[1]
        e0, e1, d0 = int(enq[0]), int(enq[1]), int(dequeue[0])
        # Partition only this exposed interval. Values outside it contribute0.
        # A consumer can begin before Enqueue returns. The push point is not
        # exported, so use its start as the earliest producer-ready bound.
        # Do not add overlapping producer/consumer call durations.
        if d0 < e0: totals['enqueue_dequeue_inversion_count'] += 1
        d0 = max(d0, e0)
        boundaries = [('frontend_not_enqueued_ns', -float('inf'), e0),
                      ('enqueue_start_to_dequeue_ns', e0, d0),
                      ('dequeue_before_launch_ns', d0, task[7])]
        parts = {name: max(0, min(end, b)-max(start, a)) for name, a, b in boundaries}
        assert abs(sum(parts.values())-total) < 1
        totals.update(parts)
        by_name[enq[2]] += parts['frontend_not_enqueued_ns']
        rows.append(dict(start_ns=start, end_ns=end, gap_us=total/1000,
                         previous_task=prev, next_task=task, enqueue=enq, dequeue=dequeue,
                         parts_ns=parts))
    return dict(rank=rank, window=[lo, hi], window_ms=(hi-lo)/1e6,
                task_input_sha=r['db_sha256'], queue_rows=len(q['queue']),
                totals_ms={k: v/1e6 for k, v in totals.items() if k.endswith('_ns')},
                inversions=totals['enqueue_dequeue_inversion_count'],
                frontend_late_by_next_name_ms={k: v/1e6 for k, v in by_name.most_common(20)},
                largest_edges=sorted(rows, key=lambda x: x['gap_us'], reverse=True)[:12],
                interpretation='Same-host profON exposed gap partition; no profOFF extrapolation, no removable bound. Queue-pending can include necessary prior backend work.')


if __name__ == '__main__':
    base = Path(__file__).resolve().parents[2]
    taskbase = base/'jobs/GLM53-DECODE-TRACE-20261006-B/reduced'
    queuebase = base/'jobs/GLM53-DECODE-QUEUES-20261006'
    result = [split(rank, taskbase/f'db_rank{rank}.json.gz', queuebase/f'queues_rank{rank}.json.gz') for rank in (13, 15)]
    (Path(__file__).parent/'queue_critical_edges.json').write_text(json.dumps(result, indent=2)+'\n')
    for r in result:
        print(json.dumps({k: v for k, v in r.items() if k != 'largest_edges'}, indent=2))
