"""Audit same-device replay partitions with exact launch anchors and repeated graph identities."""

import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
from contextlib import closing
from hashlib import sha256
import json
from pathlib import Path
import sqlite3

from attribute_SG_prefill_steps_g1355 import build_index, enclosing, file_hash, scope_rows
from attribute_named_HCCL_steps_g1492 import intervals_summary, LOW_BITS
from parse_SG_step_graph_g1213 import sqlite_integer


def exact_launches(db):
    forwards, _, _, _ = scope_rows(db)
    index = build_index(forwards)
    launches = []
    rows = db.execute("""
        SELECT t.deviceId,t.startNs,t.endNs,a.startNs,a.endNs,a.globalTid,t.connectionId
        FROM TASK t JOIN STRING_IDS kind ON kind.id=t.taskType
        JOIN CANN_API a ON a.connectionId=t.connectionId
        JOIN STRING_IDS name ON name.id=a.name
        WHERE kind.value='MODEL_EXECUTE' AND name.value='aclmdlRIExecuteAsync'
        AND (SELECT COUNT(*) FROM CANN_API x WHERE x.connectionId=a.connectionId)=1
        ORDER BY CAST(t.startNs AS INTEGER)
    """)
    for rank, begin, end, host_begin, host_end, thread, cid in rows:
        begin, end, hb, he, thread = map(sqlite_integer, (begin, end, host_begin, host_end, thread))
        assert 0 <= begin < end <= LOW_BITS and hb < he
        step, = enclosing(index, thread, hb, he)
        mode = step['span']['mode']
        assert mode in ('DRAFT', 'VERIFY')
        launches.append({'rank': sqlite_integer(rank), 'connection_id': sqlite_integer(cid),
                         'device_begin': begin, 'device_end': end, 'step': step['step'], 'mode': mode})
    raw_anchors = db.execute("""
        SELECT COUNT(*) FROM TASK t JOIN STRING_IDS s ON s.id=t.taskType WHERE s.value='MODEL_EXECUTE'
    """).fetchone()[0]
    raw_calls = db.execute("""
        SELECT COUNT(*) FROM CANN_API a JOIN STRING_IDS s ON s.id=a.name WHERE s.value='aclmdlRIExecuteAsync'
    """).fetchone()[0]
    assert raw_anchors==raw_calls
    assert len(launches)==raw_anchors and len({r['connection_id'] for r in launches})==raw_anchors
    assert len({r['step'] for r in launches})==raw_anchors
    return launches


def model_tasks(db):
    return db.execute("""
        WITH compute AS (
            SELECT x.globalTaskId, MAX(s.value IN ('AI_CORE','AI_VECTOR_CORE','MIX_AIC','MIX_AIV')) AS core
            FROM COMPUTE_TASK_INFO x JOIN STRING_IDS s ON s.id=x.taskType GROUP BY x.globalTaskId
        ), comm AS (
            SELECT x.globalTaskId,
                   MAX(s.value IN ('Memcpy','Reduce_Inline') AND typeof(x.size)='integer' AND x.size>0) AS move,
                   MAX(s.value='Notify_Wait' AND typeof(x.size)='integer' AND x.size=0) AS wait
            FROM COMMUNICATION_TASK_INFO x JOIN STRING_IDS s ON s.id=x.taskType GROUP BY x.globalTaskId
        )
        SELECT t.deviceId,t.startNs,t.endNs,t.modelId,t.streamId,t.taskId,t.contextId,k.value,
               c.core,m.move,m.wait
        FROM TASK t JOIN STRING_IDS k ON k.id=t.taskType
        LEFT JOIN compute c ON c.globalTaskId=t.globalTaskId LEFT JOIN comm m ON m.globalTaskId=t.globalTaskId
        WHERE CAST(t.modelId AS INTEGER) BETWEEN 0 AND 4294967294
    """)


def partition(launches, tasks):
    assert len(launches) >= 2
    rank, = {row['rank'] for row in launches}
    starts = [row['device_begin'] for row in launches]
    assert all(left['device_end'] < right['device_begin'] for left,right in zip(launches,launches[1:]))
    identities = [Counter() for _ in starts]
    seeds = [set() for _ in starts]
    summaries = [defaultdict(list) for _ in starts]
    last_stops = [0 for _ in starts]
    for device, start, stop, model, stream, task, context, name, compute, movement, wait in tasks:
        device, begin, end, model = map(sqlite_integer, (device, start, stop, model))
        assert device==rank and 0 <= begin <= end <= LOW_BITS
        slot = bisect_right(starts,begin)-1
        assert slot >= 0, 'Graph task starts before the first observed anchor'
        assert slot+1==len(starts) or end<=starts[slot+1], 'Graph task crosses the next replay anchor'
        if begin < launches[slot]['device_end']:
            seeds[slot].add(model)
        identities[slot][(model,int(stream),int(task),int(context),name)] += 1
        last_stops[slot] = max(last_stops[slot],end)
        assert not (compute and movement), 'Same device row is typed as both compute and HCCL'
        value = (begin << 64) | end
        for label, flag in (('compute',compute),('communication',movement),('notify_wait',wait)):
            if flag:
                summaries[slot][label].append(value)

    by_model, model_modes = {}, {}
    model_occurrences = Counter()
    results = []
    for launch, seed, ids, summary, last in zip(launches,seeds,identities,summaries,last_stops):
        model, = seed
        assert {key[0] for key in ids} == {model}, 'More than one model class in the device partition'
        assert by_model.setdefault(model,ids)==ids, 'Same model ID has a different repeated task identity set'
        assert model_modes.setdefault(model,launch['mode'])==launch['mode']
        model_occurrences[model] += 1
        counts = {group:len(intervals) for group,intervals in summary.items()}
        fingerprint = sha256(json.dumps(sorted(ids.items()),separators=(',',':')).encode()).hexdigest()
        results.append({'step':launch['step'],'mode':launch['mode'],'device_model_id':model,
                        'graph_task_identity_list_sha256':fingerprint,'graph_task_rows':sum(ids.values()),
                        'named_event_rows':counts,'intervals':intervals_summary(summary),
                        'last_observed_model_task_end_after_anchor_ns':last-launch['device_begin']})
    assert all(count>=2 for count in model_occurrences.values()), 'A graph has no observed repeat'
    return {'device_rank':rank,'model_replay_counts':dict(model_occurrences),'scoped_replays':results,
            'method': 'CPU forward to unique CANN launch to MODEL_EXECUTE uses device connection IDs. '
                      'Membership of remaining graph tasks is inferred from device-only noncrossing partitions, '
                      'the model ID visible at launch, and exact repeated graph task identities. '
                      'The last device row is the observed boundary, with no end-to-end interpretation.'}


def summarize(path, expected):
    before=path.stat()
    assert file_hash(path)==expected
    assert not any(path.with_name(path.name+ext).exists() for ext in ('-wal','-journal','-shm'))
    with closing(sqlite3.connect(path.as_uri()+'?mode=ro&immutable=1',uri=True)) as db:
        db.execute('PRAGMA query_only=ON')
        db.execute('PRAGMA temp_store=MEMORY')
        result=partition(exact_launches(db),model_tasks(db))
        graph_rows = db.execute("""
            SELECT COUNT(*) FROM TASK WHERE CAST(modelId AS INTEGER) BETWEEN 0 AND 4294967294
        """).fetchone()[0]
        assert graph_rows==sum(row['graph_task_rows'] for row in result['scoped_replays'])
    after=path.stat()
    assert (before.st_ino,before.st_mtime_ns,before.st_size)==(after.st_ino,after.st_mtime_ns,after.st_size)
    result['database_sha256']=expected
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('database',type=Path)
    parser.add_argument('sha256')
    args=parser.parse_args()
    print(json.dumps(summarize(args.database.resolve(strict=True),args.sha256),indent=2,sort_keys=True))
