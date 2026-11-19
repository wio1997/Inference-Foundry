"""Audit graph launch anchors and device-side temporal candidates in one rank."""

import argparse
import json
from pathlib import Path
import sqlite3

from attribute_SG_prefill_steps_g1355 import build_index, enclosing, file_hash, scope_rows
from parse_SG_step_graph_g1213 import optional_integer, sqlite_integer


def audit_windows(db):
    forwards, _, _, _ = scope_rows(db)
    index = build_index(forwards)
    calls = db.execute("""
        SELECT api.connectionId, api.startNs, api.endNs, api.globalTid,
               (SELECT COUNT(*) FROM CANN_API all_api
                WHERE all_api.connectionId=api.connectionId)
        FROM CANN_API api JOIN STRING_IDS name ON name.id=api.name
        WHERE name.value='aclmdlRIExecuteAsync'
        ORDER BY CAST(api.startNs AS INTEGER)
    """).fetchall()
    results = []
    comm_table = db.execute("""
        SELECT 1 FROM sqlite_master WHERE type='table' AND name='COMMUNICATION_TASK_INFO'
    """).fetchone() is not None
    for cid, start, stop, tid, connection_count in calls:
        connection, begin, end, thread = map(optional_integer, (cid, start, stop, tid))
        spans = (enclosing(index, thread, begin, end)
                 if None not in (thread, begin, end) and begin <= end else [])
        row = {'connection_id': connection, 'CANN_rows_with_connection': connection_count,
               'CPU_candidates': [{'step': span['step'], 'mode': span['span']['mode']} for span in spans]}
        results.append(row)
        anchors = db.execute("""
            SELECT t.deviceId, t.startNs, t.endNs FROM TASK t
            JOIN STRING_IDS kind ON kind.id=t.taskType
            WHERE kind.value='MODEL_EXECUTE' AND t.connectionId=?
        """, (cid,)).fetchall() if connection_count == 1 and connection is not None else []
        row['direct_MODEL_EXECUTE_count'] = len(anchors)
        if len(anchors) != 1:
            continue
        device, launch_start, launch_end = map(sqlite_integer, anchors[0])
        assert 0 <= launch_start <= launch_end
        row['device_id'] = device
        row['MODEL_EXECUTE_duration_ns'] = launch_end - launch_start
        communication = ('EXISTS (SELECT 1 FROM COMMUNICATION_TASK_INFO m '
                         'WHERE m.globalTaskId=t.globalTaskId)' if comm_table else '0')
        candidates = db.execute(f"""
            SELECT CAST(t.modelId AS INTEGER), CAST(t.startNs AS INTEGER),
                   CAST(t.endNs AS INTEGER),
                   EXISTS (SELECT 1 FROM COMPUTE_TASK_INFO c WHERE c.globalTaskId=t.globalTaskId),
                   {communication}
            FROM TASK t WHERE t.deviceId=? AND CAST(t.modelId AS INTEGER) BETWEEN 0 AND 4294967294
              AND CAST(t.startNs AS INTEGER) BETWEEN ? AND ?
        """, (anchors[0][0], launch_start, launch_end)).fetchall()
        assert all(launch_start <= begin <= end for _, begin, end, _, _ in candidates)
        row['temporal_candidate_model_ids'] = sorted({item[0] for item in candidates})
        row['model_task_starts_in_launch'] = len(candidates)
        row['model_task_ends_after_launch'] = sum(item[2] > launch_end for item in candidates)
        row['compute_or_comm_starts_in_launch'] = sum(bool(item[3] or item[4]) for item in candidates)
        row['compute_or_comm_fully_inside_launch'] = sum(
            item[2] <= launch_end and bool(item[3] or item[4]) for item in candidates)
    return {'replays': results, 'communication_metadata_present': comm_table,
            'scope': ('Unique CANN connection establishes the MODEL_EXECUTE anchor. '
                      'Timestamps of other device tasks identify temporal model candidates only. '
                      'The launch interval is not a graph-completion bound. No CPU/device clock comparison.')}


def analyze(path, expected_sha256):
    path = path.resolve(strict=True)
    wal = path.with_name(path.name + '-wal')
    assert not wal.exists() or wal.stat().st_size == 0, 'Use the frozen complete database'
    before = path.stat()
    assert file_hash(path) == expected_sha256, 'Database checksum mismatch'
    db = sqlite3.connect(path.as_uri() + '?mode=ro&immutable=1', uri=True)
    try:
        db.execute('PRAGMA query_only=ON')
        db.execute('PRAGMA temp_store=MEMORY')
        report = audit_windows(db)
    finally:
        db.close()
    after = path.stat()
    assert (before.st_ino, before.st_size, before.st_mtime_ns) == (after.st_ino, after.st_size, after.st_mtime_ns)
    return {'database_sha256': expected_sha256, **report}


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--database', type=Path, required=True)
    cli.add_argument('--sha256', required=True)
    cli.add_argument('--output', type=Path, required=True)
    args = cli.parse_args()
    with args.output.open('x') as stream:
        json.dump(analyze(args.database, args.sha256), stream, indent=2)
        stream.write('\n')
