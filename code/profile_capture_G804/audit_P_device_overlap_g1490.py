"""Measure per-rank device-time coverage in the original G804 Prefill exports."""

import argparse
from collections import Counter, defaultdict
from contextlib import closing
from hashlib import sha256
import json
from pathlib import Path
import sqlite3
import time


COMPUTE = {'AI_CORE', 'AI_VECTOR_CORE', 'MIX_AIC', 'MIX_AIV'}
DATA_MOVEMENT = {'Memcpy', 'Reduce_Inline'}
NOTIFICATIONS = {'Notify_Record', 'Notify_Wait'}


def union_duration(intervals):
    total = 0
    last = None
    for start, stop in sorted(intervals):
        if last is None or start > last:
            total += stop - start
            last = stop
        elif stop > last:
            total += stop - last
            last = stop
    return total


def classify_tasks(db):
    by_id = {}
    for task, kind in db.execute("""
        SELECT c.globalTaskId, s.value
        FROM COMPUTE_TASK_INFO c JOIN STRING_IDS s ON s.id = c.taskType
    """):
        assert task not in by_id
        assert kind in COMPUTE or kind == 'AI_CPU'
        by_id[task] = 'compute' if kind in COMPUTE else 'ai_cpu'
    for task, kind, width in db.execute("""
        SELECT c.globalTaskId, s.value, c.size
        FROM COMMUNICATION_TASK_INFO c JOIN STRING_IDS s ON s.id = c.taskType
    """):
        assert task not in by_id
        if kind in DATA_MOVEMENT:
            assert width > 0
            by_id[task] = 'hccl_data'
        else:
            assert kind in NOTIFICATIONS and width == 0
            by_id[task] = kind
    return by_id


def read_rank(db, rank):
    by_id = classify_tasks(db)
    intervals = defaultdict(list)
    kinds = Counter()
    for task, device, start, end, kind in db.execute(
        'SELECT globalTaskId,deviceId,startNs,endNs,taskType FROM TASK'
    ):
        assert device == rank
        begin, stop = int(start), int(end)
        assert 0 <= begin <= stop
        category = by_id.pop(task, 'other')
        if category == 'other':
            kinds[kind] += 1
        intervals[category].append((begin, stop))
    assert not by_id
    coverages = {key: union_duration(value) for key, value in intervals.items()}
    compute = coverages['compute']
    data = coverages['hccl_data']
    union = union_duration(intervals['compute'] + intervals['hccl_data'])
    all_tasks = union_duration([interval for group in intervals.values() for interval in group])
    overlapped = compute + data - union
    assert 0 <= overlapped <= min(compute, data) and union <= all_tasks
    return {
        'rank': rank,
        'device_task_counts': {key: len(value) for key, value in intervals.items()},
        'other_device_task_types': dict(kinds),
        'device_time_union_ns': coverages,
        'total_device_task_union_ns': all_tasks,
        'hccl_data_and_compute_overlap_ns': overlapped,
        'hccl_data_outside_compute_ns': data - overlapped,
        'hccl_data_overlap_fraction': overlapped / data,
    }


def summarize(root):
    results = []
    for rank in range(8):
        directory = root / 'offline_rank0_g809' if rank == 0 else root / 'offline_remaining_g816' / f'rank{rank}'
        receipt = (directory / 'result.json').read_bytes()
        record = json.loads(receipt)
        assert record['returncode'] == 0 and record['original_raw_SHA256_unchanged']
        export = Path(record['output']).resolve(strict=True)
        assert export.is_relative_to(directory.resolve()) and export.name == 'ASCEND_PROFILER_OUTPUT'
        db_path = export / f'ascend_pytorch_profiler_{rank}.db'
        assert not any(db_path.with_name(db_path.name + ext).exists() for ext in ('-wal', '-journal', '-shm'))
        before = db_path.stat()
        expiry = time.monotonic() + 14
        with closing(sqlite3.connect(db_path.as_uri() + '?mode=ro&immutable=1', uri=True)) as db:
            db.execute('PRAGMA query_only=ON')
            db.execute('PRAGMA temp_store=MEMORY')
            db.execute('PRAGMA cache_size=-16384')
            db.set_progress_handler(lambda: int(time.monotonic() > expiry), 10_000)
            result = read_rank(db, rank)
        after = db_path.stat()
        assert (before.st_ino, before.st_size, before.st_mtime_ns) == (
            after.st_ino, after.st_size, after.st_mtime_ns
        )
        result['worker_export_receipt_sha256'] = sha256(receipt).hexdigest()
        result['exported_database_bytes'] = before.st_size
        results.append(result)
    return {'capture': 'G804', 'epoch': 727, 'role': 'prefill', 'physical_ranks': results,
            'scope': 'Trace-wide interval coverage on each device separately. Compute includes vector and matrix '
                     'cores. HCCL device data intervals include positive-size Memcpy and Reduce_Inline. '
                     'Notification intervals are separate. Overlap is observed concurrency, without '
                     'critical-path attribution or inter-node KV-transfer coverage.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('profile_root', type=Path)
    print(json.dumps(summarize(parser.parse_args().profile_root), indent=2, sort_keys=True))
