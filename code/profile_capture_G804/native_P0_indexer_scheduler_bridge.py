"""Join native Prefill index tasks to CPU scheduler steps in a frozen SDK DB."""

import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
from contextlib import closing
from hashlib import sha256
import json
from pathlib import Path
import sqlite3
import time

INDEXER_TYPE = "VllmAscendQuantLightningIndexerV2"
CPU_INDEXER_API = "aclnnVllmAscendQuantLightningIndexerV2"
SCHEDULER_MARKER = "ds41.scheduler:"


def sequence_sha256(records):
    digest = sha256()
    for record in records:
        normalized = json.dumps(record, separators=(",", ":"), sort_keys=True)
        digest.update(normalized.encode("utf-8") + b"\n")
    return digest.hexdigest()


def read_trace(connection):
    scheduler = connection.execute("""
        SELECT api.startNs, api.endNs, api.globalTid, name.value
        FROM PYTORCH_API api
        JOIN STRING_IDS name ON name.id = api.name
        WHERE name.value LIKE ?
    """, (SCHEDULER_MARKER + "%",)).fetchall()
    tasks = connection.execute("""
        SELECT t.globalTaskId, t.connectionId, t.startNs, t.endNs,
               t.deviceId, t.modelId, shape.value,
               launch.connectionId, launch.globalTid, launch_name.value
        FROM TASK t
        JOIN COMPUTE_TASK_INFO op ON op.globalTaskId = t.globalTaskId
        JOIN STRING_IDS op_type ON op_type.id = op.opType
        JOIN STRING_IDS shape ON shape.id = op.inputShapes
        JOIN CANN_API launch ON launch.connectionId = t.connectionId
        JOIN STRING_IDS launch_name ON launch_name.id = launch.name
        WHERE op_type.value = ?
        ORDER BY t.globalTaskId
    """, (INDEXER_TYPE,)).fetchall()
    launch_ids = sorted({task[1] for task in tasks})
    assert launch_ids and len(launch_ids) == len(tasks)
    bindings = connection.execute(f"""
        SELECT connectionId, id FROM CONNECTION_IDS
        WHERE connectionId IN ({','.join('?' for _ in launch_ids)})
    """, launch_ids).fetchall()
    binding_by_launch = dict(bindings)
    api_ids = sorted(set(binding_by_launch.values()))
    assert len(bindings) == len(binding_by_launch) == len(api_ids) == len(tasks)
    launches = connection.execute(f"""
        SELECT api.connectionId, api.startNs, api.endNs, api.globalTid, name.value
        FROM PYTORCH_API api
        JOIN STRING_IDS name ON name.id = api.name
        WHERE api.connectionId IN ({','.join('?' for _ in api_ids)})
    """, api_ids).fetchall()
    launch_calls = {group: (int(begin), int(end), tid, name)
                    for group, begin, end, tid, name in launches}
    assert len(launch_calls) == len(launches) == len(tasks)
    return scheduler, tasks, binding_by_launch, launch_calls


def summarize(database, expected_logical, deadline_seconds):
    before = database.stat()
    deadline = time.monotonic() + deadline_seconds
    with closing(sqlite3.connect(database.as_uri() + "?mode=ro&immutable=1", uri=True)) as db:
        db.execute("PRAGMA query_only=ON")
        db.execute("PRAGMA temp_store=MEMORY")
        db.execute("PRAGMA cache_size=-16384")
        db.set_progress_handler(lambda: int(time.monotonic() > deadline), 10_000)
        scheduler, tasks, groups, launches = read_trace(db)
    after = database.stat()
    assert (before.st_ino, before.st_size, before.st_mtime_ns) == (
        after.st_ino, after.st_size, after.st_mtime_ns
    )

    by_thread = defaultdict(list)
    logical = {}
    for begin, end, thread, label in scheduler:
        payload = json.loads(label.removeprefix(SCHEDULER_MARKER))
        iteration = payload["iteration"]
        assert payload["rank"] == 0 and iteration not in logical
        by_thread[thread].append((int(begin), int(end), iteration))
        logical[iteration] = payload
    ordered_logical = [logical[iteration] for iteration in sorted(logical)]
    assert ordered_logical == expected_logical
    starts = {}
    for thread, spans in by_thread.items():
        spans.sort()
        assert all(begin <= end for begin, end, _ in spans)
        assert all(a[1] <= b[0] for a, b in zip(spans, spans[1:]))
        starts[thread] = [span[0] for span in spans]

    mapped = defaultdict(list)
    fingerprints = []
    distinct_tasks = set()
    cann_threads = set()
    for task in tasks:
        task_id, launch_id, start, end, device, model, shape, cann_id, cann_thread, cann_name = task
        assert task_id not in distinct_tasks and start < end
        assert (device, model, cann_id, cann_name) == (0, 4294967295, launch_id, "launch")
        distinct_tasks.add(task_id)
        cann_threads.add(cann_thread)
        group = groups[launch_id]
        host_begin, host_end, host_tid, host_name = launches[group]
        assert host_name == CPU_INDEXER_API and host_begin <= host_end
        position = bisect_right(starts.get(host_tid, ()), host_begin) - 1
        assert position >= 0, (task_id, "missing scheduler step on PyTorch API thread")
        begin, finish, iteration = by_thread[host_tid][position]
        assert begin <= host_begin <= host_end <= finish, (task_id, "API outside scheduler step")
        query_rows = int(shape.strip(' "[]').split(",", 1)[0])
        mapped[iteration].append((query_rows, end - start))
        fingerprints.append((task, group, launches[group], iteration))

    rows = []
    for iteration, payload in sorted(logical.items()):
        observed = mapped[iteration]
        query_tokens = payload["scheduled_query_tokens"]
        counts = Counter(query_rows for query_rows, _ in observed)
        assert bool(observed) == bool(query_tokens)
        rows.append({
            "iteration": iteration,
            "scheduled_query_tokens": query_tokens,
            "scheduled_requests": len(payload["requests"]),
            "device_tasks_by_query_rows": dict(sorted(counts.items())),
            "device_tasks": len(observed),
            "full_query_device_tasks": counts[query_tokens],
            "summed_device_task_duration_ns": sum(duration for _, duration in observed),
        })

    return {
        "capture": "G804",
        "source_epoch": 727,
        "role": "prefill",
        "rank": 0,
        "device_operator_name": INDEXER_TYPE,
        "pytorch_host_api_name": CPU_INDEXER_API,
        "source_database_bytes": before.st_size,
        "selected_task_and_bridge_rows_sha256": sequence_sha256(fingerprints),
        "scheduler_payload_sequence_sha256": sequence_sha256(ordered_logical),
        "scheduler_thread_count": len(by_thread),
        "cann_launch_thread_count": len(cann_threads),
        "shared_cann_launch_scheduler_threads": len(cann_threads.intersection(by_thread)),
        "unique_exact_mappings": len(distinct_tasks),
        "steps": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database", type=Path)
    parser.add_argument("canonical_annotations", type=Path)
    parser.add_argument("--deadline-seconds", default=30, type=int)
    args = parser.parse_args()
    canonical = json.loads(args.canonical_annotations.read_bytes())
    expected_logical = [row["logical"] for row in canonical]
    result = summarize(args.database.resolve(strict=True), expected_logical, args.deadline_seconds)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
