"""Read one complete exported P or D profile, preserving its per-rank coverage."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sqlite3

from parse_SG_step_graph_g1213 import analyze, sqlite_integer


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as file:
        for block in iter(lambda: file.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


class IntervalUnion:
    def __init__(self):
        self.end = None
        self.duration_ns = 0

    def add_ordered(self, start, end):
        previous_end = self.end if self.end is not None else start
        self.duration_ns += max(0, end - max(start, previous_end))
        self.end = max(end, previous_end)


def device_work(path):
    db = sqlite3.connect(path.as_uri() + '?mode=ro&immutable=1', uri=True)
    db.execute('PRAGMA query_only=ON')
    db.execute('PRAGMA temp_store=MEMORY')
    unions = {name: IntervalUnion() for name in ('compute', 'communication', 'compute_or_communication', 'all_task_types')}
    count = Counter()
    durations = Counter()
    operations = Counter()
    op_durations = Counter()
    model_compute = Counter()
    start_ns, end_ns = None, None
    try:
        tables = {name for (name,) in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert {'TASK', 'COMPUTE_TASK_INFO', 'STRING_IDS'} <= tables
        conflicts = db.execute("""
            SELECT COUNT(*) FROM (
                SELECT globalTaskId FROM COMPUTE_TASK_INFO GROUP BY globalTaskId
                HAVING COUNT(DISTINCT opType)>1
            )
        """).fetchone()[0]
        assert conflicts == 0, ('Multiple compute op types for a task', conflicts)
        comm = 'SELECT DISTINCT globalTaskId FROM COMMUNICATION_TASK_INFO'
        if 'COMMUNICATION_TASK_INFO' not in tables:
            comm = 'SELECT NULL AS globalTaskId WHERE 0'
        query = f"""
            WITH compute AS (
                SELECT globalTaskId, MIN(opType) AS opType
                FROM COMPUTE_TASK_INFO GROUP BY globalTaskId
            ), communication AS ({comm})
            SELECT task.startNs, task.endNs, task.modelId, compute.globalTaskId,
                   communication.globalTaskId, names.value, compute.opType
            FROM TASK task LEFT JOIN compute ON task.globalTaskId=compute.globalTaskId
            LEFT JOIN communication ON task.globalTaskId=communication.globalTaskId
            LEFT JOIN STRING_IDS names ON compute.opType=names.id
            ORDER BY CAST(task.startNs AS INTEGER), CAST(task.endNs AS INTEGER)
        """
        for start, end, model, compute_id, comm_id, name, op_id in db.execute(query):
            start, end = map(sqlite_integer, (start, end))
            assert start <= end
            start_ns = start if start_ns is None else min(start_ns, start)
            end_ns = end if end_ns is None else max(end_ns, end)
            value = end - start
            unions['all_task_types'].add_ordered(start, end)
            if compute_id is not None:
                unions['compute'].add_ordered(start, end)
                count['compute'] += 1
                durations['compute'] += value
                label = name if name is not None else f'unresolved_op_type:{op_id}'
                operations[label] += 1
                op_durations[label] += value
                model_compute[str(model)] += 1
            if comm_id is not None:
                unions['communication'].add_ordered(start, end)
                count['communication'] += 1
                durations['communication'] += value
            if compute_id is not None or comm_id is not None:
                unions['compute_or_communication'].add_ordered(start, end)
            else:
                count['other_tasks'] += 1
                durations['other_tasks'] += value
            count['all_task_rows'] += 1
        assert count['compute'] > 0 and start_ns is not None and end_ns is not None
        assert end_ns > start_ns
        return {'event_window_ns':end_ns-start_ns, 'task_rows':dict(count),
                'task_duration_sum_ns':dict(durations),
                'device_interval_union_ns':{k:v.duration_ns for k,v in unions.items()},
                'communication_info_table_present':'COMMUNICATION_TASK_INFO' in tables,
                'model_compute_task_rows':dict(model_compute),
                'top_compute_op_types':[{'op_type':op,'count':operations[op],'duration_sum_ns':duration}
                                        for op,duration in op_durations.most_common(40)],
                'scope':'Within one device capture clock. Independent unions overlap, and sums can count simultaneous streams.'}
    finally:
        db.close()


def analyze_rank(database, info_file, rank):
    before = database.stat()
    digest = sha(database)
    graph = analyze(database)
    assert graph['actual_device_ids'] == [rank], (rank, graph['actual_device_ids'])
    device = device_work(database)
    after = database.stat()
    assert (before.st_mtime_ns, before.st_size, before.st_ino) == (after.st_mtime_ns, after.st_size, after.st_ino)
    return {'rank':rank,'profiler_info_sha256':sha(info_file),'database_sha256':digest,
            'database_bytes':after.st_size,'profile_directory':info_file.parent.name,
            'CPU_and_replay':{k:v for k,v in graph.items() if k not in ('spans','graph_calls')},
            'device':device}


def main(args):
    root = args.root.resolve(strict=True)
    reports = {}
    for info in sorted(root.glob('*_ascend_pt/profiler_info*.json')):
        rank = sqlite_integer(json.loads(info.read_bytes())['rank_id'])
        assert rank in range(8) and rank not in reports, (rank, info.name)
        database = info.parent/'ASCEND_PROFILER_OUTPUT'/f'ascend_pytorch_profiler_{rank}.db'
        reports[rank] = (database, info)
    assert set(reports) == set(range(8)), ('Incomplete export rank inventory', sorted(reports))
    results = [analyze_rank(*reports[rank],rank) for rank in range(8)]
    report = {'role':args.role, 'all8_rank_database_coverage_verified':True, 'ranks':results,
              'scope':'Single role. Compare per-rank logical work and device intervals within each clock. Raw DBs remain in controlled storage.'}
    with args.output.open('x') as file:
        json.dump(report,file,indent=2)
        file.write('\n')


if __name__=='__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--root',type=Path,required=True)
    cli.add_argument('--role',choices=('prefill','decode'),required=True)
    cli.add_argument('--output',type=Path,required=True)
    main(cli.parse_args())
