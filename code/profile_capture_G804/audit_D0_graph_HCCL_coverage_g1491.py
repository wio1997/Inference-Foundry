"""Measure named Compute and HCCL device intervals in native G804 Decode rank 0."""

import argparse
from collections import Counter, defaultdict
from contextlib import closing
from hashlib import sha256
from heapq import merge
import json
from pathlib import Path
import sqlite3
import time

from audit_P_device_overlap_g1490 import COMPUTE, DATA_MOVEMENT, NOTIFICATIONS


LOW_BITS = (1 << 64) - 1
GROUPS = ('target', 'draft', 'other')
GRAPH_INPUT_SHA = '365dc6042d9baa14c4eabd63eb4be242c2362143727e4433fd12cade6712c8f5'


def graph_models(paired_data):
    assert sha256(paired_data).hexdigest() == GRAPH_INPUT_SHA
    paired = json.loads(paired_data)
    assert paired['capture'] == 'G804' and paired['decode_physical_rank'] == 0
    assert paired['target_CANN_replays_directly_correlated_and_enclosed_by_nonzero_scheduler_span'] == 906
    assert paired['following_draft_CANN_replays_directly_correlated_with_device_graph'] == 906
    pairs = paired['by_paired_batches']
    target = {row['target_model_id'] for row in pairs}
    draft = {row['draft_model_id'] for row in pairs}
    assert target and draft and not target & draft
    return {**dict.fromkeys(target, 'target'), **dict.fromkeys(draft, 'draft')}


def task_kinds(db):
    lookup = {}
    metadata = Counter()

    def add(key, category):
        prior = lookup.setdefault(key, category)
        assert prior == category, 'Conflicting task kind assignments'
        metadata[category] += 1

    for task, kind in db.execute("""
        SELECT c.globalTaskId, s.value
        FROM COMPUTE_TASK_INFO c LEFT JOIN STRING_IDS s ON s.id = c.taskType
    """):
        assert kind in COMPUTE or kind == 'AI_CPU'
        add(task, 'compute' if kind in COMPUTE else 'ai_cpu')
    for task, kind, width in db.execute("""
        SELECT c.globalTaskId, s.value, c.size
        FROM COMMUNICATION_TASK_INFO c LEFT JOIN STRING_IDS s ON s.id = c.taskType
    """):
        if kind in DATA_MOVEMENT:
            assert width > 0
            add(task, 'named_hccl_data')
        elif kind in NOTIFICATIONS:
            assert width == 0
            add(task, kind)
        else:
            assert kind == 'N/A' and width == -1
            metadata['untyped_comm_metadata'] += 1
    return lookup, metadata


def coalesce(encoded):
    spans = []
    last = None
    for value in encoded:
        begin, stop = value >> 64, value & LOW_BITS
        if last is None or begin > last:
            spans.append(value)
            last = stop
        elif stop > last:
            spans[-1] = (spans[-1] & ~LOW_BITS) | stop
            last = stop
    return spans


def duration(encoded):
    return sum((value & LOW_BITS) - (value >> 64) for value in encoded)


def interval_summary(compute, data, wait):
    combined = coalesce(merge(compute, data))
    c_ns, d_ns = duration(compute), duration(data)
    overlapped = c_ns + d_ns - duration(combined)
    assert 0 <= overlapped <= min(c_ns, d_ns)
    return {'compute_union_ns': c_ns, 'named_hccl_data_union_ns': d_ns,
            'notify_wait_union_ns': duration(wait),
            'compute_and_named_hccl_data_overlap_ns': overlapped,
            'named_hccl_data_outside_compute_ns': d_ns - overlapped}


def summarize(path, paired_data):
    models = graph_models(paired_data)
    intervals = defaultdict(list)
    counts, models_seen = Counter(), Counter()
    ordinals_seen = set()
    end = time.monotonic() + 85
    with closing(sqlite3.connect(path.as_uri() + '?mode=ro&immutable=1', uri=True)) as db:
        db.execute('PRAGMA query_only=ON')
        db.execute('PRAGMA temp_store=MEMORY')
        db.execute('PRAGMA cache_size=-16384')
        db.set_progress_handler(lambda: int(time.monotonic() > end), 10000)
        lookup, metadata = task_kinds(db)
        for task, rank, model, start, stop in db.execute(
            'SELECT globalTaskId,deviceId,modelId,startNs,endNs FROM TASK'
        ):
            assert rank == 0
            group = models.get(model, 'other')
            kind = lookup.get(task, 'unclassified')
            key = (group, kind)
            counts[key] += 1
            models_seen[model] += 1
            if kind in ('compute', 'named_hccl_data', 'Notify_Wait'):
                begin, finish = int(start), int(stop)
                assert 0 <= begin <= finish <= LOW_BITS
                intervals[key].append((begin << 64) | finish)
            if kind != 'unclassified':
                ordinals_seen.add(task)
    for key, values in intervals.items():
        values.sort()
        intervals[key] = coalesce(values)
    assert set(models) <= set(models_seen)
    kinds = ('compute', 'named_hccl_data', 'Notify_Wait')
    summaries = {group: interval_summary(*(intervals[group, kind] for kind in kinds)) for group in GROUPS}
    summaries['all'] = interval_summary(*(
        coalesce(merge(*(intervals[group, kind] for group in GROUPS))) for kind in kinds
    ))
    return {'capture': 'G804', 'epoch': 727, 'role': 'decode', 'physical_rank': 0,
            'closed_device_model_ids': dict(models_seen), 'graph_model_labels': models,
            'classified_task_counts': {
                group: {kind: count for (label, kind), count in counts.items() if label == group}
                for group in GROUPS
            },
            'named_kernel_metadata_by_category': dict(metadata),
            'typed_global_task_ids_without_device_rows': len(lookup) - len(ordinals_seen),
            'task_global_id_reuse_seen': sum(count for (_, kind), count in counts.items()
                                             if kind != 'unclassified') > len(ordinals_seen),
            'graph_group_interval_summary': summaries,
            'paired_graph_evidence_sha256': GRAPH_INPUT_SHA,
            'scope': 'Closed rank-zero device trace with graph model IDs classified by direct replay correlation. '
                     'Data movement includes only positive-size named HCCL Memcpy and Reduce_Inline tasks. '
                     'Untyped communication metadata is counted separately. These intervals do not isolate '
                     'all graph communication, other ranks, inter-node KV or service latency.'}


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('closed_D0_database', type=Path)
    cli.add_argument('paired_graph_evidence', type=Path)
    args = cli.parse_args()
    print(json.dumps(summarize(args.closed_D0_database.resolve(), args.paired_graph_evidence.read_bytes()),
                     indent=2, sort_keys=True))
