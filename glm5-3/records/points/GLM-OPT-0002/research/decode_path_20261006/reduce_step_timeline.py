"""Read immutable Run249 reductions; no torch imports or device requests."""
import gzip
import json
from pathlib import Path


def union(intervals, lo, hi):
    end, total = lo, 0
    for start, stop in sorted(intervals):
        start, stop = max(start, lo), min(stop, hi)
        if stop > max(start, end):
            total += stop - max(start, end)
            end = stop
    return total / 1e6


def reduce_rank(data):
    tasks = data['tasks']
    embeddings = sorted([t for t in tasks if 'aclnnEmbedding_Gather' in (t[6] or '')])
    copies = sorted([t for t in tasks if t[2] == 45 and t[9] == 'aclrtMemcpyAsyncWithCondition'])
    assert len(embeddings) == 10 and len(copies) == 5
    categories = {
        'non_event_tasks': [t for t in tasks if t[9] not in ('aclrtRecordEvent', 'aclrtStreamWaitEvent')
                            and t[6] not in ('PROFILING_ENABLE', 'PROFILING_DISABLE')],
        'named_math_tasks': [t for t in tasks if t[6] and t[9] == 'launch'
                             and t[6] not in ('AivKernel', 'MoeDistributeDispatchV2', 'MoeDistributeCombineV2')
                             and not any(s in t[6].lower() for s in ('profiling', 'batch_get'))],
        'HCOM_task': [t for t in tasks if t[6] == 'AivKernel'],
        'EP_dispatch_combine_task': [t for t in tasks if t[6] in ('MoeDistributeDispatchV2', 'MoeDistributeCombineV2')],
        'event_wait': [t for t in tasks if t[9] == 'aclrtStreamWaitEvent'],
        'copy_task': [t for t in tasks if t[9] in ('aclnnInplaceCopy', 'aclrtMemcpyAsync', 'aclrtMemcpyAsyncWithCondition')],
    }
    result = []
    for i in range(5):
        lo, hi = embeddings[2*i][0], copies[i][1]
        row = dict(round=i+1, target_embedding_ns=lo, draft_embedding_ns=embeddings[2*i+1][0],
                   sample_D2H_end_ns=hi, extent_ms=(hi-lo)/1e6)
        row.update({name + '_union_ms': union([(t[0], t[1]) for t in ts], lo, hi)
                    for name, ts in categories.items()})
        row['next_device_embedding_gap_ms'] = (embeddings[2*i+2][0]-hi)/1e6 if i < 4 else None
        result.append(row)
    return result


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[2]
    data = {}
    for rank in (0, 13, 15):
        path = root / 'jobs/GLM53-DECODE-TRACE-20261006-B/reduced' / ('db_rank%d.json.gz' % rank)
        data[str(rank)] = reduce_rank(json.loads(gzip.decompress(path.read_bytes())))
    data['limits'] = [
        'All original Run249 profile-ON timestamps on host167. Math, HCOM, EP, copies and events overlap and must not be added. '
        'Non-event inactive time is a bound, not automatically removable. Named-math union excludes unnamed subtasks; '
        'it is not a theoretical necessary compute floor. HCOM Aiv and EP kernels can include peer wait. '
        'D2H-to-next-embedding interval is a device boundary gap, not a timestamped Scheduler/Executor/ModelRunner decomposition.'
    ]
    (Path(__file__).parent / 'step_timeline.json').write_text(json.dumps(data, indent=2) + '\n')
