"""Contained CPU scopes of existing Run249 list gathers, not a gain estimator."""
import bisect
import collections
import gzip
import hashlib
import json
from pathlib import Path


def union(xs):
    end, total = 0, 0
    for start, stop in sorted(xs):
        if stop > max(start, end):
            total += stop-max(start, end)
            end = stop
    return total / 1e6


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[2]
    source = root/'jobs/GLM53-DECODE-TRACE-20261006-B/reduced/db_rank15.json.gz'
    data = json.loads(gzip.decompress(source.read_bytes()))
    gathers = sorted([(int(c[0]), int(c[1])) for c in data['cpu'] if c[2] == 'c10d::allgather_'])
    starts = [s for s, e in gathers]
    groups = collections.defaultdict(list)
    for start, stop, name, connection, kind in data['cpu']:
        start, stop = int(start), int(stop)
        i = bisect.bisect_right(starts, start)-1
        if i >= 0 and start >= gathers[i][0] and stop <= gathers[i][1]:
            groups[name].append((start, stop))
    rows = []
    for name, intervals in sorted(groups.items(), key=lambda kv: -sum(e-s for s,e in kv[1])):
        rows.append(dict(name=name, count=len(intervals), inclusive_ms=sum(e-s for s,e in intervals)/1e6,
                         union_ms=union(intervals)))
    all_splits = sorted([(int(c[0]),int(c[1])) for c in data['cpu'] if c[2]=='aten::tensor_split'])
    split_starts = [s for s,e in all_splits]
    # Prepare has three other tensor_split calls per MoE. Finalize's output
    # split is the last completed split immediately preceding its list gather.
    splits = []
    for start, stop in gathers:
        i = bisect.bisect_right(split_starts, start)-1
        assert i >= 0 and all_splits[i][1] <= start
        splits.append(all_splits[i])
    assert len(set(splits)) == 380
    assert len(gathers) == 380 and len(groups['aten::copy_']) == 6080 and len(splits) == 380
    result = dict(rank=15, db_sha256=data['db_sha256'], reduced_input_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  list_gather_count=380, contained_scopes=rows,
                  all_tensor_split_count=len(all_splits), final_output_tensor_split_count=380, tensor_split_union_ms=union(splits),
                  removed_copy_and_split_union_ms=union(groups['aten::copy_']+splits),
                  limits='Source proves these calls are removed, while a direct collective and fresh output remain. '
                  'Profile-ON host scope wall time includes profiler/runtime/scheduling overhead. '
                  'Nested scope rows overlap; no summation or profOFF saving prediction. '
                  'Copy/split scope union is not necessarily exposed on the cross-rank critical path.')
    (Path(__file__).parent/'gather_materialization.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='contained_scopes'},indent=2))
