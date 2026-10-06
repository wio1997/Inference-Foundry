"""Partition the existing matched HCOM operations into actual target/MTP rounds.

Round membership uses rank0 device-embedding boundaries in the shared host167
clock. Start-skew and tail sums overlap; neither is an additive wall budget.
"""
import collections
from decimal import Decimal
import gzip
import hashlib
import json
from pathlib import Path

if __name__ == '__main__':
    r = Path(__file__).resolve().parent
    p = r.parents[1]
    paths = dict(collectives=p/'jobs/GLM53-DECODE-TRACE-20261006/reduced/collectives0.json.gz',
                 skew=p/'jobs/GLM53-DECODE-TRACE-20261006/reduced/cross_rank_collective_start.json',
                 launch=p/'jobs/GLM53-DECODE-TRACE-20261006-B/reduced/launch_analysis.json')
    events = {x['event']['name']:x['event'] for x in json.loads(gzip.decompress(paths['collectives'].read_bytes()))}
    skew = {x['name']:x for x in json.loads(paths['skew'].read_text())}
    launch = {x['name']:x for x in json.loads(paths['launch'].read_text())}
    assert len(skew) == len(launch) == 1901 and set(skew) == set(launch)
    steps = json.loads((r/'step_timeline.json').read_text())['0']
    rounds = []
    assigned = set()
    for i, step in enumerate(steps):
        end = steps[i+1]['target_embedding_ns'] if i+1 < len(steps) else step['sample_D2H_end_ns']
        names = [n for n in skew if step['target_embedding_ns'] <= int(Decimal(str(events[n]['ts']))*1000) < end]
        assigned.update(names)
        rounds.append(dict(round=i+1, operations=len(names),
                           latest_device_rank_counts=dict(collections.Counter(skew[n]['last_rank'] for n in names)),
                           latest_host_rank_counts=dict(collections.Counter(launch[n]['latest_host_rank'] for n in names)),
                           latest_host_equals_latest_device=sum(launch[n]['latest_host_rank'] == skew[n]['last_rank'] for n in names),
                           start_skew_sum_ms=sum(skew[n]['start_skew'] for n in names)/1000,
                           last_start_to_final_end_sum_ms=sum(skew[n]['last_to_end'] for n in names)/1000))
    assert assigned == set(skew), sorted(set(skew)-assigned)
    out=dict(original_inputs={k:dict(path=str(v.relative_to(p)),sha256=hashlib.sha256(v.read_bytes()).hexdigest()) for k,v in paths.items()},
             round_boundary_sha256=hashlib.sha256((r/'step_timeline.json').read_bytes()).hexdigest(),
             rounds=rounds,
             limits=['Existing Run249 profiler-ON host167 only; not numerical decomposition of OFF265ms/token.',
                     'Skew/tail sums overlap; EP MC2 operations are additional and not in these HCOM counts.',
                     'Latest host uses original launch analysis, device uses original collective reduction; sub100us ordering remains uncertain.'])
    (r/'round_readiness.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out['rounds'],indent=2))
