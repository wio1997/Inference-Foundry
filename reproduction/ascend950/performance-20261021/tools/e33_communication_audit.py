#!/usr/bin/env python3
"""Match observed eight-rank HCCL instances and expose arrival/finish skew."""
import collections
import csv
import hashlib
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[1]


def union(intervals):
    total=0;last=None
    for a,b in sorted(intervals):
        if last is None or a>last:total+=b-a;last=b
        elif b>last:total+=b-last;last=b
    return total


def percentiles(values):
    values=sorted(values)
    if not values:return {}
    def p(q):
        x=(len(values)-1)*q;i=int(x);r=x-i
        return (values[i]*(1-r)+values[min(i+1,len(values)-1)]*r)/1e6
    return dict(p50_ms=p(.5),p90_ms=p(.9),p99_ms=p(.99),max_ms=values[-1]/1e6)


def main():
    cells=[];flat=[]
    for folder in sorted((ROOT/'evidence/E33-calibrated-census').glob('*-calibrated1')):
        by_name=collections.defaultdict(dict);sources=[]
        for rank in range(8):
            p=folder/f'rank{rank:02d}-communication-ops.json';rows=json.loads(p.read_text())
            sources.append(dict(rank=rank,path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
            for op in rows:
                assert rank==op['deviceId'] and rank not in by_name[op['name']]
                by_name[op['name']][rank]=op
        unmatched={k:sorted(v) for k,v in by_name.items() if set(v)!=set(range(8))}
        matched=[];groups=collections.defaultdict(list)
        for name,ops in sorted(by_name.items()):
            if set(ops)!=set(range(8)):continue
            assert {r['rankSize'] for r in ops.values()}=={8}
            starts=[ops[r]['startNs'] for r in range(8)];ends=[ops[r]['endNs'] for r in range(8)]
            match=re.fullmatch(r'hcom_(.*?)_AicpuKernel_(\d+)_(\d+)_(\d+)',name)
            assert match,name
            kind,group,sequence,instance=match.groups()
            row=dict(name=name,kind=kind,group_id_in_name=group,sequence=int(sequence),instance=int(instance),
                     first_start_ns=min(starts),last_start_ns=max(starts),first_end_ns=min(ends),last_end_ns=max(ends),
                     arrival_skew_ns=max(starts)-min(starts),finish_skew_ns=max(ends)-min(ends),
                     collective_envelope_ns=max(ends)-min(starts),all_eight_active_intersection_ns=max(0,min(ends)-max(starts)),
                     late_arriving_rank=max(range(8),key=lambda r:starts[r]),last_finishing_rank=max(range(8),key=lambda r:ends[r]),
                     per_rank_start_ns=starts,per_rank_end_ns=ends,per_rank_duration_ns=[b-a for a,b in zip(starts,ends)],
                     raw_count_fields=[ops[r]['count'] for r in range(8)])
            matched.append(row);groups[kind].append(row)
            flat.append(dict(run_id=folder.name.removesuffix('-calibrated1'),name=name,kind=kind,arrival_skew_ms=row['arrival_skew_ns']/1e6,
                             finish_skew_ms=row['finish_skew_ns']/1e6,envelope_ms=row['collective_envelope_ns']/1e6,
                             late_arriving_rank=row['late_arriving_rank'],last_finishing_rank=row['last_finishing_rank']))
        summaries={}
        for kind,rs in groups.items():
            summaries[kind]=dict(instances=len(rs),arrival_skew=percentiles([x['arrival_skew_ns'] for x in rs]),
                                 finish_skew=percentiles([x['finish_skew_ns'] for x in rs]),
                                 late_arriving_ranks=dict(collections.Counter(x['late_arriving_rank'] for x in rs)),
                                 last_finishing_ranks=dict(collections.Counter(x['last_finishing_rank'] for x in rs)),
                                 first_to_last_arrival_interval_union_s=union([(x['first_start_ns'],x['last_start_ns']) for x in rs])/1e9,
                                 first_to_last_finish_interval_union_s=union([(x['first_end_ns'],x['last_end_ns']) for x in rs])/1e9)
            summaries[kind]['instances_with_end_before_other_rank_recorded_start']=sum(x['first_end_ns']<x['last_start_ns'] for x in rs)
            summaries[kind]['max_recorded_nonoverlap_ns']=max((max(0,x['last_start_ns']-x['first_end_ns']) for x in rs),default=0)
        cell=dict(run_id=folder.name.removesuffix('-calibrated1'),sources=sources,unique_names=len(by_name),matched_all_eight=len(matched),unmatched=unmatched,
                  summaries=summaries,instances=matched,cross_device_timing_qualified=False,
                  limits=['Matching uses the native group/sequence/instance name and rankSize8; no unmatched operations are silently discarded.',
                          'Arrival and finish skew are observed timing differences, not network transfer time or independent optimization savings.',
                          'Intervals can overlap across collectives/streams; their durations cannot be summed as service wall time.',
                          'DI MegaMoE and Engram use custom fused communication inside native kernels. Sparse HCCL OP records do not mean zero or little DI communication.',
                          'Some matched operations finish on one rank before the recorded start on another rank. Device clock mapping and native event-boundary semantics need independent validation; the current data must not be used for slow-rank attribution or precise cross-rank wait savings.',
                          'The raw count field is retained without converting it to bytes because datatype/count semantics were not independently resolved for every operation.'])
        assert not unmatched
        cells.append(cell)
    (ROOT/'data/E33-eight-rank-communication.json').write_text(json.dumps(dict(cells=cells),indent=2)+'\n')
    with (ROOT/'data/E33-eight-rank-communication.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    for c in cells:
        print(c['run_id'],'instances',c['matched_all_eight'])
        for kind,s in c['summaries'].items():print(kind,s)


if __name__=='__main__':main()
