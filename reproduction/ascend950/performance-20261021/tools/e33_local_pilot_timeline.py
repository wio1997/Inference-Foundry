#!/usr/bin/env python3
"""Use the already downloaded lower-overhead real eight-rank capture."""
import csv
import collections
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re
from e33_clock_map import ClockMap
from e33_native_timeline import family,partition

ROOT=Path(__file__).resolve().parents[1]


def union(intervals):
    total=0;last=None
    for a,b in sorted(intervals):
        if last is None or a>last:total+=b-a;last=b
        elif b>last:total+=b-last;last=b
    return total


def main():
    prior=ROOT/'evidence/evidence-e33/analysis/E33-V-L32K-C32-NORMAL-P-audit1/summary.json'
    summary=json.loads(prior.read_text());start,end=summary['client_window_wall_ns']
    root=ROOT/'evidence/E33-L0-local-selected/evidence-e33/vllm/profiles'
    out=ROOT/'data/E33-V-NORMAL-L0-timeline-calibrated';out.mkdir(exist_ok=False);results=[]
    for old in summary['rank_summaries']:
        rank=old['rank'];profile=root/Path(old['profile_dir']).name;clock=ClockMap.from_profile(profile)
        p=profile/'ASCEND_PROFILER_OUTPUT/kernel_details.csv';intervals=collections.defaultdict(list);stats=collections.defaultdict(lambda:dict(count=0,sum_duration_ns=0))
        checks={}
        for boundary in ['start','end']:
            h=next(x for x in old['host_anchors'] if x['extra']['boundary']==boundary)
            v=clock.wall_ns(old['marker'][boundary+'Ns']);checks[boundary+'_marker_in_host_bracket']=h['wall_ns']<=v<=h['wall_end_ns']
        checks['entire_http_window_inside_marker']=clock.wall_ns(old['marker']['startNs'])<=start<=end<=clock.wall_ns(old['marker']['endNs'])
        with p.open() as f:
            for row in csv.DictReader(f):
                a=int(Decimal(row['Start Time(us)'].strip())*1000);b=a+int(Decimal(row['Duration(us)'].strip())*1000)
                a=max(start,clock.wall_ns(a));b=min(end,clock.wall_ns(b))
                if b<=a:continue
                category=family(row['Name'],row['Type']);intervals[category].append((a,b));g=stats[category];g['count']+=1;g['sum_duration_ns']+=b-a
        combinations=partition(intervals,start,end)
        for cat,s in stats.items():s['union_ns']=union(intervals[cat]);s['exclusive_ns']=next((x['wall_ns'] for x in combinations if x['families']==[cat]),0)
        checks['disjoint_wall_partition_conserved']=sum(x['wall_ns'] for x in combinations)==end-start
        assert all(checks.values()),(rank,checks)
        result=dict(rank=rank,profile=profile.name,raw_kernel_csv_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),clock=clock.as_dict(),checks=checks,
                    observed_kernel_records=sum(x['count'] for x in stats.values()),families=dict(stats),disjoint_wall_partition=combinations)
        (out/f'rank{rank:02d}.json').write_text(json.dumps(result,indent=2)+'\n');results.append(result)
        print('rank',rank,'checks_pass',all(checks.values()),'records',result['observed_kernel_records'],flush=True)
    result=dict(source_summary=str(prior.relative_to(ROOT)),source_summary_sha256=hashlib.sha256(prior.read_bytes()).hexdigest(),
                run_id=summary['run_id'],client_window_wall_ns=[start,end],client_duration_s=(end-start)/1e9,
                profile_output_tps=8192/((end-start)/1e9),ranks=results,
                limits=['This is the original real Level0 normal capture, not a substitute for the detailed hot capture or outstanding controls.',
                        'Level0 and Level1 have different CANN API coverage. Exported API record counts alone cannot prove changed execution or executor-cache misses.',
                        'Name-based families group observed kernels, not internal fused phases. HCCL wall intervals include waiting.'])
    (ROOT/'data/E33-lower-overhead-normal-timeline.json').write_text(json.dumps(result,indent=2)+'\n')
    print('all8 lower-overhead native clocks and wall partitions pass')


if __name__=='__main__':main()
