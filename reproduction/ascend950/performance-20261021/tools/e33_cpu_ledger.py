#!/usr/bin/env python3
"""Bound existing cumulative per-thread samples by the measured HTTP window."""
import collections
import csv
import hashlib
import json
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[1]


def bounds(observations,start,end,key):
    # A row's counters were read between sample entry and completion.  Keep
    # this uncertainty rather than assigning all reads the final timestamp.
    before_start=[x for x in observations if x['sample_end_ns']<=start]
    after_start=[x for x in observations if x['sample_start_ns']>=start]
    before_end=[x for x in observations if x['sample_end_ns']<=end]
    after_end=[x for x in observations if x['sample_start_ns']>=end]
    if not all([before_start,after_start,before_end,after_end]):return None
    a=max(before_start,key=lambda x:x['sample_end_ns'])
    b=min(after_start,key=lambda x:x['sample_start_ns'])
    c=max(before_end,key=lambda x:x['sample_end_ns'])
    d=min(after_end,key=lambda x:x['sample_start_ns'])
    return dict(bracket_delta_low=max(0,c[key]-b[key]),bracket_delta_high=d[key]-a[key],
                start_sample_bracket_ns=[a['sample_start_ns'],b['sample_end_ns']],
                end_sample_bracket_ns=[c['sample_start_ns'],d['sample_end_ns']],
                max_boundary_bracket_ns=max(b['sample_end_ns']-a['sample_start_ns'],d['sample_end_ns']-c['sample_start_ns']))


def main():
    cells=[];flat=[]
    for p in sorted((ROOT/'evidence/E33-calibrated-census').glob('*/summary.json')):
        summary=json.loads(p.read_text());run=summary['run_id'];start,end=summary['client_window_wall_ns']
        s=ROOT/'evidence/E33-cpu-samples/evidence-e33/control'/run/'scheduler.jsonl'
        ready=json.loads(s.with_suffix('.ready.json').read_text());done=json.loads(s.with_suffix('.done.json').read_text())
        rows=[json.loads(line) for line in s.read_text().splitlines()]
        main_threads=collections.defaultdict(list);all_threads=collections.defaultdict(list)
        rank_by_pid={r['pid']:r['rank'] for r in summary['rank_summaries']};identities={}
        for row in rows:
            assert row['run_id']==run
            for actor in row['actors']:
                for thread in actor['threads']:
                    key=(actor['pid'],thread['tid']);identities[key]=dict(pid=actor['pid'],tid=thread['tid'],role=actor['role'],rank=rank_by_pid.get(actor['pid']))
                    observation=dict(sample_start_ns=row['wall_ns']-row['observer_sample_duration_ns'],sample_end_ns=row['wall_ns'],**thread)
                    all_threads[key].append(observation)
                    if actor['pid']==thread['tid']:main_threads[key].append(observation)
        result=[]
        for key,observations in sorted(main_threads.items()):
            runtime=bounds(observations,start,end,'runtime_ns');wait=bounds(observations,start,end,'runqueue_wait_ns')
            negative=[i for i in range(1,len(observations)) if any(observations[i][k]<observations[i-1][k] for k in ['runtime_ns','runqueue_wait_ns','timeslices'])]
            assert not negative
            result.append(dict(**identities[key],samples=len(observations),runtime=runtime,runqueue_wait=wait,
                               observed_runqueue_counter_increases=observations[-1]['runqueue_wait_ns']>observations[0]['runqueue_wait_ns'],
                               cpus_allowed_list=observations[-1].get('Cpus_allowed_list')))
            if runtime and wait:
                flat.append(dict(run_id=run,**identities[key],cpu_s_low=runtime['bracket_delta_low']/1e9,cpu_s_high=runtime['bracket_delta_high']/1e9,
                                 runqueue_wait_ms_low=wait['bracket_delta_low']/1e6,runqueue_wait_ms_high=wait['bracket_delta_high']/1e6,
                                 max_boundary_bracket_ms=runtime['max_boundary_bracket_ns']/1e6))
        groups={}
        for role in sorted({x['role'] for x in result}):
            selected=[x for x in result if x['role']==role]
            groups[role]=dict(main_thread_count=len(selected),all_boundaries_available=all(x['runtime'] is not None and x['runqueue_wait'] is not None for x in selected),
                              cpu_s_low=sum(x['runtime']['bracket_delta_low'] for x in selected if x['runtime'])/1e9,
                              cpu_s_high=sum(x['runtime']['bracket_delta_high'] for x in selected if x['runtime'])/1e9,
                              runqueue_wait_ms_low=sum(x['runqueue_wait']['bracket_delta_low'] for x in selected if x['runqueue_wait'])/1e6,
                              runqueue_wait_ms_high=sum(x['runqueue_wait']['bracket_delta_high'] for x in selected if x['runqueue_wait'])/1e6)
        cells.append(dict(run_id=run,client_window_wall_ns=[start,end],source=str(s.relative_to(ROOT)),source_sha256=hashlib.sha256(s.read_bytes()).hexdigest(),
                          samples=len(rows),full_thread_samples=sum(x['sample_kind']=='all_threads' for x in rows),
                          kernel_schedstats_at_capture=ready['kernel_schedstats'],sampler=done,
                          role_groups=groups,main_threads=result,total_observed_tids=len(all_threads),
                          limits=['CPU seconds summed across threads are not service wall seconds.',
                                  'Ranges bracket cumulative counter deltas using observed sample boundaries; scheduler accounting updates can lag execution.',
                                  'Main threads sampled about every 50ms; full-thread inventories about every 5s. This is not a sub-millisecond scheduling trace.',
                                  'Non-runnable sleep, device wait, GIL wait and framework queueing are not the runqueue-wait counter.',
                                  'A zero sched_schedstats switch alone does not disable CONFIG_SCHED_INFO or prove per-task counters unavailable.']))
    report=dict(created_wall_ns=time.time_ns(),cells=cells,
                interpretation_correction=dict(previous='Per-thread schedstat runqueue counters treated as unavailable solely because sched_schedstats=0.',
                                               corrected='Counters are present, monotone and increase in the captures. Running Linux6.6 config has CONFIG_SCHED_INFO=y. Upstream sched_info_on() tests this compile setting, separately from global sched_schedstats.',
                                               reference='https://github.com/torvalds/linux/blob/v6.6/include/linux/sched/stat.h',
                                               definition='https://www.kernel.org/doc/html/latest/scheduler/sched-stats.html'))
    (ROOT/'data/E33-cpu-ledger.json').write_text(json.dumps(report,indent=2)+'\n')
    with (ROOT/'data/E33-cpu-main-threads.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    for c in cells:print(c['run_id'],json.dumps(c['role_groups']))


if __name__=='__main__':main()
