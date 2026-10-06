"""Reduce immutable Run252 CPU counters. Does not import torch or run requests."""
import hashlib
import json
import re
from pathlib import Path


def reduce(root):
    raw = (root / 'cpu_clock_raw.json').read_bytes()
    v = json.loads(raw)
    guards = json.loads((root / 'guards_before.json').read_text())
    assert guards['D']['worker_pids'] == list(map(int, v['start']['rows']))
    a, b = v['start'], v['end']
    lower = (b['lo_ns'] - a['hi_ns']) / 1e9
    upper = (b['hi_ns'] - a['lo_ns']) / 1e9
    rows = []
    for pid, start in a['rows'].items():
        stop = b['rows'][pid]
        assert start['main']['start_ticks'] == stop['main']['start_ticks']
        rank = int(re.search(r'Worker_TP(\d+)_', guards['D']['worker_cmdline'][pid]).group(1))
        main = (stop['main']['runtime_ns'] - start['main']['runtime_ns']) / 1e9
        process = (stop['process']['cpu_ticks'] - start['process']['cpu_ticks']) / v['clock_ticks']
        # These snapshots are wider than the go/done window. Never substitute
        # their totals for the precisely bounded main-thread measurements.
        threads = []
        for tid, x in v['threads_before']['rows'][pid].items():
            y = v['threads_after']['rows'][pid].get(tid)
            if y is not None and x['start_ticks'] == y['start_ticks']:
                threads.append(dict(tid=int(tid), comm=x['comm'],
                                    cpu_s=(y['runtime_ns']-x['runtime_ns'])/1e9))
        before = v['threads_before']['rows'][pid][pid]
        idle_delta = (start['main']['runtime_ns'] - before['runtime_ns']) / 1e9
        assert idle_delta >= 0
        rows.append(dict(rank=rank, pid=int(pid), main_cpu_s=main,
                         main_cpu_wall_ratio_lower=main/upper,
                         main_cpu_wall_ratio_upper=main/lower,
                         process_cpu_s_100Hz=process,
                         pre_go_idle_main_cpu_s=idle_delta,
                         wider_top_threads=sorted(threads, key=lambda z: z['cpu_s'], reverse=True)[:5]))
    assert sorted(x['rank'] for x in rows) == list(range(16))
    result = json.loads((root/'cpuclock_result.json').read_text())
    assert result['token_ids'] == [785,1196,374,10156,264,3405,304,8452]
    assert result['chunks'] == [1,2,2,2,1] and result['prompt_tokens'] == 2334
    assert not result['profiler_active'] and any('hits_total' in k and n == 2334 for k,n in result['external_KV_delta'].items())
    state = json.loads((root/'state.json').read_text())
    phase = json.loads((root/'cpuclock.phase.json').read_text())
    assert state['status'] == 'completed' and phase['exit_code'] == 0 and not phase['timed_out']
    after = json.loads((root/'guards_after.json').read_text())
    assert after == guards
    intervals = [(y['lo_ns']-x['lo_ns'])/1e6 for x,y in zip(v['samples'],v['samples'][1:])]
    return dict(run='GLM-RUN-0252', raw_sha256=hashlib.sha256(raw).hexdigest(),
                wall_bounds_s=[lower,upper], clock_ticks=v['clock_ticks'],
                sched_schedstats=v['sched_schedstats'], sample_count=len(v['samples']),
                polling_interval_ms_range=[min(intervals),max(intervals)],
                observer_cpu_s=v['observer_poll_cpu_ns']/1e9,
                all_thread_snapshot_duration_ms=[(v[k]['hi_ns']-v[k]['lo_ns'])/1e6 for k in ['threads_before','threads_after']],
                pre_go_idle_wall_bounds_s=[(a['lo_ns']-v['threads_before']['hi_ns'])/1e9,(a['hi_ns']-v['threads_before']['lo_ns'])/1e9],
                request=result, workers=sorted(rows,key=lambda z:z['rank']),
                limits=[
                    'Main runtime is actual scheduled CPU time, including active polling. It is not a Python/function CPU profile.',
                    'Installed SHM SpinCondition can yield/poll for1s after a read. Main CPU occupancy alone cannot separate model frontend from RPC spin or runtime busy waits.',
                    'All main counters are unchanged over the pre-go idle interval. Wider post-response snapshots include SHM spin and must not be interpreted as more model computation.',
                    'sched_schedstats=0: runnable wait is unavailable; no wait-time calculation. Process ticks have100Hz resolution; threads overlap.',
                    'Window includes observer go acknowledgement, D POST/KV/response, metric fetch and SSH done acknowledgement. No per-round CPU accounting or exact Scheduler/Executor timing.',
                    'Observer uses0.324 CPU seconds over the window, not zero overhead. Request TPOT is diagnostic only, not a gain or SLA claim.'
                ])


if __name__ == '__main__':
    here = Path(__file__).resolve().parent
    root = here.parents[1]/'runs/GLM-RUN-0252'
    out = reduce(root)
    (here/'cpu_clock_summary.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:out[k] for k in ['wall_bounds_s','observer_cpu_s','sample_count']},indent=2))
