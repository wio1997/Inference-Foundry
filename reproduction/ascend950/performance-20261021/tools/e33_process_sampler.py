#!/usr/bin/env python3
"""Read task-container scheduling counters; never attach to or signal workers."""
import argparse
import datetime
import json
import os
from pathlib import Path
import signal
import time

def read(path):
    try:return Path(path).read_text().strip()
    except OSError:return None

def targets():
    rows=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        comm=read(p/'comm') or ''
        try:cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
        except OSError:continue
        role=None
        if 'VLLM::Worker' in cmd:role='worker'
        elif 'VLLM::EngineCore' in cmd:role='engine'
        elif 'VLLM::APIServer' in cmd:role='api'
        elif 'deepinfer_dp8_adapter_e33.py' in cmd and 'e33_process_sampler.py' not in cmd:role='di_adapter_or_worker'
        elif comm.startswith('VLLM'):role='vllm_other'
        if role:rows.append({'pid':int(p.name),'role':role,'comm':comm})
    return rows

def thread(pid,tid):
    root=Path('/proc')/str(pid)/'task'/str(tid);sched=read(root/'schedstat');status=read(root/'status')
    if sched is None:return None
    try:runtime,runqueue,slices=map(int,sched.split()[:3])
    except ValueError:return None
    result={'tid':int(tid),'runtime_ns':runtime,'runqueue_wait_ns':runqueue,'timeslices':slices}
    if status:
        wanted={'State','voluntary_ctxt_switches','nonvoluntary_ctxt_switches','Cpus_allowed_list','Mems_allowed_list'}
        for line in status.splitlines():
            key,_,value=line.partition(':')
            if key in wanted:result[key]=value.strip()
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);p.add_argument('--framework',required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--stop-file',type=Path,required=True)
    p.add_argument('--actors-file',type=Path,help='Explicit container-namespace worker/API PID identities, particularly for spawned DI workers.')
    p.add_argument('--interval',type=float,default=.05);p.add_argument('--all-threads-interval',type=float,default=5.0);p.add_argument('--duration-limit',type=float,default=1800)
    a=p.parse_args();assert a.interval>=.02 and a.duration_limit>0
    assert not a.out.exists() and not a.stop_file.exists(),'Refuse to overwrite or reuse a completed sampler'
    a.out.parent.mkdir(parents=True,exist_ok=True);ready=a.out.with_suffix('.ready.json');done=a.out.with_suffix('.done.json')
    stop=False
    def requested(sig,frame):
        nonlocal stop
        stop=True
    signal.signal(signal.SIGTERM,requested);signal.signal(signal.SIGINT,requested)
    started=time.monotonic();next_all=0;next_scan=0;actors=[];samples=0;thread_rows=0;cpu0=time.process_time();last_error=None
    with a.out.open('x',buffering=1024*1024) as fh:
        while not stop and not a.stop_file.exists() and time.monotonic()-started<a.duration_limit:
            t0=time.monotonic();all_threads=t0>=next_all
            if t0>=next_scan:
                if a.actors_file:
                    actors=json.loads(a.actors_file.read_text())
                    assert actors and all(isinstance(x.get('pid'),int) and x['pid']>1 for x in actors)
                else:actors=targets()
                next_scan=t0+1
            if all_threads:next_all=t0+a.all_threads_interval
            rows=[]
            for actor in actors:
                pid=actor['pid']
                if all_threads:
                    try:tids=[int(x.name) for x in (Path('/proc')/str(pid)/'task').iterdir() if x.name.isdigit()]
                    except OSError:tids=[]
                else:tids=[pid]
                threads=[x for tid in tids if (x:=thread(pid,tid)) is not None];thread_rows+=len(threads)
                rows.append({**actor,'threads':threads})
            row={'run_id':a.run_id,'framework':a.framework,'wall_ns':time.time_ns(),'monotonic_ns':time.perf_counter_ns(),
                 'sample_kind':'all_threads' if all_threads else 'main_threads','actors':rows,
                 'loadavg':read('/proc/loadavg'),'observer_sample_duration_ns':int((time.monotonic()-t0)*1e9)}
            if all_threads:
                row['cgroup_cpu_stat']={x:read(x) for x in ['/sys/fs/cgroup/cpu.stat','/sys/fs/cgroup/cpu/cpu.stat']}
            fh.write(json.dumps(row,separators=(',',':'))+'\n');samples+=1
            if samples==1:
                fh.flush();ready.write_text(json.dumps({'pid':os.getpid(),'run_id':a.run_id,'actors':actors,'ready_wall_ns':time.time_ns(),'kernel_schedstats':read('/proc/sys/kernel/sched_schedstats')})+'\n')
            time.sleep(max(0,a.interval-(time.monotonic()-t0)))
        fh.flush()
    result={'run_id':a.run_id,'pid':os.getpid(),'samples':samples,'thread_rows':thread_rows,'duration_s':time.monotonic()-started,
            'observer_process_cpu_s':time.process_time()-cpu0,'reason':'stop_requested' if stop or a.stop_file.exists() else 'duration_limit',
            'scope':'schedstat is cumulative scheduling evidence; 50ms sampling is not sub-millisecond phase timing','error':last_error}
    done.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)

if __name__=='__main__':main()
