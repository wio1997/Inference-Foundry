"""20 s procfs counters for two worker processes, NOT a Python profiler.

Execute inside the user's baseline container only when the intended workload is
actually active. This script never launches a request, NPU work, or another model.
"""
import datetime
import json
import os
from pathlib import Path
import sys
import time
import uuid

APP = Path('/data/tiankuan/wio/projects/910c-deepseek-v4.1-flash')
OUT = Path(os.environ.get('WIO_PROC_OUT', str(APP/'logs'/('enginecore-proc-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8]+'.json'))))
if not OUT.resolve().is_relative_to((APP/'logs').resolve()):
    raise ValueError('proc sample must remain under the task runtime logs directory')
DURATION = 20.0
INTERVAL = 0.2

def read_stat(path):
    raw=path.read_text()
    end=raw.rfind(')')
    if end<0: raise ValueError('invalid proc stat')
    fields=raw[end+2:].split()
    return {'state':fields[0], 'utime_ticks':int(fields[11]), 'stime_ticks':int(fields[12]),
            'starttime_ticks':int(fields[19])}

def identify():
    found={}
    for p in Path('/proc').iterdir():
        if not p.name.isdigit(): continue
        try: name=(p/'cmdline').read_bytes().split(b'\0',1)[0].decode('ascii').strip()
        except (OSError,UnicodeDecodeError): continue
        if name in ('VLLM::Worker_DP0_TP0_EP0','VLLM::Worker_DP1_TP0_EP8'):
            if name in found: raise RuntimeError('duplicate engine name '+name)
            found[name]={'pid':int(p.name),'identity':read_stat(p/'stat')}
    if set(found)!={'VLLM::Worker_DP0_TP0_EP0','VLLM::Worker_DP1_TP0_EP8'}:
        raise RuntimeError('expected exactly DP0 and DP1: '+repr(sorted(found)))
    return found

def read_one(info):
    pid=info['pid'];p=Path('/proc')/str(pid)
    group=read_stat(p/'stat')
    if group['starttime_ticks']!=info['identity']['starttime_ticks']:
        raise RuntimeError('PID reused or engine restarted: '+str(pid))
    # /proc/PID/task/PID is only the main thread; group stat is process aggregate.
    thread=p/'task'/str(pid)
    leader=read_stat(thread/'stat')
    result={'process':group,'leader_thread':leader}
    for field in ('wchan','schedstat'):
        try: result['leader_'+field]=(thread/field).read_text().strip()
        except OSError as exc: result['leader_'+field+'_error']=str(exc)
    return result

def main():
    info=identify()
    record={'scope':'two worker processes; no Python function stacks',
            'duration_s':DURATION,'interval_s':INTERVAL,
            'clock_ticks_per_second':os.sysconf('SC_CLK_TCK'),
            'started_wall_ns':time.time_ns(),'started_monotonic_ns':time.monotonic_ns(),
            'identity':info,'samples':[],'complete':False}
    deadline=time.monotonic()+DURATION
    try:
        while time.monotonic()<deadline:
            t=time.monotonic_ns()
            sample={'monotonic_ns':t,'wall_ns':time.time_ns(),'engines':{}}
            for name in sorted(info): sample['engines'][name]=read_one(info[name])
            record['samples'].append(sample)
            remaining=deadline-time.monotonic()
            if remaining>0: time.sleep(min(INTERVAL,remaining))
        record['complete']=True
    except BaseException as exc:
        record['error']=repr(exc)
        raise
    finally:
        record['ended_monotonic_ns']=time.monotonic_ns()
        fd=os.open(OUT,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        with os.fdopen(fd,'w') as file: json.dump(record,file,ensure_ascii=False,indent=2)
        print(str(OUT),file=sys.stderr,flush=True)

if __name__=='__main__': main()
