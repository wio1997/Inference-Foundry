"""Read-only Linux process/thread CPU observer, controlled by go/done stdin."""
import json
import os
import select
import sys
import time
from pathlib import Path


def stat(pid, tid=None):
    p = Path('/proc/%d' % pid)
    if tid is not None:
        p = p/'task'/str(tid)
    parts = (p/'stat').read_text().rsplit(')',1)[1].split()
    runtime = int((p/'schedstat').read_text().split()[0]) if tid is not None else None
    return dict(runtime_ns=runtime, cpu_ticks=int(parts[11])+int(parts[12]),
                start_ticks=parts[19], comm=(p/'comm').read_text().strip())


def main():
    dest = Path(sys.argv[1]);pids = json.loads(sys.argv[2])
    assert len(pids)==16 and len(set(pids))==16
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    identities={p:stat(p)['start_ticks'] for p in pids}
    def snap():
        lo=time.monotonic_ns();rows={}
        for pid in pids:
            main=stat(pid,pid);process=stat(pid)
            assert main['start_ticks']==process['start_ticks']==identities[pid]
            rows[str(pid)]=dict(main=main,process=process)
        return dict(lo_ns=lo,hi_ns=time.monotonic_ns(),rows=rows)
    def threads():
        lo=time.monotonic_ns();rows={}
        for pid in pids:
            rows[str(pid)]={}
            for p in Path('/proc/%d/task'%pid).iterdir():
                try:rows[str(pid)][p.name]=stat(pid,int(p.name))
                except FileNotFoundError:pass
        return dict(lo_ns=lo,hi_ns=time.monotonic_ns(),rows=rows)
    before=threads()
    print(json.dumps(dict(ready=True,pid=os.getpid(),worker_pids=pids)),flush=True)
    assert sys.stdin.readline().strip()=='go'
    cpu_start=time.process_time_ns();start=snap();samples=[start]
    print(json.dumps(dict(go=True,start=start)),flush=True)
    deadline=time.monotonic()+90
    while True:
        ready,_,_=select.select([sys.stdin],[],[],0.02)
        if ready:
            assert sys.stdin.readline().strip()=='done'
            break
        assert time.monotonic()<deadline,'observer deadline'
        samples.append(snap())
    end=snap();cpu_end=time.process_time_ns();after=threads()
    out=dict(boot_id=boot,clock_ticks=os.sysconf('SC_CLK_TCK'),sched_schedstats=Path('/proc/sys/kernel/sched_schedstats').read_text().strip(),
             start=start,end=end,samples=samples,threads_before=before,threads_after=after,observer_poll_cpu_ns=cpu_end-cpu_start)
    (dest/'cpu_clock_raw.json').write_text(json.dumps(out)+'\n')
    print(json.dumps(dict(done=True,window_ns=end['hi_ns']-start['lo_ns'],observer_poll_cpu_ns=cpu_end-cpu_start)),flush=True)


if __name__=='__main__':main()
