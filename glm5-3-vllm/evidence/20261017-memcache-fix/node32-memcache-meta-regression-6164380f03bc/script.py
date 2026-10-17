#!/usr/bin/env python3
"""Compare baseline and task-local loader fix using real MetaService shutdown."""
import json
from pathlib import Path
import socket
import subprocess
import time

ROOT = Path('/data/wio/glm5-3-vllm-execution/20261017-memcache/meta-regression-v1')
FIX = Path('/data/wio/glm5-3-vllm-execution/20261017-memcache/loader-fix-v1')
IMAGE = 'sha256:090eeee66e347509127601e565b971a0801701e32e4e865d89cc8dec72ca7d83'
TRACE = r'''
#include <execinfo.h>
#include <signal.h>
#include <unistd.h>
static void handler(int sig) {
    void *bt[96]; int n = backtrace(bt,96);
    write(2,"NATIVE_CRASH_BACKTRACE\n",23);
    backtrace_symbols_fd(bt,n,2);
    signal(sig,SIG_DFL); raise(sig);
}
void install_fault_trace(void) {
    signal(SIGABRT,handler); signal(SIGSEGV,handler); signal(SIGTRAP,handler);
}
'''
ENTRY = '''
import ctypes,json,pathlib,sys
from memcache_hybrid import MetaConfig,MetaService
ctypes.CDLL('/tests/trace.so').install_fault_trace()
pathlib.Path('/out/maps.txt').write_text(pathlib.Path('/proc/self/maps').read_text())
c=MetaConfig();c.meta_service_url='tcp://172.16.12.32:27410';c.config_store_url='tcp://172.16.12.32:27411';c.metrics_url='http://172.16.12.32:27412';c.ha_enable=False;c.log_level='info';c.log_output_target='screen'
print('META_SETUP',MetaService.setup(c),flush=True)
rc=MetaService.main()
print('META_MAIN_RETURN',rc,flush=True)
'''


def main():
    assert socket.gethostname() == 'B4-33U-Atlas900RCKA3-32'
    ROOT.mkdir(parents=True, exist_ok=False)
    (ROOT/'trace.c').write_text(TRACE)
    (ROOT/'meta.py').write_text(ENTRY)
    p = subprocess.run(['docker', 'run', '--rm', '--pull', 'never', '--network', 'none',
                        '-v', str(ROOT)+':/tests', '--entrypoint', 'gcc', IMAGE,
                        '-shared', '-fPIC', '-g', '-o', '/tests/trace.so', '/tests/trace.c'],
                       capture_output=True, text=True, timeout=60)
    assert p.returncode == 0, p.stderr
    records = []
    for case in ('baseline', 'deduplicated-1', 'deduplicated-2'):
        for port in (27410, 27411, 27412):
            with socket.socket() as s:
                s.bind(('172.16.12.32', port))
        out = ROOT/case
        out.mkdir()
        name = 'glm53-memcache-meta-regression-'+case+'-20261017'
        argv = ['docker', 'run', '--rm', '--pull', 'never', '--name', name, '--network', 'host',
                '--cpus', '4', '--memory', '4g', '--pids-limit', '512', '-e', 'OMP_NUM_THREADS=1',
                '-v', '/usr/local/Ascend/driver:/usr/local/Ascend/driver:ro',
                '-v', str(ROOT)+':/tests:ro', '-v', str(out)+':/out',
                '-e', 'ASCEND_PROCESS_LOG_PATH=/out/ascend-log']
        if case != 'baseline':
            argv += ['-v', str(FIX/'site-packages')+':/opt/memcache-runtime:ro',
                     '-e', 'PYTHONPATH=/opt/memcache-runtime']
        argv += ['--entrypoint', 'python3', IMAGE, '/tests/meta.py']
        with (out/'stdout.txt').open('w') as stdout, (out/'stderr.txt').open('w') as stderr:
            proc = subprocess.Popen(argv, stdout=stdout, stderr=stderr)
            ready = False
            deadline = time.monotonic()+100
            while proc.poll() is None and time.monotonic() < deadline:
                try:
                    with socket.create_connection(('172.16.12.32', 27410), timeout=.5):
                        pass
                    with socket.create_connection(('172.16.12.32', 27411), timeout=.5):
                        pass
                    ready = True
                    break
                except OSError:
                    time.sleep(1)
            if ready:
                subprocess.run(['docker', 'kill', '--signal', 'TERM', name], capture_output=True, timeout=10)
            try:
                rc = proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                subprocess.run(['docker', 'stop', '-t', '3', name], capture_output=True, timeout=10)
                rc = proc.wait(timeout=10)
        maps = (out/'maps.txt').read_text() if (out/'maps.txt').exists() else ''
        libs = sorted(set(s.split()[-1] for s in maps.splitlines() if 'libmf_' in s))
        records.append({'case': case, 'ready': ready, 'returncode': rc, 'libraries': libs,
                        'stderr_tail': (out/'stderr.txt').read_text()[-12000:]})
        (ROOT/'result.json').write_text(json.dumps(records, indent=2)+'\n')
    print(json.dumps(records, indent=2))
    assert records[0]['ready']
    assert all(r['ready'] and r['returncode'] == 0 for r in records[1:]), records


if __name__ == '__main__':
    main()
