import argparse
import re
import subprocess
import time
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--dataset', required=True)
p.add_argument('--out', required=True)
p.add_argument('--offset', type=int, default=0)
p.add_argument('--limit', type=int, default=4)
p.add_argument('--concurrency', type=int, default=1)
p.add_argument('--max-tokens', type=int, default=128)
a = p.parse_args()
root = Path(a.out)
root.mkdir(parents=True, exist_ok=True)
ps = subprocess.check_output(['docker', 'exec', 'dsv4ab', 'ps', '-eo', 'pid,args'], text=True)
match = re.search(r'^\s*(\d+)\s+VLLM::Worker_TP0_EP0\s*$', ps, re.M)
if not match:
    raise RuntimeError('current TP0/EP0 worker not found')
pid = match.group(1)
(root / 'worker_pid.txt').write_text(pid + '\n')
prof_cmd = ['docker', 'exec', '-i', 'dsv4ab', 'msprof', '--dynamic=on', f'--pid={pid}',
            f'--output={root}/raw', '--runtime-api=on', '--task-time=on', '--ai-core=on']
bench_cmd = ['docker', 'exec', 'dsv4ab', 'python3',
             '/data/wio/Inference_Foundry/scripts/bench.py', '--dataset', a.dataset,
             '--out', str(root / 'bench.json'), '--offset', str(a.offset),
             '--limit', str(a.limit), '--concurrency', str(a.concurrency),
             '--max-tokens', str(a.max_tokens)]
(root / 'commands.txt').write_text(' '.join(prof_cmd) + '\n' + ' '.join(bench_cmd) + '\n')
with (root / 'msprof.log').open('w') as profiler_log, (root / 'bench.log').open('w') as bench_log:
    profiler = subprocess.Popen(prof_cmd, stdin=subprocess.PIPE, stdout=profiler_log,
                                stderr=subprocess.STDOUT, text=True)
    time.sleep(8)
    if profiler.poll() is not None:
        raise RuntimeError(f'msprof exited before start: {profiler.returncode}')
    profiler.stdin.write('start\n')
    profiler.stdin.flush()
    time.sleep(3)
    result = subprocess.run(bench_cmd, stdout=bench_log, stderr=subprocess.STDOUT)
    profiler.stdin.write('stop\n')
    profiler.stdin.flush()
    time.sleep(6)
    profiler.stdin.write('quit\n')
    profiler.stdin.flush()
    profiler.wait(timeout=240)
    if result.returncode:
        raise RuntimeError(f'benchmark failed: {result.returncode}')
    if profiler.returncode:
        raise RuntimeError(f'msprof failed: {profiler.returncode}')
print((root / 'bench.log').read_text())
print((root / 'msprof.log').read_text()[-2000:])
