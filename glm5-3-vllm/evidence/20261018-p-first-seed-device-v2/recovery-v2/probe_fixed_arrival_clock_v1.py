"""Host CPU-only wake accuracy on the frozen schedule; no model requests."""
from pathlib import Path
import concurrent.futures, ctypes, errno, hashlib, json, os, socket, statistics, sys, time

assert socket.gethostname() == 'B5-5U-Atlas900RCKA3-33'
root = Path('/data/wio/glm5-3-vllm-execution/20261018-state-tail/prefix-count-live1-20261019/node33')
data = json.loads((root/'moe-prefix-gmm2-count-c4-fixed-arrival-v1/result.json').read_text())
assert data['completed'] and data['numeric_all_pass'] and not data['fixed_arrival_control']['timing_valid']
assert (root/'moe-prefix-gmm2-count-probe/disabled').exists() and not (root/'moe-prefix-gmm2-count-probe/enable').exists()
schedule = data['fixed_arrival_control']['source']['rows']; assert len(schedule) == 32

class Timespec(ctypes.Structure):
    _fields_ = [('tv_sec', ctypes.c_long), ('tv_nsec', ctypes.c_long)]

libc = ctypes.CDLL(None, use_errno=True)
sleep_until = libc.clock_nanosleep
sleep_until.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(Timespec), ctypes.c_void_p]
sleep_until.restype = ctypes.c_int

def wait_absolute(target):
    deadline = Timespec(target//1_000_000_000, target%1_000_000_000)
    while True:
        rc = sleep_until(time.CLOCK_MONOTONIC, 1, ctypes.byref(deadline), None)
        if rc == errno.EINTR: continue
        assert rc == 0, rc
        return

rows = []
for mode in ['relative_python_sleep', 'absolute_clock_nanosleep']:
    start = time.monotonic_ns()+200_000_000
    def one(item):
        target = start+item['offset_ns']
        if mode == 'relative_python_sleep':
            left = (target-time.monotonic_ns())/1e9
            if left > 0: time.sleep(left)
        else: wait_absolute(target)
        return dict(mode=mode, offset_s=item['offset_ns']/1e9, wake_lag_ms=(time.monotonic_ns()-target)/1e6)
    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as pool:
        rows.extend(pool.map(one, schedule))

def q(values):
    a = sorted(values); i = (len(a)-1)*.95
    return a[int(i)]+(a[min(int(i)+1,len(a)-1)]-a[int(i)])*(i%1)

summary = {mode: dict(p95_ms=q([z['wake_lag_ms'] for z in rows if z['mode'] == mode]), max_ms=max(z['wake_lag_ms'] for z in rows if z['mode'] == mode), median_ms=statistics.median(z['wake_lag_ms'] for z in rows if z['mode'] == mode)) for mode in ['relative_python_sleep', 'absolute_clock_nanosleep']}
cpu_quota = []
for line in Path('/proc/self/cgroup').read_text().splitlines():
    _, names, rel = line.split(':', 2)
    if 'cpu' not in names.split(','): continue
    p = Path('/sys/fs/cgroup/cpu')/rel.lstrip('/')
    while str(p).startswith('/sys/fs/cgroup/cpu'):
        cpu_quota.append(dict(path=str(p), values={n: (p/n).read_text() for n in ['cpu.cfs_quota_us', 'cpu.cfs_period_us', 'cpu.stat'] if (p/n).exists()}))
        p = p.parent
result = dict(completed=True, rows=rows, summary=summary, python=sys.version,
              timer_slack_ns=Path('/proc/self/timerslack_ns').read_text(), affinity_cpus=len(os.sched_getaffinity(0)), cpu_quota=cpu_quota,
              scope='One CPU-only pair of32 frozen offsets. No device allocation, model request, source change or inference timing. This establishes wake accuracy; real replay still needs unchanged dispatch gates under actual SSE traffic.')
print('GLM53_RESULT_JSON_BEGIN\n'+json.dumps(result)+'\nGLM53_RESULT_JSON_END')
