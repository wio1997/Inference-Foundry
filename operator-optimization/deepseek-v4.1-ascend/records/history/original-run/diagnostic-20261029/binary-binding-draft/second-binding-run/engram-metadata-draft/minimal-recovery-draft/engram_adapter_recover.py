import sys
if sys.flags.optimize:
    raise RuntimeError('optimized Python is forbidden for diagnostic safety checks')
'Reviewable recovery helper for the owned binding diagnostic. Not run automatically.'
from pathlib import Path
import json, os, shlex, subprocess, sys, time
if __name__ != '__main__' or sys.argv[1:] != ['--execute']:
    print('Review draft: no operations. Use --execute only to recover the owned binding diagnostic.')
    raise SystemExit(0)
# Enabled for the user-authorized isolated diagnostic after controller review.
HERE = Path(__file__).resolve().parent
SSH = '/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh'
BASE = 'wio-vllm-v41-repro169-20261028'
BASE_ID = 'b00f4d2018e3d495d876b6e396ba9579e78b47a97111b99c27d6678e31039092'
RUN_ID = os.environ.get('WIO_BINDING_RUN_ID', '')
DIAG = 'wio-vllm-v41-engram-' + RUN_ID
STAGE = 'wio-vllm-v41-repro169-stage-20261028'
RUN_ID = os.environ.get('WIO_BINDING_RUN_ID', '')
if not (len(RUN_ID) == 32 and all((c in '0123456789abcdef' for c in RUN_ID))):
    raise AssertionError('unique watchdog run ID required')
ROOT = '/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029/engram-diagnostic-' + RUN_ID
WORK = '/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029/engram-diagnostic-' + RUN_ID
LOCK_ROOT = '/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029/binding-diagnostic'
REMOTE = 'import sys\nif sys.flags.optimize:\n    raise RuntimeError(\'optimized Python is forbidden for diagnostic safety checks\')\nimport json, subprocess\nfence_fd = acquire_owner_fence(LOCK_ROOT)\nensure_owner_fence(fence_fd)\nmarker_dir=\'/work/\'+LOCK_ROOT[len(HOST_WORK_ROOT):]\nmarker=marker_dir+\'/cancel-\'+RUN_ID\ncreate="import os,sys; sys.flags.optimize and sys.exit(4); p=sys.argv[1]; fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL|getattr(os,\'O_NOFOLLOW\',0),0o600); os.write(fd,b\'cancelled\'); os.fsync(fd); os.close(fd)"\nr=subprocess.run([\'docker\',\'exec\',STAGE,\'python3\',\'-c\',create,marker],capture_output=True,timeout=20)\nensure_owner_fence(fence_fd)\nif r.returncode:\n    probe="import os,sys,stat; sys.flags.optimize and sys.exit(4); s=os.lstat(sys.argv[1]); sys.exit(0 if stat.S_ISREG(s.st_mode) and s.st_nlink==1 else 2)"\n    if subprocess.run([\'docker\',\'exec\',STAGE,\'python3\',\'-c\',probe,marker],capture_output=True,timeout=20).returncode:\n        raise RuntimeError(\'cannot persist cancellation marker\')\nensure_owner_fence(fence_fd)\n\n\ndef run(a, timeout=60):\n    ensure_owner_fence(fence_fd)\n    out = subprocess.check_output(a, text=True, timeout=timeout)\n    ensure_owner_fence(fence_fd)\n    return out\n\ndef saved_identity():\n    for name in (\'launch.json\', \'identity-\' + RUN_ID + \'.json\'):\n        ensure_owner_fence(fence_fd)\n        r = subprocess.run([\'docker\', \'exec\', STAGE, \'cat\', WORK + \'/\' + name], capture_output=True, text=True, timeout=30)\n        if r.returncode:\n            continue\n        try:\n            x = json.loads(r.stdout)\n        except (ValueError, TypeError):\n            continue\n        if x.get(\'baseline_id\') == BASE_ID and x.get(\'diagnostic_name\') == DIAG and (x.get(\'binding_run_id\') == RUN_ID):\n            return x\n    return None\nlaunch = saved_identity()\nb = json.loads(run([\'docker\', \'inspect\', BASE_ID]))[0]\nif not b[\'Id\'] == BASE_ID:\n    raise AssertionError()\nrows = run([\'docker\', \'ps\', \'-a\', \'--no-trunc\', \'--format\', \'{{.ID}}\\t{{.Names}}\']).splitlines()\nfound = []\nfor row in rows:\n    cols = row.split(\'\\t\')\n    if len(cols) != 2:\n        raise ValueError(\'malformed docker container list\')\n    if cols[1] == DIAG:\n        found.append(cols[0])\nif len(found) > 1:\n    raise ValueError(\'ambiguous diagnostic container name\')\nown_before = set()\nif found:\n    diag_id = found[0]\n    d = json.loads(run([\'docker\', \'inspect\', diag_id]))[0]\n    if d[\'Id\'] != diag_id or d[\'Config\'][\'Labels\'].get(\'wio.task\') != \'operator-ceiling169-20261029\' or d[\'Config\'][\'Labels\'].get(\'wio.binding_run_id\') != RUN_ID:\n        raise ValueError(\'diagnostic identity or owner label mismatch\')\n    if launch is not None and launch.get(\'diagnostic_id\') != diag_id:\n        raise ValueError(\'diagnostic identity does not match launch record\')\n    if d[\'State\'][\'Running\']:\n        own_before = {int(x.strip()) for x in run([\'docker\', \'top\', diag_id, \'-eo\', \'pid\']).splitlines() if x.strip().isdigit()}\n        if not own_before:\n            raise ValueError(\'cannot identify own diagnostic processes before stop\')\n        run([\'docker\', \'stop\', \'--time\', \'45\', diag_id], timeout=100)\n    stopped = json.loads(run([\'docker\', \'inspect\', diag_id]))[0]\n    if stopped[\'Id\'] != diag_id or stopped[\'State\'][\'Running\']:\n        raise ValueError(\'diagnostic stop not confirmed\')\nb = json.loads(run([\'docker\', \'inspect\', BASE_ID]))[0]\nif not b[\'Id\'] == BASE_ID:\n    raise AssertionError()\nif b[\'State\'][\'Running\']:\n    print(json.dumps({\'state\': \'BASELINE_ALREADY_RUNNING\', \'identity_source\': \'saved\' if launch else \'labels_or_missing\'}))\n    raise SystemExit(0)\nreference = launch.get(\'hbm_reference\') if launch else None\nrelease = wait_for_empirical_release(lambda : run([\'docker\', \'exec\', STAGE, \'npu-smi\', \'info\']), reference)\nif release[\'satisfies_policy\']:\n    run([\'docker\', \'start\', BASE_ID])\n    out = {\'state\': \'BASELINE_RESTORING\'}\nelif release.get(\'pids\') and own_before & release[\'pids\']:\n    out = {\'state\': \'RESTORE_DEFERRED_OWNED_WORKERS\', \'pids\': sorted(own_before & release[\'pids\'])}\nelif release.get(\'pids\'):\n    out = {\'state\': \'RESTORE_DEFERRED_OTHER_NPU_OWNER\', \'other_pids\': sorted(release[\'pids\'])}\nelse:\n    out = {\'state\': \'RESTORE_DEFERRED_UNATTRIBUTED_HBM\', \'reason\': release[\'reason\']}\nout[\'identity_source\'] = \'saved\' if launch else \'labels_or_missing\'\nprint(json.dumps(out))\n'
HEALTH = 'import sys\nif sys.flags.optimize:\n    raise RuntimeError(\'optimized Python is forbidden for diagnostic safety checks\')\nimport json, subprocess, urllib.request\nb = json.loads(subprocess.check_output([\'docker\', \'inspect\', BASE_ID], text=True))[0]\nif not b[\'Id\'] == BASE_ID:\n    raise AssertionError()\ncheck = "import sys\\nif sys.flags.optimize:\\n    raise RuntimeError(\'optimized Python is forbidden for diagnostic safety checks\')\\nimport urllib.request\\nop = urllib.request.build_opener(urllib.request.ProxyHandler({}))\\nprint(op.open(\'http://127.0.0.1:8172/health\', timeout=5).status)\\n"\nr = subprocess.run([\'docker\', \'exec\', BASE_ID, \'python3\', \'-c\', check], capture_output=True, text=True, timeout=15) if b[\'State\'][\'Running\'] else None\nprint(json.dumps({\'running\': b[\'State\'][\'Running\'], \'health\': int(r.stdout.strip()) if r and r.returncode == 0 else None}))\n'

def ssh(code, timeout):
    import hashlib
    if not hashlib.sha256((HERE.parent.parent / 'remote_guard.py').read_bytes()).hexdigest() == 'f475694ae5f660394d60d5239fa6f2ec640d64ab09d67401827f4bfee83383d7':
        raise AssertionError()
    pre = '\n'.join((k + '=' + repr(v) for (k, v) in dict(BASE=BASE, BASE_ID=BASE_ID, DIAG=DIAG, STAGE=STAGE, WORK=WORK, ROOT=ROOT, LOCK_ROOT=LOCK_ROOT, RUN_ID=RUN_ID).items())) + '\n'
    r = subprocess.run([SSH, shlex.join(['python3', '-c', pre + (HERE.parent.parent / 'remote_guard.py').read_text() + '\nHOLDER_CODE = "import sys\\nif sys.flags.optimize: raise RuntimeError(\'optimized Python forbidden\')\\n" + HOLDER_CODE\n' + code])], capture_output=True, text=True, timeout=timeout)
    if r.returncode:
        raise RuntimeError(r.stderr[-2000:] + r.stdout[-1000:])
    return json.loads(r.stdout)

def save(obj):
    obj['checked_unix'] = time.time()
    (HERE / ('manual-recovery-state-' + RUN_ID + '.json')).write_text(json.dumps(obj, indent=2) + '\n')
    print(json.dumps(obj, ensure_ascii=False), flush=True)
try:
    result = ssh(REMOTE, 400)
    save(result)
    if result['state'] in ('RESTORE_DEFERRED_OWNED_WORKERS', 'RESTORE_DEFERRED_OTHER_NPU_OWNER', 'RESTORE_DEFERRED_UNATTRIBUTED_HBM'):
        raise SystemExit(2)
    deadline = time.monotonic() + 2400
    while time.monotonic() < deadline:
        try:
            health = ssh(HEALTH, 30)
            if health['running'] and health['health'] == 200:
                save({'state': 'BASELINE_HEALTHY', 'health': health})
                raise SystemExit(0)
        except Exception as e:
            save({'state': 'BASELINE_HEALTH_PENDING', 'error': repr(e)})
        time.sleep(20)
    save({'state': 'BASELINE_HEALTH_TIMEOUT'})
    raise SystemExit(3)
except Exception as e:
    save({'state': 'RECOVERY_ERROR', 'error': repr(e)})
    raise
