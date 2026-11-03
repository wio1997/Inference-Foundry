import sys
if sys.flags.optimize:
    raise RuntimeError('optimized Python is forbidden for diagnostic safety checks')
"""Bounded supervisor: wait for this diagnostic, capture once, restore baseline."""
from pathlib import Path
import datetime
import json
import os
import shlex
import subprocess
import time
import traceback
import urllib.error
import sys
if __name__ != '__main__' or sys.argv[1:] != ['--execute']:
    print('Review draft: no operations. Pass --execute only for an existing owned diagnostic.');raise SystemExit(0)

raise SystemExit('DISABLED: review the independent adapter before enabling execution')
HERE=Path(__file__).resolve().parent
SSH='/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh'
BASE='wio-vllm-v41-repro169-20261028'
STAGE='wio-vllm-v41-repro169-stage-20261028'
RUN_ID=os.environ.get('WIO_BINDING_RUN_ID','')
DIAG='wio-vllm-v41-engram-'+RUN_ID
APP='/data/tiankuan/wio/projects/910c-deepseek-v4.1-flash'
ROOT='/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-169-20261028'
WORK='/work/vllm-ascend-reproduce-172-on-169-20261028'
REL='/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029/engram-diagnostic-'+RUN_ID
launch=json.loads((HERE/('launch-'+RUN_ID+'.json')).read_text())
RUN_ID=os.environ.get('WIO_BINDING_RUN_ID','')
if not (len(RUN_ID) == 32 and all((c in '0123456789abcdef' for c in RUN_ID))): raise AssertionError()
if not (launch['binding_run_id'] == RUN_ID and launch['diagnostic_name'] == DIAG): raise AssertionError()
DIAG_ID=launch['diagnostic_id']
REFERENCE=launch.get('hbm_reference')
state={'state':'WAITING_DIAGNOSTIC_HEALTH','diagnostic_id':DIAG_ID,'capture_attempted':False,'baseline_restored':False}
def save(label, **fields):
    state.update(state=label,updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),**fields)
    (HERE/('supervisor-'+RUN_ID+'.json')).write_text(json.dumps(state,indent=2)+'\n')
    print(label,flush=True)
def ssh(code,timeout=90):
    import hashlib
    if not (hashlib.sha256((HERE.parent / 'remote_guard.py').read_bytes()).hexdigest() == 'f475694ae5f660394d60d5239fa6f2ec640d64ab09d67401827f4bfee83383d7'): raise AssertionError()
    prefix='BASE='+repr(BASE)+'\nSTAGE='+repr(STAGE)+'\nDIAG='+repr(DIAG)+'\nDIAG_ID='+repr(DIAG_ID)+'\nRUN_ID='+repr(RUN_ID)+'\nREFERENCE='+repr(REFERENCE)+'\nAPP='+repr(APP)+'\nROOT='+repr(ROOT)+'\nWORK='+repr(WORK)+'\nREL='+repr(REL)+'\n'
    result=subprocess.run([SSH,shlex.join(['python3','-c',prefix+(HERE.parent/'remote_guard.py').read_text()+'\nHOLDER_CODE = "import sys\\nif sys.flags.optimize: raise RuntimeError(\'optimized Python forbidden\')\\n" + HOLDER_CODE\n'+code])],capture_output=True,text=True,timeout=timeout)
    if result.returncode:raise RuntimeError(result.stderr[-4500:]+result.stdout[-1500:])
    return result.stdout
STATUS='import sys\nif sys.flags.optimize:\n    raise RuntimeError(\'optimized Python is forbidden for diagnostic safety checks\')\nimport json, subprocess\nd = json.loads(subprocess.check_output([\'docker\', \'inspect\', DIAG_ID], text=True))[0]\nif not (d[\'Id\'] == DIAG_ID and d[\'Config\'][\'Labels\'].get(\'wio.binding_run_id\') == RUN_ID):\n    raise AssertionError()\ncheck = "import sys\\nif sys.flags.optimize:\\n    raise RuntimeError(\'optimized Python is forbidden for diagnostic safety checks\')\\nimport urllib.request, json\\nop = urllib.request.build_opener(urllib.request.ProxyHandler({}))\\nu = \'http://127.0.0.1:8172\'\\nprint(json.dumps({\'health\': op.open(u + \'/health\', timeout=5).status}))\\n"\nr = subprocess.run([\'docker\', \'exec\', DIAG_ID, \'python3\', \'-c\', check], capture_output=True, text=True, timeout=15) if d[\'State\'][\'Running\'] else None\nown = {int(x.strip()) for x in subprocess.check_output([\'docker\', \'top\', DIAG_ID, \'-eo\', \'pid\'], text=True).splitlines() if x.strip().isdigit()} if d[\'State\'][\'Running\'] else set()\npids = parse_npu_processes(subprocess.check_output([\'docker\', \'exec\', STAGE, \'npu-smi\', \'info\'], text=True))\nif pids - own:\n    raise RuntimeError(\'OTHER_NPU_PROCESSES_DURING_LOAD\')\nprint(json.dumps({\'running\': d[\'State\'][\'Running\'], \'exit_code\': d[\'State\'][\'ExitCode\'], \'started_at\': d[\'State\'][\'StartedAt\'], \'health\': json.loads(r.stdout)[\'health\'] if r and r.returncode == 0 else None}))\n'
CAPTURE="import sys\nif sys.flags.optimize:\n    raise RuntimeError('optimized Python is forbidden for diagnostic safety checks')\nimport json, re, subprocess\nfence_fd = acquire_owner_fence(ROOT + '/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029/binding-diagnostic')\nensure_owner_fence(fence_fd)\nd = json.loads(subprocess.check_output(['docker', 'inspect', DIAG_ID], text=True))[0]\nif not (d['Id'] == DIAG_ID and d['Config']['Labels'].get('wio.binding_run_id') == RUN_ID and d['State']['Running']):\n    raise AssertionError()\nown = {int(x.strip()) for x in subprocess.check_output(['docker', 'top', DIAG_ID, '-eo', 'pid'], text=True).splitlines() if x.strip().isdigit()}\nsmi = subprocess.check_output(['docker', 'exec', STAGE, 'npu-smi', 'info'], text=True)\npids = parse_npu_processes(smi)\nif not (pids and pids <= own):\n    raise AssertionError(('OTHER_NPU_PROCESSES_BEFORE_CAPTURE', sorted(pids - own)))\nensure_owner_fence(fence_fd)\nimport time\nr = subprocess.Popen(['docker', 'exec', '-e', 'WIO_ENGRAM_RUN_ID=' + RUN_ID, DIAG_ID, 'python3', APP + '/engram_client.py'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, text=True)\ndeadline = time.monotonic() + 2400\ntry:\n    while r.poll() is None:\n        if time.monotonic() > deadline:\n            raise TimeoutError('client timeout')\n        ensure_owner_fence(fence_fd)\n        current_own = {int(x.strip()) for x in subprocess.check_output(['docker', 'top', DIAG_ID, '-eo', 'pid'], text=True).splitlines() if x.strip().isdigit()}\n        current = parse_npu_processes(subprocess.check_output(['docker', 'exec', STAGE, 'npu-smi', 'info'], text=True))\n        if current - current_own:\n            raise RuntimeError('OTHER_NPU_PROCESSES_DURING_CAPTURE')\n        time.sleep(5)\n    r.communicate(timeout=10)\n    if r.returncode:\n        raise RuntimeError('no-profiler client failed; see dedicated sanitized progress')\n    print('no-profiler client completed')\nfinally:\n    if r.poll() is None:\n        r.terminate()\n        try:\n            r.wait(timeout=5)\n        except subprocess.TimeoutExpired:\n            r.kill()\n            r.wait(timeout=5)\n"
SUMMARY='import sys\nif sys.flags.optimize:\n    raise RuntimeError(\'optimized Python is forbidden for diagnostic safety checks\')\nimport subprocess\ncode = "import sys\\nif sys.flags.optimize:\\n    raise RuntimeError(\'optimized Python is forbidden for diagnostic safety checks\')\\nfrom pathlib import Path\\nimport collections, json, time\\np = Path(APP)\\nm = p / \'metadata\'\\ncounts = collections.Counter()\\nmalformed = 0\\nlog = m / \'engram-metadata.jsonl\'\\nlog_exists = log.exists()\\nlog_oversize = log_exists and log.stat().st_size > 1024 * 1024\\nif log_exists and (not log_oversize):\\n    for line in log.open():\\n        try:\\n            row = json.loads(line)\\n            counts[str(row.get(\'event\', \'<missing>\'))] += 1\\n        except (ValueError, TypeError):\\n            malformed += 1\\nprogress = json.loads((p / \'engram-client-results/progress.json\').read_text())\\ndeadline = time.monotonic() + 2\\nstatuses = []\\nstatus_invalid = 0\\nwhile True:\\n    files = list(m.glob(\'observer-*.json\'))\\n    statuses = []\\n    status_invalid = 0\\n    if len(files) > 64:\\n        status_invalid += 1\\n        files = []\\n    for f in files:\\n        try:\\n            if f.is_symlink() or f.stat().st_size > 4096:\\n                raise ValueError(\'status file invalid\')\\n            s = json.loads(f.read_text())\\n            if s.get(\'run_id\') != progress.get(\'run_id\'):\\n                raise ValueError(\'status run mismatch\')\\n            statuses.append(s)\\n        except (OSError, ValueError, TypeError):\\n            status_invalid += 1\\n    if statuses and (not status_invalid) and all((s.get(\'closed_window\') == progress.get(\'window_id\') for s in statuses)) or time.monotonic() >= deadline:\\n        break\\n    time.sleep(0.1)\\nclosed = sum((s.get(\'closed_window\') == progress.get(\'window_id\') for s in statuses))\\nprint(json.dumps({\'progress\': progress, \'event_counts\': dict(counts), \'malformed_lines\': malformed, \'log_exists\': log_exists, \'log_oversize\': log_oversize, \'observer_status_file_count\': len(files), \'valid_statuses\': len(statuses), \'invalid_statuses\': status_invalid, \'observed_closed_window\': closed, \'rank_coverage\': \'UNKNOWN\', \'graph_replay_coverage\': \'UNKNOWN\', \'window_ack_coverage\': \'UNKNOWN\', \'log_completeness\': \'UNKNOWN\'}))\\n"\nr = subprocess.run([\'docker\', \'exec\', DIAG_ID, \'python3\', \'-c\', \'APP=\' + repr(APP) + \'\\n\' + code], capture_output=True, text=True, timeout=90)\nif r.returncode:\n    raise RuntimeError(\'metadata summary unavailable\')\nprint(r.stdout)\n'
RESTORE='import sys\nif sys.flags.optimize:\n    raise RuntimeError(\'optimized Python is forbidden for diagnostic safety checks\')\nimport subprocess, json, re, datetime\nfence_fd = acquire_owner_fence(ROOT + \'/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029/binding-diagnostic\')\n\ndef cmd(args, timeout=60):\n    ensure_owner_fence(fence_fd)\n    if \'python3\' in args:\n        i = args.index(\'python3\')\n        if \'-c\' in args[i + 1:]:\n            j = args.index(\'-c\', i + 1) + 1\n            args = list(args)\n            args[j] = \'import sys\\nif sys.flags.optimize: raise RuntimeError("optimized Python forbidden")\\n\' + args[j]\n    out = subprocess.check_output(args, text=True, timeout=timeout)\n    ensure_owner_fence(fence_fd)\n    return out\nbaseline = json.loads(cmd([\'docker\', \'inspect\', BASE]))[0]\nd = json.loads(cmd([\'docker\', \'inspect\', DIAG_ID]))[0]\nif not baseline[\'Id\'] == \'b00f4d2018e3d495d876b6e396ba9579e78b47a97111b99c27d6678e31039092\':\n    raise AssertionError()\nif not (d[\'Id\'] == DIAG_ID and d[\'Config\'][\'Labels\'].get(\'wio.task\') == \'operator-ceiling169-20261029\' and (d[\'Config\'][\'Labels\'].get(\'wio.binding_run_id\') == RUN_ID)):\n    raise AssertionError()\nown_before = set()\nif d[\'State\'][\'Running\']:\n    own_before = {int(x.strip()) for x in cmd([\'docker\', \'top\', DIAG_ID, \'-eo\', \'pid\']).splitlines() if x.strip().isdigit()}\n    if not own_before:\n        raise ValueError(\'cannot identify own diagnostic processes before stop\')\n    cmd([\'docker\', \'stop\', \'--time\', \'45\', DIAG_ID], timeout=100)\nstopped = json.loads(cmd([\'docker\', \'inspect\', DIAG_ID]))[0]\nif stopped[\'Id\'] != DIAG_ID or stopped[\'State\'][\'Running\']:\n    raise ValueError(\'diagnostic stop not confirmed\')\nbaseline = json.loads(cmd([\'docker\', \'inspect\', BASE]))[0]\nif not baseline[\'Id\'] == \'b00f4d2018e3d495d876b6e396ba9579e78b47a97111b99c27d6678e31039092\':\n    raise AssertionError()\nif baseline[\'State\'][\'Running\']:\n    out = {\'state\': \'BASELINE_ALREADY_RUNNING\'}\nelse:\n    release = wait_for_empirical_release(lambda : cmd([\'docker\', \'exec\', STAGE, \'npu-smi\', \'info\']), REFERENCE)\n    if release[\'satisfies_policy\']:\n        cmd([\'docker\', \'start\', baseline[\'Id\']])\n        out = {\'state\': \'BASELINE_RESTORING\'}\n    elif release.get(\'pids\') and own_before & release[\'pids\']:\n        out = {\'state\': \'WAITING_OWNED_WORKERS_BEFORE_RESTORE\', \'pids\': sorted(own_before & release[\'pids\'])}\n    elif release.get(\'pids\'):\n        out = {\'state\': \'WAITING_OTHER_NPU_PROCESSES_BEFORE_RESTORE\', \'other_pids\': sorted(release[\'pids\'])}\n    else:\n        out = {\'state\': \'RESTORE_DEFERRED_UNATTRIBUTED_HBM\', \'reason\': release[\'reason\']}\nout[\'time\'] = datetime.datetime.now(datetime.timezone.utc).isoformat()\nsave = \'from pathlib import Path;Path(\' + repr(WORK + REL + \'/restore.json\') + \').write_text(\' + repr(json.dumps(out, indent=2) + \'\\n\') + \')\'\ncmd([\'docker\', \'exec\', STAGE, \'python3\', \'-c\', save])\nprint(json.dumps(out))\n'

error=None
try:
    save('WAITING_DIAGNOSTIC_HEALTH')
    deadline=time.monotonic()+2400
    while time.monotonic()<deadline:
        snapshot=json.loads(ssh(STATUS,timeout=30));state['last_health']=snapshot
        (HERE/('supervisor-'+RUN_ID+'.json')).write_text(json.dumps(state,indent=2)+'\n')
        if not snapshot['running']:raise RuntimeError('Diagnostic container exited: '+str(snapshot))
        if snapshot['health']==200:break
        time.sleep(20)
    else:raise TimeoutError('Diagnostic health did not become ready within 40 minutes')
    save('CAPTURE_RUNNING',capture_attempted=True)
    output=ssh(CAPTURE,timeout=2500);(HERE/('capture-'+RUN_ID+'.stdout')).write_text(output)
    save('CAPTURE_COMPLETE')
    summary=json.loads(ssh(SUMMARY,timeout=120))
    (HERE/('capture-summary-'+RUN_ID+'.json')).write_text(json.dumps(summary,indent=2)+'\n')
    save('SUMMARY_SAVED',rank_coverage='UNKNOWN',graph_replay_coverage='UNKNOWN',log_completeness='UNKNOWN')

except BaseException as exc:
    error=exc
    (HERE/('supervisor-error-'+RUN_ID+'.txt')).write_text(traceback.format_exc())
    save('CAPTURE_FAILED',error=repr(exc))
finally:
    try:
        restored=json.loads(ssh(RESTORE,timeout=400))
        (HERE/('restore-'+RUN_ID+'.json')).write_text(json.dumps(restored,indent=2)+'\n')
        save(restored['state'],capture_error=None if error is None else repr(error))
        if restored['state'] not in ('BASELINE_RESTORING','BASELINE_ALREADY_RUNNING'):raise RuntimeError('Baseline recovery deferred: '+str(restored))
    except BaseException as restore_exc:
        save('RESTORE_ERROR',restore_error=repr(restore_exc))
        raise
if error:raise error
print(json.dumps({'capture_client_complete':True,'coverage':'UNKNOWN','restore':restored}))
