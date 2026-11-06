#!/usr/bin/env python3
"""Release one finished, previously retained batch after fail-closed checks.

This does not signal processes, change allocation, or adopt an unknown job.
Use the same private operator configuration and shared lock as the runner.
"""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile


def read(path):
    return json.loads(Path(path).read_text())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', required=True)
    ap.add_argument('--run-id', required=True)
    args = ap.parse_args()
    if not re.fullmatch(r'[a-z0-9][a-z0-9-]{0,31}', args.run_id):
        ap.error('invalid run ID')
    cfg = read(args.config)
    root = Path(cfg['task_root']).resolve()
    run = (root / cfg['run_relative'] / ('bounded-cli-' + args.run_id)).resolve()
    if not run.is_relative_to(root):
        raise RuntimeError('run path escapes task root')
    allocation = sorted(cfg['allocation'], key=lambda x: x['rank'])
    if not allocation or [x['rank'] for x in allocation] != list(range(len(allocation))) or len({x['die'] for x in allocation}) != len(allocation):
        raise RuntimeError('invalid rank allocation')
    with open(cfg['shared_lock'], 'a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        a, own = read(cfg['assignment']), read(cfg['ownership'])
        for key in ('task_id', 'generation', 'executor', 'host'):
            if a[key] != own[key] or (key != 'host' and a[key] != cfg[key]):
                raise RuntimeError('assignment identity mismatch: ' + key)
        if a['host'] != os.uname().nodename or not set(x['die'] for x in allocation).issubset(a['authorized_physical_dies']):
            raise RuntimeError('host or allocation is not authorized')
        reservation, summary = read(run / 'reservation.json'), read(run / 'summary.json')
        batch = 'bounded-cli-' + args.run_id
        if reservation['id'] != batch or reservation['workers'] != len(allocation) or summary['batch_id'] != 'cli-' + args.run_id or summary['semantic_status'] != 'fail' or 'worker exit not verified; ownership retained' not in summary['semantic_errors']:
            raise RuntimeError('batch does not have the expected retained-ownership failure')
        expected = {f'{batch}-rank{x["rank"]}' for x in allocation}
        jobs = own['active_jobs']
        if len(jobs) != len(expected) or {j.get('job_id') for j in jobs} != expected:
            raise RuntimeError('ownership contains missing, extra, or unrelated jobs')
        jobs = sorted(jobs, key=lambda x: x['rank'])
        for j, item in zip(jobs, allocation):
            if j['rank'] != item['rank'] or j['physical_die'] != item['die'] or j['host'] != a['host'] or type(j.get('host_worker_pid')) is not int or j['host_worker_pid'] <= 0:
                raise RuntimeError('worker mapping mismatch')
        workers = sorted(summary['workers'], key=lambda x: x['rank'])
        if len(workers) != len(jobs) or any(w.get('rank') != j['rank'] or w.get('die') != j['physical_die'] or w.get('docker_exec_pid') != j.get('docker_exec_pid') or w.get('exit_code') is None for w, j in zip(workers, jobs)):
            raise RuntimeError('manager and registered worker records do not match')
        pids = [j['host_worker_pid'] for j in jobs]
        if len(set(pids)) != len(pids) or any((Path('/proc') / str(pid)).exists() for pid in pids):
            raise RuntimeError('worker process still exists or PID was reused')
        inspect = json.loads(subprocess.check_output(['docker', 'inspect', cfg['container']], text=True))[0]
        if inspect['Id'] != cfg['container_id'] or inspect['Image'] != cfg['image_id'] or not inspect['State']['Running']:
            raise RuntimeError('container identity or state mismatch')
        top = subprocess.check_output(['docker', 'top', cfg['container'], '-eo', 'pid,ppid,args'], text=True).strip().splitlines()
        if len(top) != 2 or 'sleep infinity' not in top[1]:
            raise RuntimeError('task container is not idle')
        smi = subprocess.check_output(['npu-smi', 'info'], text=True, timeout=20)
        if any(f'No running processes found in NPU {card}' not in smi for card in {x['card'] for x in allocation}):
            raise RuntimeError('allocated card process listing not empty or uncertain')
        if any(re.search(r'(?<!\d)' + str(pid) + r'(?!\d)', line) for pid in pids for line in smi.splitlines() if '|' in line):
            raise RuntimeError('worker PID remains in device output')
        ranks = [read(run / f'rank{x["rank"]}' / 'result.json') for x in allocation]
        if any(r.get('rank') != x['rank'] or r.get('die') != x['die'] or r.get('status') not in ('passed', 'failed') or r.get('destroyed') is not True for r, x in zip(ranks, allocation)):
            raise RuntimeError('worker results missing, mismatched, or still running')
        own['active_jobs'] = []
        own.setdefault('adopted_completion_history', []).append({'run_id': args.run_id, 'manager_status': 'fail', 'worker_statuses': [r.get('status') for r in ranks], 'verified_worker_exit_at': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'pids': pids, 'reason': 'retained claims released after verified worker, NPU and container exit'})
        path = Path(cfg['ownership'])
        fd, tmp = tempfile.mkstemp(prefix='.owned-recovery-', dir=path.parent)
        try:
            original = path.stat()
            created = os.fstat(fd)
            if (created.st_uid, created.st_gid) != (original.st_uid, original.st_gid):
                raise RuntimeError('cannot preserve registry owner/group')
            os.fchmod(fd, original.st_mode & 0o7777)
            with os.fdopen(fd, 'w') as f:
                json.dump(own, f, indent=2)
                f.write('\n')
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
        print(json.dumps({'run_id': args.run_id, 'claims_released': True, 'pids': pids, 'worker_statuses': [r.get('status') for r in ranks], 'manager_status': summary['semantic_status']}))


if __name__ == '__main__':
    main()
