"""Hold the canonical DeepSeek P/D workload lock for a staged wide campaign."""

import argparse
from datetime import datetime, timedelta, timezone
import fcntl
from hashlib import sha256
import json
import os
from pathlib import Path
import time

from plan_SG_wide_campaign_leases_g1424 import windows


def save(path, state):
    draft = path.with_suffix('.tmp')
    with draft.open('x') as stream:
        json.dump(state, stream, indent=2)
        stream.write('\n')
    draft.replace(path)


def hold(lock, control, epoch, plan, digest):
    assert lock.name == 'deepseek-v41-p3d4-workload.lock'
    assert control.name == f'workload_guard_g{epoch}'
    duration = plan['pair_guard_lease_seconds']
    with lock.open('r') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        control.mkdir(mode=0o700, exist_ok=False)
        now = datetime.now(timezone.utc)
        cutoff = time.monotonic() + duration
        state = {
            'at': now.isoformat(),
            'deadline_UTC': (now + timedelta(seconds=duration)).isoformat(),
            'epoch': epoch, 'input_tokens': plan['input_tokens'],
            'campaign_id': f'sg_pd_g{epoch}_aisbench_{plan["input_tokens"] // 1024}k',
            'pair': 'P2203/D2204', 'pid': os.getpid(), 'phase': 'holding',
            'bound_s': duration, 'preparation_manifest_sha256': digest,
        }
        state_path = control / 'state.json'
        save(state_path, state)
        stop = control / 'stop.request'
        while not stop.exists():
            remaining = cutoff - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(5, remaining))
        state['phase'] = ('released_after_stop_request' if stop.exists()
                          else 'expired_requires_external_job_audit')
        state['ended_at'] = datetime.now(timezone.utc).isoformat()
        save(state_path, state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preparation-manifest', type=Path, required=True)
    parser.add_argument('--sha256', required=True)
    parser.add_argument('--canonical-lock', type=Path, required=True)
    parser.add_argument('--control', type=Path, required=True)
    args = parser.parse_args()
    raw = args.preparation_manifest.read_bytes()
    digest = sha256(raw).hexdigest()
    assert digest == args.sha256
    manifest = json.loads(raw)
    epoch = manifest['epoch']
    assert type(epoch) is int and epoch > 1317 and manifest['derived_from_actual_epoch'] == 1317
    assert args.control.is_absolute() and args.canonical_lock.is_absolute()
    hold(args.canonical_lock.resolve(strict=True), args.control, epoch,
         windows(manifest['workload']), digest)


if __name__ == '__main__':
    main()
