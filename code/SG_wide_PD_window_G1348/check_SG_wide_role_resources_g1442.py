"""Inspect canonical NPU and port state before a staged wide role is started."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket

from observe_SG_wide_role_g1439 import fds, read
from prepare_SG_wide_PD_window_g1347 import read_pinned_files
from wide_initial_gate_context_g1434 import PLAN_BY_WIDTH, load_plan


PORTS = {'p': (19093, 18998, 18678), 'd': (19094,)}


def exclusive_flocks():
    holders = {}
    for line in Path('/proc/locks').read_text().splitlines():
        _, kind, _, mode, pid, address, *_ = line.split()
        if (kind, mode) == ('FLOCK', 'WRITE'):
            major, minor, inode = address.split(':')
            holders[(int(major, 16), int(minor, 16), int(inode))] = int(pid)
    return holders


def lock_reader(path, holders):
    try:
        stat = path.stat()
    except FileNotFoundError:
        return None
    return holders.get((os.major(stat.st_dev), os.minor(stat.st_dev), stat.st_ino))


def owns_descriptor(path, pid):
    try:
        return path in fds(pid)
    except (FileNotFoundError, PermissionError):
        return False


def device_status(coord, host, rank, holders):
    base = coord / f'{host}-npu{rank}'
    registry = base.with_suffix('.owner.json')
    lock = base.with_suffix('.lock')
    holder = lock_reader(lock, holders)
    owner = read(registry) if registry.exists() else {}
    pid = owner.get('pid')
    state = owner.get('state')
    generation = owner.get('generation')
    trusted = (owner.get('host') == host and type(pid) is int and pid > 1
               and type(generation) is int and generation > 0)
    released = trusted and state == 'released' and holder is None and lock.is_file()
    held = trusted and pid == holder and owns_descriptor(lock, pid)
    return {'rank': rank, 'owner_state': state, 'owner_PID': pid, 'owner_generation': generation,
            'flock_PID': holder, 'owner_and_exclusive_flock_agree': state == 'claimed' and held,
            'released': released}


def inspect(root, coord, width, role):
    epoch, _ = PLAN_BY_WIDTH[width]
    run = root / f'sg_pd_g{epoch}'
    plan, plan_sha = load_plan(run / 'wide_workload_plan.json', width)
    source_sha = plan['roles'][role]['deployment_manifest_sha256']
    verified = read_pinned_files(run, source_sha)
    assert 'node_controller.py' in verified
    previous_artifacts = [name for name in ('actual_launch_handle.json', 'state.json', 'controller.log', 'stop.request')
                          if (run / name).exists()]
    holders = exclusive_flocks()
    host = socket.gethostname()
    devices = [device_status(coord, host, rank, holders) for rank in range(8)]
    ports = {port: lock_reader(coord / f'{host}-port{port}.lock', holders) for port in PORTS[role]}
    available = (all(item['released'] for item in devices) and all(pid is None for pid in ports.values())
                 and not previous_artifacts)
    return {
        'at_UTC': datetime.now(timezone.utc).isoformat(), 'epoch': epoch, 'role': role,
        'width_tokens': width, 'plan_SHA256': plan_sha, 'source_manifest_SHA256': source_sha,
        'verified_source_file_count': len(verified), 'prior_attempt_artifacts': previous_artifacts,
        'devices': devices, 'port_flock_PID': ports, 'released_in_snapshot': available,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task-root', type=Path, required=True)
    parser.add_argument('--coord', type=Path, required=True)
    parser.add_argument('--width', type=int, choices=PLAN_BY_WIDTH, required=True)
    parser.add_argument('--role', choices=PORTS, required=True)
    args = parser.parse_args()
    result = inspect(args.task_root.resolve(strict=True), args.coord.resolve(strict=True), args.width, args.role)
    print(json.dumps(result, sort_keys=True))
    return 0 if result['released_in_snapshot'] else 3


if __name__ == '__main__':
    raise SystemExit(main())
