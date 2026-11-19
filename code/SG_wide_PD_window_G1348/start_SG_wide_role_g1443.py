"""Start one staged wide role after a verified predecessor release and pair guard."""

import argparse
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

from check_SG_wide_role_resources_g1442 import PORTS, exclusive_flocks, inspect, lock_reader
from observe_SG_wide_role_g1439 import IMAGE, cmdline, docker, prove_guard, read
from plan_SG_wide_campaign_leases_g1424 import windows
from wide_initial_gate_context_g1434 import PLAN_BY_WIDTH, load_plan


PREDECESSOR = {65536: 1317, 131072: 1347, 262144: 14331}
STARTING = {'waiting_verified_weights', 'claimed', 'model_starting'}


def release_proof(root, coord, width, role, resources):
    epoch = PREDECESSOR[width]
    old = root / f'sg_pd_g{epoch}'
    state = read(old / 'state.json')
    prior = read(old / 'actual_launch_handle.json')
    pid = prior['pid']
    assert prior['epoch'] == state['generation'] == epoch
    assert prior['role'] == state['role'] == role
    assert state['pid'] == pid and state['task_id'] == root.name and state['host'] == socket.gethostname()
    if not resources['released_in_snapshot']:
        return {'predecessor_epoch': epoch, 'predecessor_controller_PID': pid,
                'predecessor_phase': state['phase'], 'ready_after_predecessor_release': False,
                'resource_snapshot': resources}

    assert state['phase'] in ('stopped_released', 'failed_released')
    assert state['cleanup']['all8_process_lists_clear'] is True
    assert state['image_id'] == prior['image'] == IMAGE
    controller = str(old / 'node_controller.py').encode()
    try:
        assert controller not in cmdline(pid), 'Previous controller has not exited'
    except FileNotFoundError:
        pass
    host = state['host']
    for rank in resources['devices']:
        owner = read(coord / f"{host}-npu{rank['rank']}.owner.json")
        assert rank['released'] and rank['owner_generation'] == owner['generation']
        assert owner['state'] == 'released' and owner['pid'] == rank['owner_PID'] == pid
        assert owner['task_id'] == root.name and owner['run_directory'] == str(old)

    container = json.loads(docker('inspect', state['container']))[0]
    labels = container['Config']['Labels']
    assert container['Image'] == IMAGE and labels['codex.task'] == root.name
    assert labels['codex.controller_pid'] == str(pid) and labels['codex.run'] == old.name
    assert container['State']['Running'] is False and container['State']['Pid'] == 0
    device_info = subprocess.check_output(['npu-smi', 'info'], text=True, timeout=15)
    assert all(f'No running processes found in NPU {rank}' in device_info for rank in range(8))
    return {'predecessor_epoch': epoch, 'predecessor_controller_PID': pid, 'predecessor_phase': state['phase'],
            'predecessor_container_ID': container['Id'], 'current_NPU_process_lists_clear': True,
            'ready_after_predecessor_release': True, 'resource_snapshot': resources}


def guard_for_start(root, coord, role, plan, plan_sha, prefill_receipt, receipt_sha):
    epoch = plan['epoch']
    width = plan['workload']['width_tokens']
    if role == 'p':
        state = read(root / f'workload_guard_g{epoch}' / 'state.json')
        guard = prove_guard(root, coord, plan, plan_sha, {'guard_PID': state['pid']})
        lock = coord / 'deepseek-v41-p3d4-workload.lock'
        assert lock_reader(lock, exclusive_flocks()) == guard['pid']
    else:
        raw = prefill_receipt.read_bytes()
        assert sha256(raw).hexdigest() == receipt_sha
        prefill = json.loads(raw)
        assert prefill['role'] == 'p' and prefill['epoch'] == epoch
        assert prefill['plan_SHA256'] == plan_sha and prefill['image'] == IMAGE
        assert prefill['source_manifest_SHA256'] == plan['roles']['p']['deployment_manifest_sha256']
        age = datetime.now(timezone.utc) - datetime.fromisoformat(prefill['at'])
        assert timedelta(seconds=-30) <= age < timedelta(seconds=180)
        guard = prefill['pair_guard']
        assert guard['pid'] == prefill['guard_PID']
        assert prefill['predecessor_release']['ready_after_predecessor_release'] is True
        assert prefill['native_controller_phase'] in STARTING and prefill['pid'] > 1

    bounds = windows(plan['workload'])
    assert guard['phase'] == 'holding' and (guard['epoch'], guard['input_tokens']) == (epoch, width)
    assert guard['campaign_id'] == f'sg_pd_g{epoch}_aisbench_{width // 1024}k'
    assert guard['pair'] == 'P2203/D2204' and guard['preparation_manifest_sha256'] == plan_sha
    assert guard['bound_s'] == bounds['pair_guard_lease_seconds'] and type(guard['pid']) is int and guard['pid'] > 1
    deadline = datetime.fromisoformat(guard['deadline_UTC'])
    assert deadline == datetime.fromisoformat(guard['at']) + timedelta(seconds=guard['bound_s'])
    remaining = deadline - datetime.now(timezone.utc)
    assert remaining > timedelta(seconds=bounds['minimum_same_client_lease_from_first_warmup_seconds'])
    return guard



def write_receipt(path, handle):
    pending = path.with_suffix('.pending')
    with pending.open('x') as stream:
        json.dump(handle, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    pending.rename(path)


def start(run, role, plan, plan_sha, proof, guard):
    argv = [sys.executable, '-u', '-B', str(run / 'node_controller.py')]
    started = datetime.now(timezone.utc)
    with (run / 'controller.log').open('x') as log:
        child = subprocess.Popen(argv, cwd=run, stdin=subprocess.DEVNULL, stdout=log,
                                 stderr=subprocess.STDOUT, start_new_session=True)
    handle = {
        'epoch': plan['epoch'], 'role': role, 'pid': child.pid, 'at': started.isoformat(), 'argv': argv,
        'guard_PID': guard['pid'], 'pair_guard': guard, 'image': IMAGE, 'plan_SHA256': plan_sha,
        'source_manifest_SHA256': plan['roles'][role]['deployment_manifest_sha256'],
        'predecessor_release': proof,
    }
    write_receipt(run / 'actual_launch_handle.json', handle)
    end = time.monotonic() + 10
    while time.monotonic() < end and child.poll() is None:
        state = run / 'state.json'
        if state.is_file():
            observed = read(state)
            assert (observed['pid'], observed['generation'], observed['role']) == (child.pid, plan['epoch'], role)
            assert observed['phase'] in STARTING, observed['phase']
            assert str(run / 'node_controller.py').encode() in cmdline(child.pid)
            handle['native_controller_phase'] = observed['phase']
            write_receipt(run / 'verified_launch_receipt.json', handle)
            return handle
        time.sleep(0.1)
    raise RuntimeError('Native controller launch not verified. Preserve and inspect the launch handle and log.')


def main():
    if not __debug__:
        raise RuntimeError('Launch requires Python assertions to be enabled')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task-root', type=Path, required=True)
    parser.add_argument('--coord', type=Path, required=True)
    parser.add_argument('--width', type=int, choices=PLAN_BY_WIDTH, required=True)
    parser.add_argument('--role', choices=PORTS, required=True)
    parser.add_argument('--launch', action='store_true')
    parser.add_argument('--prefill-receipt', type=Path)
    parser.add_argument('--prefill-receipt-sha256')
    args = parser.parse_args()
    root, coord = args.task_root.resolve(strict=True), args.coord.resolve(strict=True)
    resources = inspect(root, coord, args.width, args.role)
    proof = release_proof(root, coord, args.width, args.role, resources)
    if not proof['ready_after_predecessor_release'] or not args.launch:
        print(json.dumps(proof, sort_keys=True))
        return 0 if proof['ready_after_predecessor_release'] else 3
    if args.role == 'd' and not (args.prefill_receipt and args.prefill_receipt_sha256):
        parser.error('--launch on D requires the fresh P launch receipt and its SHA256')
    run = root / f"sg_pd_g{resources['epoch']}"
    plan, plan_sha = load_plan(run / 'wide_workload_plan.json', args.width)
    guard = guard_for_start(root, coord, args.role, plan, plan_sha, args.prefill_receipt,
                            args.prefill_receipt_sha256)
    print(json.dumps(start(run, args.role, plan, plan_sha, proof, guard), sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
