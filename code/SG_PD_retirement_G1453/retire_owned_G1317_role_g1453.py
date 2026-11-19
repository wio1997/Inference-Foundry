"""Verify or gracefully retire one exact G1317 role after preserved 32K evidence."""

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import fcntl
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import select
import socket
import urllib.request

from check_SG_wide_role_resources_g1442 import PORTS, device_status, exclusive_flocks, lock_reader
from observe_SG_wide_role_g1439 import IMAGE, QUEUES, cmdline, docker, read


ROLE_PIDS = {'p': 2302205, 'd': 2967504}
ROUTER_PID = 2308473
GUARD_PID = 2335591
GUARD_SOURCE_SHA = '1f1f1d76282a71a5b300fc622f9f858332959bfcbe397fc4997d0e2549e34f39'
CAMPAIGN = 'sg_pd_g1317_aisbench_32k_g1324'


def now():
    return datetime.now(timezone.utc)


def record(path, expected):
    raw = path.read_bytes()
    assert sha256(raw).hexdigest() == expected
    return json.loads(raw)


def accepted_closure(path, digest):
    report = record(path, digest)
    assert report['epoch'] == 1317 and report['prior_phase_evidence_complete'] is True
    assert report['quality_all_1319_comparable'] is True and report['quality_cases_verified'] == 1319
    traces = report['trace_inventory']
    assert traces['paired_8_plus_8_raw_files_present'] is True
    assert report['capture_identity_SHA256'] == traces['capture_identity_sha256']
    for name in ('formal_audit_SHA256', 'quality_terminal_SHA256', 'capture_identity_SHA256'):
        assert re.fullmatch(r'[0-9a-f]{64}', report[name])
    return report


def request(port, path):
    with urllib.request.urlopen(f'http://127.0.0.1:{port}/{path}', timeout=6) as out:
        assert out.status == 200
        return out.read()


def queue_readings(role):
    metrics = request(PORTS[role][0], 'metrics').decode()
    keys = ('num_running_reqs', 'num_queue_reqs', 'num_grammar_queue_reqs', *QUEUES[role])
    if role == 'd':
        keys += ('num_decode_host_receive_queue_reqs',)
    return {key: [float(v) for v in re.findall(
        r'^sglang:' + key + r'(?:\{[^\n]*\})?\s+([\d.eE+-]+)', metrics, re.M)] for key in keys}


def local_container(run, role, state):
    pid = ROLE_PIDS[role]
    assert state['task_id'] == run.parent.name and state['host'] == socket.gethostname()
    assert state['generation'] == 1317 and state['role'] == role and state['pid'] == pid
    handle = read(run / 'actual_launch_handle.json')
    assert handle['epoch'] == 1317 and handle['pid'] == pid and handle['role'] == role and handle['image'] == IMAGE
    info = json.loads(docker('inspect', state['container']))[0]
    labels = info['Config']['Labels']
    assert info['Image'] == IMAGE and labels['codex.task'] == run.parent.name
    assert labels['codex.controller_pid'] == str(pid) and labels['codex.run'] == run.name
    return info


def pair_guard(run, coord):
    source = run.parent / 'workload_guard_g1326.py'
    assert sha256(source.read_bytes()).hexdigest() == GUARD_SOURCE_SHA
    guard = read(run.parent / 'workload_guard_g1326/state.json')
    assert guard['phase'] == 'holding' and guard['pid'] == GUARD_PID and guard['campaign_id'] == CAMPAIGN
    assert guard['router_PID'] == ROUTER_PID and str(source).encode() in cmdline(GUARD_PID)
    router = read(run / 'router_handle.json')
    assert router['router_PID'] == ROUTER_PID
    assert router['P_controller_PID'] == ROLE_PIDS['p'] and router['D_controller_PID'] == ROLE_PIDS['d']
    assert guard['prefill_container_ID'] == router['P_container_ID']
    assert guard['decode_container_ID'] == router['D_container_ID']
    assert lock_reader(coord / 'deepseek-v41-p3d4-workload.lock', exclusive_flocks()) == GUARD_PID
    return {'prefill_container_ID': router['P_container_ID'], 'decode_container_ID': router['D_container_ID'],
            'guard_PID': GUARD_PID, 'guard_deadline_UTC': guard['deadline_UTC']}


def p_active_pair(run, coord, cid):
    pair = pair_guard(run, coord)
    assert pair['prefill_container_ID'] == cid
    router = read(run / 'router_handle.json')
    assert [item.encode() for item in router['argv']] == [item for item in cmdline(ROUTER_PID) if item]
    workers = json.loads(request(18090, 'workers'))['workers']
    assert len(workers) == 2 and {worker['worker_type'] for worker in workers} == {'prefill', 'decode'}
    return {**pair, 'router_idle': all(worker['is_healthy'] and worker['load'] == 0 for worker in workers)}


def current_receipt(run, coord, role, closure_sha):
    state = read(run / 'state.json')
    info = local_container(run, role, state)
    assert info['State']['Running'] and info['Id'] == state['container_id']
    pid = ROLE_PIDS[role]
    assert str(run / 'node_controller.py').encode() in cmdline(pid)
    holders = exclusive_flocks()
    devices = [device_status(coord, state['host'], rank, holders) for rank in range(8)]
    assert all(dev['owner_and_exclusive_flock_agree'] and dev['owner_PID'] == pid for dev in devices)
    counts = queue_readings(role)
    empty = all(values and all(v == 0 for v in values) for values in counts.values())
    report = {'at_UTC': now().isoformat(), 'epoch': 1317, 'role': role, 'controller_PID': pid,
              'container_ID': info['Id'], 'queues': counts, 'all_queues_empty': empty,
              'all8_owned_flocks_verified': True, 'closure_SHA256': closure_sha,
              'retirement_request_present': (run / 'stop.request').exists()}
    if role == 'p':
        report['active_pair'] = p_active_pair(run, coord, info['Id'])
    return report


def assert_both_idle(p, d, closure_sha):
    for role, item in (('p', p), ('d', d)):
        age = (now() - datetime.fromisoformat(item['at_UTC'])).total_seconds()
        assert -30 <= age < 120
        assert item['role'] == role and item['epoch'] == 1317 and item['controller_PID'] == ROLE_PIDS[role]
        assert item['closure_SHA256'] == closure_sha
        assert item['all_queues_empty'] and item['all8_owned_flocks_verified']
        assert not item['retirement_request_present']
    guard = p['active_pair']
    assert guard['guard_PID'] == GUARD_PID and guard['router_idle']
    assert guard['prefill_container_ID'] == p['container_ID'] and guard['decode_container_ID'] == d['container_ID']
    assert (datetime.fromisoformat(guard['guard_deadline_UTC']) - now()).total_seconds() > 240


def exited(pid):
    try:
        fd = os.pidfd_open(pid)
    except ProcessLookupError:
        return True
    with os.fdopen(fd, 'rb') as event:
        polls = select.poll()
        polls.register(event, select.POLLIN)
        return bool(polls.poll(0))


def terminal_receipt(run, coord, role):
    state = read(run / 'state.json')
    info = local_container(run, role, state)
    assert state['phase'] in ('stopped_released', 'failed_released')
    assert state['cleanup']['all8_process_lists_clear'] and exited(state['pid'])
    assert not info['State']['Running'] and info['State']['Pid'] == 0
    host = state['host']
    holders = exclusive_flocks()
    for rank in range(8):
        dev = device_status(coord, host, rank, holders)
        owner = read(coord / f'{host}-npu{rank}.owner.json')
        assert dev['released'] and owner['pid'] == state['pid'] and owner['run_directory'] == str(run)
    ports = (*PORTS[role], 18090) if role == 'p' else PORTS[role]
    assert all(lock_reader(coord / f'{host}-port{port}.lock', holders) is None for port in ports)
    if role == 'p':
        assert exited(ROUTER_PID)
    stop = read(run / 'stop.request')
    assert stop['owned_controller_PID'] == state['pid'] and stop['container_ID'] == info['Id']
    result = {'at_UTC': now().isoformat(), 'epoch': 1317, 'role': role, 'controller_PID': state['pid'],
              'container_ID': info['Id'], 'phase': state['phase'], 'all8_owned_resources_released': True,
              'container_stopped': True, 'router_released_with_prefill_container': True if role == 'p' else None,
              'closure_SHA256': stop['closure_SHA256']}
    if role == 'p' and read(run.parent / 'workload_guard_g1326/state.json')['phase'] == 'holding':
        result['active_pair'] = pair_guard(run, coord)
    return result


def checked_terminal(item, role, closure_sha):
    age = (now() - datetime.fromisoformat(item['at_UTC'])).total_seconds()
    assert -30 <= age < 120
    assert item['epoch'] == 1317 and item['role'] == role and item['controller_PID'] == ROLE_PIDS[role]
    assert item['closure_SHA256'] == closure_sha and item['all8_owned_resources_released'] and item['container_stopped']
    assert item['phase'] in ('failed_released', 'stopped_released')
    if role == 'p':
        assert item['router_released_with_prefill_container']


def assert_decode_after_prefill(p, d, closure_sha):
    checked_terminal(p, 'p', closure_sha)
    guard = p['active_pair']
    assert guard['guard_PID'] == GUARD_PID and guard['prefill_container_ID'] == p['container_ID']
    assert guard['decode_container_ID'] == d['container_ID'] and d['all_queues_empty']
    assert (datetime.fromisoformat(guard['guard_deadline_UTC']) - now()).total_seconds() > 120


def request_stop(run, coord, role, closure_sha, peer):
    with ExitStack() as locks:
        if role == 'd':
            path = run.parent / 'client_ds41_g1122/sglang_runner.lock'
            lock = locks.enter_context(path.open('r'))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        local = current_receipt(run, coord, role, closure_sha)
        if role == 'p':
            assert_both_idle(local, peer, closure_sha)
        else:
            assert_decode_after_prefill(peer, local, closure_sha)
        pid = local['controller_PID']
        proc = locks.enter_context(os.fdopen(os.pidfd_open(pid), 'rb'))
        event = select.poll()
        event.register(proc, select.POLLIN)
        assert not event.poll(0) and str(run / 'node_controller.py').encode() in cmdline(pid)
        with (run / 'stop.request').open('x') as stream:
            json.dump({'at_UTC': now().isoformat(), 'owned_controller_PID': pid,
                       'container_ID': local['container_ID'], 'closure_SHA256': closure_sha}, stream)
            stream.write('\n')
        if not event.poll(90000):
            return {**local, 'status': 'waiting_on_existing_stop_request'}
    return terminal_receipt(run, coord, role)


def release_guard(run, coord, role, closure_sha, peer):
    assert role == 'p'
    checked_terminal(peer, 'd', closure_sha)
    local = terminal_receipt(run, coord, 'p')
    checked_terminal(local, 'p', closure_sha)
    control = run.parent / 'workload_guard_g1326'
    if 'active_pair' in local:
        assert local['active_pair']['decode_container_ID'] == peer['container_ID']
        with os.fdopen(os.pidfd_open(GUARD_PID), 'rb') as guard:
            event = select.poll()
            event.register(guard, select.POLLIN)
            assert not event.poll(0)
            pair = pair_guard(run, coord)
            assert pair['prefill_container_ID'] == local['container_ID']
            stop = control / 'stop.request'
            if not stop.exists():
                with stop.open('x') as stream:
                    json.dump({'closure_SHA256': closure_sha, 'guard_PID': GUARD_PID}, stream)
                    stream.write('\n')
            if not event.poll(25000):
                return {'status': 'waiting_on_existing_guard_stop_request', 'guard_PID': GUARD_PID}
    request_record = read(control / 'stop.request')
    guard_end = read(control / 'state.json')
    assert request_record == {'closure_SHA256': closure_sha, 'guard_PID': GUARD_PID}
    assert guard_end['phase'] == 'released_after_stop_request' and guard_end['pid'] == GUARD_PID
    assert exited(GUARD_PID)
    assert lock_reader(coord / 'deepseek-v41-p3d4-workload.lock', exclusive_flocks()) is None
    return {'status': 'pair_guard_released_after_both_roles', 'guard_PID': GUARD_PID,
            'closure_SHA256': closure_sha, 'P_controller_PID': local['controller_PID'],
            'D_controller_PID': peer['controller_PID']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task-root', type=Path, required=True)
    parser.add_argument('--coord', type=Path, required=True)
    parser.add_argument('--role', choices=ROLE_PIDS, required=True)
    parser.add_argument('--closure', type=Path)
    parser.add_argument('--closure-sha256')
    parser.add_argument('--peer', type=Path)
    parser.add_argument('--peer-sha256')
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--request-stop', action='store_true')
    action.add_argument('--release-pair-guard', action='store_true')
    args = parser.parse_args()
    if not __debug__:
        raise RuntimeError('Retirement requires validation assertions')
    run = args.task_root.resolve(strict=True) / 'sg_pd_g1317'
    coord = args.coord.resolve(strict=True)
    if args.closure:
        accepted_closure(args.closure, args.closure_sha256)
    if args.request_stop or args.release_pair_guard:
        if not all((args.closure, args.closure_sha256, args.peer, args.peer_sha256)):
            parser.error('A stop request requires the accepted closure and fresh peer receipt, with digests')
        peer = record(args.peer, args.peer_sha256)
        if args.release_pair_guard:
            result = release_guard(run, coord, args.role, args.closure_sha256, peer)
        elif (run / 'stop.request').exists():
            if read(run / 'state.json')['phase'] in ('stopped_released', 'failed_released'):
                result = terminal_receipt(run, coord, args.role)
                assert result['closure_SHA256'] == args.closure_sha256
            else:
                result = {'status': 'waiting_on_existing_stop_request', 'role': args.role}
        else:
            result = request_stop(run, coord, args.role, args.closure_sha256, peer)
    elif (run / 'stop.request').exists() and read(run / 'state.json')['phase'] in ('stopped_released', 'failed_released'):
        result = terminal_receipt(run, coord, args.role)
    else:
        result = current_receipt(run, coord, args.role, args.closure_sha256 if args.closure else None)
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
