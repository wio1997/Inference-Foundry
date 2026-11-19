"""Run the fixed A2 C2 short correctness probe on an exclusively leased card 0."""

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import fcntl
from hashlib import sha256
import json
import os
from pathlib import Path
import signal
import socket
import subprocess

from check_SG_wide_role_resources_g1442 import device_status, exclusive_flocks, lock_reader


IMAGE = 'sha256:7da963869ee54984dfeea2445befdf181710ef846448466db2361247759904ec'
SOURCE = {
    'c2_lightning_indexer_geometry_g1425.py': '538eec6347fe17ffeeb3f4bbed6876615da930799112c0a456288468a788fcff',
    'causal_score.py': '208d35ea7b783cb2a3fe831ce12bfbdcdd4084079c296bf8c00db352b58f9393',
    'probe.py': '3aaf5d3ca10d25970904ccb5b7d8e4cbc1e60228d63bde876fafa3c94d7b0c35',
    'probe_c2_lightning_indexer_g1426.py': '1b01efdf8dce5201ca2bd7dc5a2b4f5b89df3bc723152993b6c95fe274a57fe1',
}
DEVICE_NODES = ('/dev/davinci0', '/dev/davinci_manager', '/dev/devmm_svm', '/dev/hisi_hdc')
WORKLOAD_LOCK = 'deepseek-v41-p3d4-workload.lock'
PROBE_CPU = 179
CONTAINER = 'ds41-c2-short-g1450'


def read(path):
    return json.loads(path.read_bytes())


def save(path, body):
    candidate = path.with_suffix('.pending')
    with candidate.open('x') as stream:
        json.dump(body, stream, indent=2)
        stream.write('\n')
    candidate.replace(path)


def command(*args, timeout=12):
    return subprocess.check_output(args, text=True, timeout=timeout)


def same_stage(task, stage):
    assert stage.is_relative_to(task)
    for name, digest in SOURCE.items():
        path = stage / name
        assert not path.is_symlink() and sha256(path.read_bytes()).hexdigest() == digest, name


def inspect(coord, host):
    holders = exclusive_flocks()
    card = device_status(coord, host, 0, holders)
    pair = coord / WORKLOAD_LOCK
    guard_pid = lock_reader(pair, holders)
    cpu = coord / f'{host}-cpu176-179.lock'
    cpu_pid = lock_reader(cpu, holders)
    ready = (pair.is_file() and guard_pid is None and card['released'] and cpu_pid is None
             and cpu.is_file() and read(cpu.with_suffix('.owner.json'))['state'] == 'released')
    return {'physical_device': 0, 'card': card, 'pair_flock_PID': guard_pid, 'CPU_flock_PID': cpu_pid,
            'available_in_snapshot': ready}


def assert_unoccupied():
    assert 'No running processes found in NPU 0' in command('npu-smi', 'info', timeout=15)


def own_slot(coord, host, previous, run, task):
    card_lock = coord / f'{host}-npu0.lock'
    owner = card_lock.with_suffix('.owner.json')
    cpu_lock = coord / f'{host}-cpu176-179.lock'
    with ExitStack() as leases:
        for path in (coord / WORKLOAD_LOCK, card_lock, cpu_lock):
            descriptor = leases.enter_context(path.open('r'))
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert read(cpu_lock.with_suffix('.owner.json'))['state'] == 'released'
        os.sched_setaffinity(0, {PROBE_CPU})
        prior = read(owner)
        assert prior['state'] == 'released' and prior['host'] == host and prior['task_id'] == task.name
        previous_state = read(previous / 'state.json')
        assert previous_state['role'] == 'p' and previous_state['phase'] in ('stopped_released', 'failed_released')
        assert previous_state['cleanup']['all8_process_lists_clear'] is True
        assert prior['run_directory'] == str(previous) and prior['pid'] == previous_state['pid']
        assert_unoccupied()
        run.mkdir(mode=0o700)
        claimed = {**prior, 'state': 'claimed', 'pid': os.getpid(), 'generation': prior['generation'] + 1,
                   'run_directory': str(run), 'purpose': 'C2 short correctness component probe',
                   'predecessor_owner': {name: prior[name] for name in ('pid', 'generation', 'run_directory')}}
        save(owner, claimed)
        try:
            result = run_container(task, run, run.parent)
        finally:
            # Preserve the claim when container exit/removal or NPU quiescence is unverified.
            cleanup = run / 'container_cleanup.json'
            if cleanup.is_file():
                outcome = read(cleanup)
                assert outcome['stopped_and_removed'] and outcome['controller_PID'] == os.getpid()
                assert_unoccupied()
                assert read(owner) == claimed
                save(owner, {**claimed, 'state': 'released',
                             'container_cleanup_receipt_sha256': sha256(cleanup.read_bytes()).hexdigest()})
        return result


def run_container(task, run, stage):
    for device in DEVICE_NODES:
        assert Path(device).is_char_device(), device
    log_path = run / 'container.log'
    result_path = run / 'short_probe.jsonl'
    labels = {'codex.task': task.name, 'codex.run': run.name, 'codex.controller_pid': str(os.getpid())}
    script = ('if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then '
              '. /usr/local/Ascend/ascend-toolkit/set_env.sh; fi; '
              'exec python3 -B /probe/probe_c2_lightning_indexer_g1426.py '
              '--device 0 --suite short --repeat 1 --output /out/short_probe.jsonl')
    args = ['docker', 'create', '--name', CONTAINER, '--runtime', 'runc', '--network', 'none',
            '--read-only', '--no-healthcheck', '--memory', '16g', '--cpus', '1',
            '--cpuset-cpus', str(PROBE_CPU), '--workdir', '/out',
            '--env', 'ASCEND_RT_VISIBLE_DEVICES=0', '--env', 'HOME=/out', '--env', 'TMPDIR=/out',
            '--env', 'OMP_NUM_THREADS=1', '--mount', f'type=bind,source={stage},target=/probe,readonly',
            '--mount', f'type=bind,source={run},target=/out',
            '--mount', 'type=bind,source=/usr/local/Ascend/driver,target=/usr/local/Ascend/driver,readonly',
            '--entrypoint', '/bin/bash']
    for key, value in labels.items():
        args.extend(('--label', f'{key}={value}'))
    for path in DEVICE_NODES:
        args.extend(('--device', path))
    # Pin the same vendor driver metadata as the model container when supplied on the host.
    if Path('/etc/ascend_install.info').is_file():
        args.extend(('--mount', 'type=bind,source=/etc/ascend_install.info,target=/etc/ascend_install.info,readonly'))
    args.extend((IMAGE, '-lc', script))
    cid = None
    try:
        cid = command(*args, timeout=20).strip()
        info = json.loads(command('docker', 'inspect', cid))[0]
        config = info['HostConfig']
        assert info['Image'] == IMAGE and all(info['Config']['Labels'][key] == val for key, val in labels.items())
        assert config['NetworkMode'] == 'none' and config['ReadonlyRootfs'] and not config['Privileged']
        assert {node['PathOnHost'] for node in config['Devices']} == set(DEVICE_NODES)
        assert config['CpusetCpus'] == str(PROBE_CPU) and config['Runtime'] == 'runc'
        save(run / 'launch_handle.json', {'container_ID': cid, 'image': IMAGE, 'physical_device': 0,
                                         'controller_PID': os.getpid(), 'source_sha256': SOURCE})
        with log_path.open('x') as log:
            result = subprocess.run(['docker', 'start', '--attach', cid], cwd=run, stdout=log,
                                    stderr=subprocess.STDOUT, timeout=150)
        lines = [json.loads(line) for line in result_path.read_text().splitlines()] if result_path.exists() else []
        if lines:
            assert lines[0]['probe_sha256'] == SOURCE['probe_c2_lightning_indexer_g1426.py']
        accepted = (result.returncode == 0 and len(lines) == 7 and lines[-1] == {'stage': 'complete', 'all_exact': True}
                    and all(row['passed_exact_component_gate'] for row in lines[1:-1]))
        return {'at_UTC': datetime.now(timezone.utc).isoformat(), 'container_ID': cid, 'image': IMAGE,
                'physical_device': 0, 'standalone_short_probe_only': True,
                'exit_code': result.returncode, 'all_short_cases_exact': accepted,
                'probe_records': len(lines)}
    finally:
        if cid:
            item = json.loads(command('docker', 'inspect', cid))[0]
            assert all(item['Config']['Labels'][key] == val for key, val in labels.items())
            if item['State']['Running']:
                command('docker', 'stop', '--time', '5', cid, timeout=16)
            end = json.loads(command('docker', 'inspect', cid))[0]['State']
            assert not end['Running'] and end['Pid'] == 0
            command('docker', 'rm', cid)
            save(run / 'container_cleanup.json', {'container_ID': cid, 'controller_PID': os.getpid(),
                                                  'stopped_and_removed': True})


def interrupted(signum, _frame):
    raise RuntimeError(f'Probe controller interrupted by signal {signum}')


def main():
    if not __debug__:
        raise RuntimeError('Probe requires assertions enabled')
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ('task-root', 'coord', 'stage', 'previous-prefill'):
        cli.add_argument('--' + name, type=Path, required=True)
    cli.add_argument('--launch', action='store_true')
    args = cli.parse_args()
    task, coord, stage, previous = (path.resolve(strict=True) for path in
                                    (args.task_root, args.coord, args.stage, args.previous_prefill))
    assert previous.is_relative_to(task)
    before = read(previous / 'state.json')
    host = socket.gethostname()
    assert before['role'] == 'p' and before['host'] == host
    same_stage(task, stage)
    availability = inspect(coord, host)
    if not args.launch or not availability['available_in_snapshot']:
        print(json.dumps(availability, sort_keys=True))
        return 0 if availability['available_in_snapshot'] else 3
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    run = stage / 'actual_short_npu_g1450'
    outcome = own_slot(coord, host, previous, run, task)
    outcome['device_process_list_clear_and_lease_released'] = True
    save(run / 'outcome.json', outcome)
    print(json.dumps(outcome, sort_keys=True))
    return 0 if outcome['all_short_cases_exact'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
