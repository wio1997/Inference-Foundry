"""Hold an independently bounded, CPU-only client using the existing task resource lock."""
import argparse
from datetime import datetime, timedelta, timezone
import fcntl
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import time

HOST_CONTRACT = ('CpusetCpus', 'NanoCpus', 'Memory', 'NetworkMode', 'ReadonlyRootfs',
                 'Privileged', 'Devices', 'DeviceRequests', 'LogConfig', 'SecurityOpt',
                 'CapAdd', 'CapDrop', 'PidMode', 'IpcMode')


def read(path):
    return json.loads(path.read_bytes())


def save(path, state):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(state, indent=2) + '\n')
    temporary.replace(path)


def utcnow():
    return datetime.now(timezone.utc)


def command(*argv, timeout=20):
    return subprocess.check_output(argv, text=True, timeout=timeout)


def inspect(reference, *, missing_ok=False):
    result = subprocess.run(['docker','inspect','--type','container',reference], text=True, capture_output=True, timeout=15)
    if result.returncode != 0:
        if missing_ok and 'No such' in result.stderr:
            return None
        raise RuntimeError(result.stderr.strip())
    item, = json.loads(result.stdout)
    return item


def immutable_hash(item):
    fields = {name: item[name] for name in ('Id', 'Image', 'Config', 'HostConfig')}
    fields['Mounts'] = sorted(item['Mounts'], key=lambda m: (m['Destination'], m['Source']))
    return sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest()


def mounts(item):
    return sorted((m['Type'], m['Source'], m['Destination'], m['RW']) for m in item['Mounts'])


def execution_contract(item):
    return {
        'image': item['Image'],
        'host': {key: item['HostConfig'].get(key) for key in HOST_CONTRACT},
        'mounts': mounts(item),
        'workdir': item['Config']['WorkingDir'],
        'entrypoint': item['Config']['Entrypoint'],
        'user': item['Config']['User'],
        'environment': sorted(item['Config']['Env']),
    }


def current_reference(ticket, *, stopped):
    item = inspect(ticket['predecessor_container_ID'])
    assert item['Image'] == ticket['cpu_client_image_ID']
    assert immutable_hash(item) == ticket['predecessor_immutable_config_sha256']
    labels = item['Config']['Labels']
    assert labels['codex.task'] == Path(ticket['task_root']).name
    assert labels['codex.run'] == Path(ticket['client_directory']).name
    if stopped:
        assert not item['State']['Running'], 'Original CPU container must be stopped by its owner'
    host = item['HostConfig']
    assert host['CpusetCpus'] == ticket['cpuset']
    assert host['NanoCpus'] == 4_000_000_000 and host['Memory'] == 32 * 1024**3
    assert host['NetworkMode']=='host' and host['ReadonlyRootfs']
    assert not host['Privileged'] and not host['Devices'] and not host.get('DeviceRequests')
    assert host['LogConfig'] == {'Type':'none', 'Config':{}}
    assert item['Config']['Entrypoint'] == ['sleep'] and item['Config']['Cmd'] == ['72000']
    return item


def validate_ticket(ticket):
    assert ticket['schema'] == 1 and ticket['host'] == socket.gethostname() and os.geteuid() == 0
    assert ticket['controller_sha256'] == sha256(Path(__file__).read_bytes()).hexdigest()
    root = Path(ticket['task_root']).resolve(strict=True)
    client = Path(ticket['client_directory']).resolve(strict=True)
    assert client.parent == root and client.name == 'client_ds41_g1122'
    lock = Path(ticket['canonical_cpu_lock']).resolve(strict=True)
    assert ticket['cpuset'] == '184-187'
    assert lock.name == socket.gethostname() + '-cpu' + ticket['cpuset'] + '.lock'
    assert re.fullmatch(r'yxy-ds41-aisbench-g[0-9]+-(?:128|256)k', ticket['successor_container_name'])
    assert ticket['input_tokens'] in (131072, 262144)
    assert all(type(ticket[key]) is int and ticket[key] > 0 for key in (
        'lease_seconds', 'minimum_required_formal_window_seconds', 'container_sleep_seconds'))
    assert ticket['container_sleep_seconds'] > ticket['lease_seconds'] > ticket['minimum_required_formal_window_seconds']
    return lock, lock.with_suffix('.owner.json')


def creation_argv(ticket, reference):
    root = Path(ticket['task_root'])
    client = Path(ticket['client_directory'])
    argv = ['docker','create','--name',ticket['successor_container_name'],
            '--label',f'codex.task={root.name}','--label',f'codex.run={client.name}',
            '--label',f'codex.controller_pid={os.getpid()}','--network','host','--log-driver','none',
            '--read-only','--cpuset-cpus',ticket['cpuset'],'--cpus','4','--memory','32g']
    for kind, source, destination, writable in mounts(reference):
        assert kind == 'bind' and Path(source).is_dir()
        assert not (':' in source or ':' in destination)
        argv += ['-v', source+':'+destination+('' if writable else ':ro')]
    env = dict(value.split('=',1) for value in reference['Config']['Env'])
    for key in ('TORCH_DEVICE_BACKEND_AUTOLOAD','TOKENIZERS_PARALLELISM','OMP_NUM_THREADS','TMPDIR','HF_HOME'):
        argv += ['-e', key+'='+env[key]]
    return argv+['-w', reference['Config']['WorkingDir'], '--entrypoint','sleep',
                 ticket['cpu_client_image_ID'], str(ticket['container_sleep_seconds'])]


def prove_ownership(item, ticket, pid):
    labels = item['Config']['Labels']
    assert labels['codex.controller_pid'] == str(pid)
    assert labels['codex.task'] == Path(ticket['task_root']).name
    assert labels['codex.run'] == Path(ticket['client_directory']).name


def report(ticket, state_path, lock, owner_path):
    state = read(state_path)
    assert state['phase'] == 'ready'
    item = inspect(state['container_ID'])
    prove_ownership(item, ticket, state['pid'])
    assert item['State']['Running'] and item['State']['StartedAt'] == state['container_started_at']
    assert immutable_hash(item) == state['immutable_config_SHA256']
    owner = read(owner_path)
    assert owner['pid'] == state['pid'] and owner['generation'] == state['owner_generation'] and owner['state'] == 'claimed'
    proc = Path('/proc',str(state['pid']))
    assert str(Path(__file__).resolve()).encode() in (proc/'cmdline').read_bytes().split(b'\0')
    assert any(fd.resolve() == lock for fd in (proc/'fd').iterdir())
    assert utcnow() < datetime.fromisoformat(state['conservative_deadline_UTC'])
    return {'container_ID':item['Id'], 'CPU_controller_PID':state['pid'], 'CPU_owner_generation':owner['generation'],
            'conservative_deadline_UTC':state['conservative_deadline_UTC'], 'lease':'wide_cpu_only',
            'immutable_config_SHA256':state['immutable_config_SHA256'], 'reported_at_UTC':utcnow().isoformat()}


def hold(ticket, state_path, lock, owner_path):
    assert not state_path.exists() and not state_path.with_name('stop.request').exists()
    with lock.open('r') as descriptor:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        previous = read(owner_path)
        assert previous['state'] == 'released' and previous['task_id'] == Path(ticket['task_root']).name
        assert previous['run_directory'] == ticket['client_directory']
        wanted = set(range(184,188))
        for path in lock.parent.glob(ticket['host'] + '-cpu*.owner.json'):
            if read(path)['state'] == 'released':
                continue
            interval = re.search(r'-cpu(\d+)(?:-(\d+))?\.owner\.json$',path.name)
            assert interval, ('Unknown claimed CPU lock', path.name)
            assert not wanted.intersection(range(int(interval[1]),int(interval[2] or interval[1])+1))
        reference = current_reference(ticket, stopped=True)
        name = ticket['successor_container_name']
        assert inspect(name, missing_ok=True) is None
        pid = os.getpid()
        owner = {**previous, 'pid':pid,'state':'claimed','generation':previous['generation']+1}
        state = {'pid':pid,'phase':'claimed','owner_generation':owner['generation']}
        save(owner_path, owner)
        item = None
        outcome = 'lease_expired'

        def interrupted(*_):
            raise InterruptedError('Owned CPU window controller interrupted')

        try:
            signal.signal(signal.SIGTERM,interrupted)
            signal.signal(signal.SIGINT,interrupted)
            save(state_path,state)
            cid = command(*creation_argv(ticket,reference),timeout=60).strip()
            item = inspect(cid)
            prove_ownership(item,ticket,pid)
            assert execution_contract(item) == execution_contract(reference)
            assert item['Config']['Cmd'] == [str(ticket['container_sleep_seconds'])]
            cutoff = time.monotonic()+ticket['lease_seconds']
            expiry = utcnow()+timedelta(seconds=ticket['lease_seconds'])
            command('docker','start',cid,timeout=30)
            item = inspect(cid)
            assert item['State']['Running']
            state.update(phase='ready',container_ID=cid,container_started_at=item['State']['StartedAt'],
                         conservative_deadline_UTC=expiry.isoformat(),immutable_config_SHA256=immutable_hash(item))
            save(state_path,state)
            while time.monotonic()<cutoff and not state_path.with_name('stop.request').exists():
                assert inspect(cid)['State']['Running']
                time.sleep(min(10,max(0,cutoff-time.monotonic())))
            if state_path.with_name('stop.request').exists():
                outcome = 'owner_stop_request'
        except BaseException as error:
            outcome = 'failed'
            state['error'] = type(error).__name__
            raise
        finally:
            item = item or inspect(name, missing_ok=True)
            if item is not None:
                prove_ownership(item,ticket,pid)
                actual = inspect(item['Id'])
                if actual['State']['Running']:
                    command('docker','stop','--time','10',item['Id'],timeout=30)
                assert not inspect(item['Id'])['State']['Running']
            assert read(owner_path) == owner
            state.update(phase='released',outcome=outcome,released_at=utcnow().isoformat())
            save(state_path,state)
            save(owner_path,{**owner, 'state':'released'})


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('action', choices=('check','report','hold'))
    cli.add_argument('--assignment', type=Path, required=True)
    args = cli.parse_args()
    ticket = read(args.assignment)
    lock, owner_path = validate_ticket(ticket)
    assert args.assignment.resolve().is_relative_to(Path(ticket['client_directory']).resolve())
    state_path = args.assignment.with_name('state.json')
    if args.action == 'hold':
        hold(ticket,state_path,lock,owner_path)
    elif args.action == 'report':
        print(json.dumps(report(ticket,state_path,lock,owner_path)))
    else:
        original = current_reference(ticket,stopped=False)
        argv = creation_argv(ticket,original)
        assert argv[0:2] == ['docker','create'] and '--device' not in argv and '--gpus' not in argv
        print(json.dumps({'same_config_reference_verified':True,'proposed_create_argv_verified':True,
                          'original_running':original['State']['Running'],
                          'owner_state':read(owner_path)['state'],'candidate_lifetime_seconds':ticket['lease_seconds'],
                          'container_started_or_stopped':False}))


if __name__=='__main__':
    main()
