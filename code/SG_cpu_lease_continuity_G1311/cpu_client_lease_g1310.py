"""Verify or renew the original task-owned, CPU-only AISBench container."""
import argparse
from datetime import datetime, timedelta, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import select
import signal
import socket
import subprocess
import sys
import time
import urllib.request


ROOT = Path('/PRIVATE_TASK_ROOT')
CLIENT = ROOT / 'client_ds41_g1122'
CONTROL = CLIENT / 'cpu_client_renewal_g1310'
LOCK = Path('/root/.codex/coord/<DECODE_HOSTNAME>-cpu184-187.lock')
OWNER = LOCK.with_suffix('.owner.json')
CID = 'd8593a995ef7192cfd5fd9e696d652e24b3d02d77253d2b96a3415a4097f9090'
IMAGE = 'sha256:05358d58d23f2679c71e48d93797d5d2a28d72eedc7fb4900479c89955465d76'
INITIAL_CONTROLLER = 2760614
CLIENT_LIFETIME = 70000


def now():
    return datetime.now(timezone.utc)


def read(path):
    return json.loads(path.read_bytes())


def update(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def command(*argv, timeout=15):
    return subprocess.check_output(argv, text=True, cwd=CLIENT, timeout=timeout)


def process_argv(pid):
    try:
        return Path('/proc', str(pid), 'cmdline').read_bytes().split(b'\0')
    except FileNotFoundError:
        return []


def prove_owner(pid, source):
    assert str(source).encode() in process_argv(pid)
    assert any(fd.resolve() == LOCK for fd in Path('/proc', str(pid), 'fd').iterdir())
    owner = read(OWNER)
    assert owner['pid'] == pid and owner['state'] == 'claimed'
    assert owner['task_id'] == ROOT.name and owner['run_directory'] == str(CLIENT)
    return owner


def inspect_client():
    item = json.loads(command('docker', 'inspect', CID))[0]
    assert item['Id'] == CID and item['Image'] == IMAGE
    config, host = item['Config'], item['HostConfig']
    assert config['Labels']['codex.controller_pid'] == str(INITIAL_CONTROLLER)
    assert config['Labels']['codex.task'] == ROOT.name and config['Labels']['codex.run'] == CLIENT.name
    assert config['Entrypoint'] == ['sleep'] and config['Cmd'] == ['72000']
    assert host['CpusetCpus'] == '184-187' and not host['Devices'] and not host['Privileged']
    assert host['ReadonlyRootfs'] and host['Memory'] == 34359738368 and host['NetworkMode'] == 'host'
    return item


def immutable_sha(item):
    data = {name:item[name] for name in ('Id','Image','Config','HostConfig')}
    data['Mounts'] = sorted(item['Mounts'], key=lambda value:(value['Destination'],value['Source']))
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def client_lease():
    item = inspect_client()
    assert item['State']['Running']
    renewed_state = CONTROL / 'state.json'
    if renewed_state.exists():
        state = read(renewed_state)
        assert state['phase'] == 'ready' and state['container_started_at'] == item['State']['StartedAt']
        assert state['immutable_config_SHA256'] == immutable_sha(item)
        assert state['container_ID'] == CID and state['previous_controller_PID'] == INITIAL_CONTROLLER
        owner = prove_owner(state['pid'], Path(__file__).resolve())
        assert owner['generation'] == state['owner_generation']
        deadline = datetime.fromisoformat(state['conservative_deadline_UTC'])
        identity = 'renewed_same_container'
    else:
        state = read(CLIENT / 'controller_state.json')
        assert state['pid'] == INITIAL_CONTROLLER and state['container_id'] == CID
        owner = prove_owner(state['pid'], CLIENT / 'cpu_client_controller.py')
        deadline = datetime.fromisoformat(state['at']) + timedelta(seconds=CLIENT_LIFETIME)
        identity = 'original'
    assert (deadline - now()).total_seconds() > 0
    return {'container_ID':CID,'CPU_controller_PID':state['pid'],'CPU_owner_generation':owner['generation'],
            'conservative_deadline_UTC':deadline.isoformat(),'immutable_config_SHA256':immutable_sha(item),
            'lease':identity,'reported_at_UTC':now().isoformat()}


def require_idle(run_id):
    prior = CLIENT / 'runs' / 'sglang' / run_id
    state = read(prior / 'state.json')
    assert state['run_id'] == run_id and read(prior / 'run_manifest.json')['client_container_id'] == CID
    if state['phase'] == 'warmup_paused_for_P_snapshot':
        count = state['completed_warmups']
        assert 1 <= count <= 4
    else:
        assert state['phase'] == 'warmup_diagnostics_complete_pending_compile_audit' and state['readiness_passed']
        count = 5
    length = read(prior / 'run_manifest.json')['lengths'][0]
    for index in range(1,count+1):
        audited=read(prior / f'warmup{index}_{length}_audit.json')
        assert audited['fully_validated'] and audited['requests']==audited['successful_requests']==16
    assert not any(path.name.startswith(('cold_','hot_','before_cold_','before_hot_')) for path in prior.iterdir())
    rows = command('docker','top',CID,'-eo','pid,args').splitlines()
    assert len(rows) == 2 and 'sleep 72000' in rows[1]
    scope = {'prefill':('num_prefill_bootstrap_queue_reqs','num_prefill_inflight_queue_reqs'),
             'decode':('num_decode_prealloc_queue_reqs','num_decode_transfer_queue_reqs')}
    roles = read(prior / 'run_manifest.json')['gate']['roles']
    for role, names in scope.items():
        with urllib.request.urlopen(roles[role]['metrics_url'], timeout=8) as response:
            assert response.status == 200
            raw = response.read().decode()
        for name in ('num_running_reqs','num_queue_reqs',*names):
            values = [float(x) for x in re.findall(r'^sglang:'+name+r'(?:\{[^\n]*\})?\s+([0-9.eE+\-]+)',raw,re.M)]
            assert values and all(x == 0 for x in values), (role,name,values)
    return state, count


def renew(run_id):
    assert run_id == Path(run_id).name and run_id not in ('.','..')
    with (CLIENT/'sglang_runner.lock').open('a+') as runner_lock:
        fcntl.flock(runner_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        before=client_lease()
        assert before['lease']=='original' and not CONTROL.exists()
        assert not (CLIENT/'stop.request').exists(), 'The original owner already has a release request'
        state,count=require_idle(run_id)
        CONTROL.mkdir(exist_ok=False)
        ticket={'run_id':run_id,'terminal_warmup_state':state,'completed_warmups':count,'old_lease':before}
        update(CONTROL/'assignment.json',ticket)
        with os.fdopen(os.pidfd_open(INITIAL_CONTROLLER),'rb') as pidfd:
            prove_owner(INITIAL_CONTROLLER,CLIENT/'cpu_client_controller.py')
            with (CLIENT/'stop.request').open('x') as marker:
                marker.write('Release the idle original CPU client for same-container renewal.\n')
            selector=select.poll()
            selector.register(pidfd,select.POLLIN)
            assert selector.poll(45_000), 'Inspect the same exiting original controller'
        previous=read(OWNER)
        assert previous['pid']==INITIAL_CONTROLLER and previous['state']=='released'
        assert not inspect_client()['State']['Running']
        with (CONTROL/'controller.log').open('x') as log:
            child=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'hold'],cwd=CLIENT,
                                   stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        update(CONTROL/'actual_launch_handle.json',{'at':now().isoformat(),'PID':child.pid,'container_ID':CID})
        for _ in range(60):
            assert child.poll() is None, 'Inspect successor log and the same container identity'
            state_file=CONTROL/'state.json'
            if state_file.exists() and read(state_file)['phase']=='ready':
                proven=client_lease()
                assert proven['CPU_controller_PID']==child.pid
                update(CONTROL/'actual_renewal_receipt.json',proven)
                return proven
            time.sleep(0.25)
        raise TimeoutError('Successor readiness unconfirmed; inspect the existing launch handle')


def hold():
    ticket=read(CONTROL/'assignment.json')
    before=ticket['old_lease']
    with LOCK.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        previous=read(OWNER)
        assert previous['pid']==INITIAL_CONTROLLER and previous['state']=='released'
        for path in OWNER.parent.glob('<DECODE_HOSTNAME>-cpu*.owner.json'):
            if read(path)['state']=='released':
                continue
            slots=re.search(r'-cpu(\d+)(?:-(\d+))?\.owner\.json$',path.name)
            assert slots and not set(range(int(slots[1]),int(slots[2] or slots[1])+1)) & set(range(184,188))
        original=inspect_client()
        assert not original['State']['Running'] and immutable_sha(original)==before['immutable_config_SHA256']
        claim={**previous,'state':'claimed','pid':os.getpid(),'generation':previous['generation']+1}
        update(OWNER,claim)
        deadline_start=now()
        expires=time.monotonic()+CLIENT_LIFETIME
        state={'phase':'starting','pid':os.getpid(),'previous_controller_PID':INITIAL_CONTROLLER,
               'container_ID':CID,'immutable_config_SHA256':before['immutable_config_SHA256'],
               'owner_generation':claim['generation'],
               'conservative_deadline_UTC':(deadline_start+timedelta(seconds=CLIENT_LIFETIME)).isoformat()}

        def stop(*_):
            raise InterruptedError('This owned CPU lease was stopped')

        signal.signal(signal.SIGTERM,stop)
        signal.signal(signal.SIGINT,stop)
        try:
            update(CONTROL/'state.json',state)
            command('docker','start',CID,timeout=30)
            active=inspect_client()
            assert active['State']['Running'] and immutable_sha(active)==state['immutable_config_SHA256']
            state.update(phase='ready',container_started_at=active['State']['StartedAt'])
            update(CONTROL/'state.json',state)
            while time.monotonic()<expires and not (CONTROL/'stop.request').exists():
                time.sleep(min(15,max(0,expires-time.monotonic())))
                assert inspect_client()['State']['Running']
        finally:
            assert read(OWNER)==claim
            actual=inspect_client()
            assert immutable_sha(actual)==state['immutable_config_SHA256']
            if actual['State']['Running']:
                command('docker','stop','--time','10',CID,timeout=30)
            assert not inspect_client()['State']['Running']
            state.update(phase='released',ended_at=now().isoformat())
            update(CONTROL/'state.json',state)
            update(OWNER,{**claim,'state':'released'})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('report','renew','hold'))
    parser.add_argument('--run-id')
    args=parser.parse_args()
    assert socket.gethostname()=='<DECODE_HOSTNAME>' and Path(__file__).resolve().parent==CLIENT
    if args.action=='hold':
        hold()
    elif args.action=='renew':
        assert args.run_id
        print(json.dumps(renew(args.run_id),indent=2))
    else:
        print(json.dumps(client_lease(),indent=2))


if __name__=='__main__':
    main()
