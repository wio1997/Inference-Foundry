"""One owned GLM-5.3 PD functional reference; no performance or kernel patch.

Existing controller owns scheduling. Signals require boot/start-tick identity;
all device owners must belong to the inspected native tree. Historical state,
weights, installed packages and kernels are never modified by this script.
"""
import hashlib
import json
import os
import re
import runpy
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PLAN = ROOT / 'functional_plan.json'
PYTHON = '/usr/local/python3.12.13/bin/python3'


def write(name, value):
    path = ROOT / name
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w') as f:
        json.dump(value, f, indent=2)
        f.write('\n')
    os.replace(tmp, path)


def identity(pid):
    try:
        p = Path('/proc') / str(pid)
        parts = (p / 'stat').read_text().rsplit(')', 1)[1].split()
        return dict(pid=int(pid), boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                    start_ticks=parts[19], state=parts[0], ppid=int(parts[1]))
    except FileNotFoundError:
        return None


def live(expected):
    now = identity(expected['pid'])
    return bool(now and now['state'] != 'Z' and all(now[k] == expected[k]
                for k in ('pid', 'boot_id', 'start_ticks')))


def catalogue():
    out = {}
    for p in Path('/proc').iterdir():
        if p.name.isdigit():
            item = identity(int(p.name))
            if item and item['state'] != 'Z':
                out[item['pid']] = item
    return out


def device_owners():
    text = subprocess.check_output(['/usr/local/bin/npu-smi', 'info'], text=True, timeout=20)
    owners = set()
    for line in text.splitlines():
        columns = [c.strip() for c in line.split('|')]
        if (len(columns) > 4 and len(columns[1].split()) == 2
                and all(x.isdigit() for x in columns[1].split()) and columns[2].isdigit()):
            owners.add(int(columns[2]))
    return owners, text


def tree(root, table):
    selected = {root}
    while True:
        expanded = selected | {pid for pid, p in table.items() if p['ppid'] in selected}
        if expanded == selected:
            return [table[p] for p in selected if p in table]
        selected = expanded


def fetch(url, body=None, timeout=4):
    raw = None if body is None else json.dumps(body, ensure_ascii=False).encode()
    req = urllib.request.Request(url, data=raw, headers={'content-type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return response.status, response.read()


def assert_idle(port):
    status, raw = fetch('http://127.0.0.1:%d/metrics' % port)
    assert status == 200
    samples = [line for line in raw.decode().splitlines() if not line.startswith('#')
               and re.match(r'vllm:num_requests_(running|waiting)(\{|\s)', line)]
    assert samples and all(float(line.rsplit(' ', 1)[1]) == 0 for line in samples), samples


def snapshot(host, plan):
    entry = plan['old'][host]
    assert live(entry['root']), 'native root changed'
    table = catalogue()
    members = tree(entry['root']['pid'], table)
    owners, text = device_owners()
    assert len(owners) == 16 and owners == {p['pid'] for p in entry['workers']}
    assert owners.issubset({p['pid'] for p in members}), 'foreign device owner'
    assert all(live(p) for p in entry['workers']), 'worker identity changed'
    if host == '166':
        owner = json.loads(Path('/data/tiankuan/wio/glm52-pd/controller-owner.json').read_text())
        assert owner['run_id'] == plan['run_id'] and owner['status'] == 'running'
        assert live(owner['owner']), 'unique controller is absent'
        assert all(live(p) for p in entry['public']), 'public/observer identity changed'
        assert_idle(8000)
    assert_idle(plan['ports'][host])
    connections = subprocess.check_output(['ss', '-Htn', 'state', 'established'], text=True)
    connections = [line for line in connections.splitlines() if re.search(r':(?:8000|9081|9900)\s', line)]
    args = Path('/proc/%s/cmdline' % entry['root']['pid']).read_bytes().split(b'\0')[:-1]
    env = dict(x.decode().split('=', 1) for x in Path('/proc/%s/environ' % entry['root']['pid']).read_bytes().split(b'\0') if b'=' in x)
    keys = {'PATH', 'LD_LIBRARY_PATH', 'PYTHONPATH', 'ASCEND_HOME_PATH', 'ASCEND_OPP_PATH', 'CANN_HOME'}
    saved = dict(host=host, root=entry['root'], members=members, public=entry.get('public', []),
                 argv=[x.decode() for x in args], environment={k: v for k, v in env.items() if k in keys},
                 npu_owners=sorted(owners), npu_raw=text, established_connections=connections)
    write('before_%s.json' % host, saved)
    return dict(host=host, owners=sorted(owners), members=len(members), idle=True)


def signal_verified(p, sig):
    if live(p):
        os.kill(p['pid'], sig)


def stop_tree(members, root):
    signal_verified(root, signal.SIGTERM)
    deadline = time.monotonic() + 30
    while any(live(p) for p in members) and time.monotonic() < deadline:
        time.sleep(1)
    for p in members:
        signal_verified(p, signal.SIGTERM)
    deadline = time.monotonic() + 10
    while any(live(p) for p in members) and time.monotonic() < deadline:
        time.sleep(1)
    for p in members:
        signal_verified(p, signal.SIGKILL)
    deadline = time.monotonic() + 10
    while any(live(p) for p in members) and time.monotonic() < deadline:
        time.sleep(1)
    assert not any(live(p) for p in members), 'owned processes still live'


def stop_old(host, plan):
    before = json.loads((ROOT / ('before_%s.json' % host)).read_text())
    if host == '166':
        # Observer is stopped before retiring the old native epochs; state is retained.
        for p in reversed(before['public']):
            stop_tree([p], p)
    assert_idle(plan['ports'][host])
    stop_tree(before['members'], before['root'])
    deadline = time.monotonic() + 90
    owners, text = device_owners()
    while owners and time.monotonic() < deadline:
        assert owners.issubset({p['pid'] for p in before['members']}), 'foreign device owner appeared'
        time.sleep(3)
        owners, text = device_owners()
    assert not owners, 'device memory owner not released'
    write('retired_%s.json' % host, dict(owned_processes_exited=True, npu_raw=text,
                                        historical_state_deleted=False))
    return dict(host=host, retired=True, device_owners=[])


def native(host, plan):
    fd = os.open(ROOT / ('native_%s.log' % host), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.dup2(fd, 1)
    os.dup2(fd, 2)
    os.close(fd)
    write('namespace_owner_%s.json' % host, identity(os.getpid()))
    args = plan['native_args'][host]
    print(json.dumps(dict(event='native_role_start', host=host, args=args)), flush=True)
    sys.argv = [str(ROOT / 'runtime_bundle/native_acl_lifecycle.py'), 'cli'] + args
    runpy.run_path(sys.argv[0], run_name='__main__')


def launch(host, plan):
    assert not device_owners()[0], 'devices are not free'
    env = json.loads((ROOT / ('before_%s.json' % host)).read_text())['environment']
    env['PYTHONPATH'] = ':'.join(p for p in env.get('PYTHONPATH', '').split(':')
                               if '/glm52-pd/deploy/plugins/' not in p)
    env.update(plan['environment'][host])
    command = ['docker', 'exec', '-d']
    for key, value in sorted(env.items()):
        command += ['-e', key + '=' + value]
    command += ['glm52-single', PYTHON, str(Path(__file__).resolve()), 'native', host]
    subprocess.run(command, check=True, timeout=30)
    write('launch_%s.json' % host, dict(command=command, launch_exit=0, readiness='unknown'))
    return dict(host=host, launch_exit=0, readiness='unknown')


def new_owner(host):
    path = ROOT / ('namespace_owner_%s.json' % host)
    if not path.exists():
        return None
    saved = json.loads(path.read_text())
    for pid, p in catalogue().items():
        if any(p[k] != saved[k] for k in ('boot_id', 'start_ticks')):
            continue
        try:
            line = next(x for x in Path('/proc/%d/status' % pid).read_text().splitlines() if x.startswith('NSpid:'))
            if int(line.split()[-1]) == saved['pid']:
                return p
        except (FileNotFoundError, StopIteration):
            continue
    return None


def poll(host, plan):
    root = new_owner(host)
    members = tree(root['pid'], catalogue()) if root else []
    if root:
        write('owned_%s.json' % host, dict(root=root, members=members))
    ready = False
    try:
        status, _ = fetch('http://127.0.0.1:%d/health' % plan['ports'][host])
        status2, raw = fetch('http://127.0.0.1:%d/v1/models' % plan['ports'][host])
        ready = bool(root and status == status2 == 200 and any(x['id'] == 'glm-53' for x in json.loads(raw)['data']))
    except Exception:
        pass
    log = ROOT / ('native_%s.log' % host)
    tail = log.read_bytes()[-65536:].decode(errors='replace') if log.exists() else ''
    failure = any(marker in tail for marker in ['Engine core initialization failed', 'EngineCore failed to start', 'Traceback (most recent call last)'])
    return dict(host=host, root=root, ready=ready, failure=failure, tail=tail[-12000:])


def cleanup(host, plan):
    root = new_owner(host)
    if root:
        stop_tree(tree(root['pid'], catalogue()), root)
    owned = ROOT / ('owned_%s.json' % host)
    if owned.exists():
        record = json.loads(owned.read_text())
        table = catalogue()
        selected = {p['pid']: p for p in record['members']}
        for p in record['members']:
            if live(p):
                selected.update({x['pid']: x for x in tree(p['pid'], table)})
        stop_tree(list(selected.values()), record['root'])
    owners, text = device_owners()
    write('cleanup_%s.json' % host, dict(npu_owners=sorted(owners), npu_raw=text,
                                        weights_or_installed_sources_modified=False))
    return dict(host=host, remaining_device_owners=sorted(owners))


def call(host, action):
    args = ['/usr/bin/python3', str(Path(__file__).resolve()), action, host]
    if host == '167':
        args = ['ssh', '-o', 'BatchMode=yes', 'root@172.16.10.167'] + args
    p = subprocess.run(args, capture_output=True, text=True, timeout=160)
    print(json.dumps(dict(action=action, host=host, exit=p.returncode)), flush=True)
    if p.returncode:
        raise RuntimeError('%s/%s exit%d: %s' % (host, action, p.returncode, p.stderr[-6000:]))
    return json.loads(p.stdout)


def probe(plan):
    prompt = '请阅读以下动态请求并简短回答。\n' + ('推理框架需要保持请求语义和有效输出。 ' * 192) + '\n问题：P和D分别负责什么？'
    body = dict(model='glm-53', messages=[dict(role='user', content=prompt)], max_tokens=32,
                min_tokens=32, temperature=0, ignore_eos=True, return_token_ids=True,
                cache_salt='glm53-functional-245')
    helper = dict(body, stream=False, max_tokens=1, min_tokens=1,
                  kv_transfer_params=dict(do_remote_decode=True, do_remote_prefill=False))
    started = time.monotonic_ns()
    ps, raw = fetch('http://172.16.10.166:9081/v1/chat/completions', helper, timeout=300)
    (ROOT / 'helper.raw').write_bytes(raw)
    value = json.loads(raw)
    kv = value.get('kv_transfer_params')
    assert ps == 200 and value['usage']['completion_tokens'] == 1
    assert isinstance(kv, dict) and kv['do_remote_prefill'] is True
    assert kv['remote_engine_id'] == plan['engine_ids']['166'] and kv['remote_host'] == '172.16.10.166'
    assert kv['remote_port'] == 36000 and kv['remote_dcp_size'] == 16 and kv['remote_pcp_size'] == 1
    assert kv['remote_block_ids'] and all(row for row in kv['remote_block_ids'])
    db = dict(body, stream=False, kv_transfer_params=kv)
    write('public_request.json', body)
    write('helper_request.json', helper)
    write('decoder_request.json', db)
    ds, out = fetch('http://172.16.10.167:9900/v1/chat/completions', db, timeout=300)
    (ROOT / 'decoder.raw').write_bytes(out)
    decoded = json.loads(out)
    assert ds == 200 and decoded['usage']['completion_tokens'] == 32
    assert decoded['usage']['prompt_tokens'] == value['usage']['prompt_tokens']
    cached = (decoded['usage'].get('prompt_tokens_details') or {}).get('cached_tokens')
    assert cached and cached > 0, 'native external KV reuse not established'
    choice = decoded['choices'][0]
    assert choice['finish_reason'] != 'error' and choice.get('token_ids') and len(choice['token_ids']) == 32
    assert any(choice['message'].get(k) for k in ('content', 'reasoning', 'reasoning_content'))
    result = dict(P_status=ps, D_status=ds, helper_internal_tokens=1, effective_D_tokens=32,
                  prompt_tokens=decoded['usage']['prompt_tokens'], cached_tokens=cached,
                  host_wall_s=(time.monotonic_ns()-started)/1e9, kv_metadata=kv,
                  standard_PD_function=True, full_API_contract=False, performance_gain=None)
    write('functional_result.json', result)
    return result


def workflow(plan):
    launched = False
    fitted = False
    try:
        for host in ('166', '167'):
            call(host, 'snapshot')
        for host in ('166', '167'):
            call(host, 'stop')
        launched = True
        for host in ('166', '167'):
            call(host, 'launch')
        deadline = time.monotonic() + 1800
        while True:
            rows = {h: call(h, 'poll') for h in ('166', '167')}
            write('readiness.json', rows)
            print(json.dumps({h: {k: v[k] for k in ('ready', 'failure', 'root')} for h, v in rows.items()}), flush=True)
            if any(v['failure'] for v in rows.values()):
                raise RuntimeError('native initialization failure; see frozen raw logs')
            if all(v['ready'] for v in rows.values()):
                fitted = True
                break
            if time.monotonic() > deadline:
                raise TimeoutError('model readiness deadline exceeded')
            time.sleep(10)
        result = probe(plan)
        write('final_status.json', dict(status='completed', result=result,
                                       active_roles='P166/D167', public8000='old gateway stopped; full contract not accepted'))
        print(json.dumps(result), flush=True)
    except BaseException as error:
        clean = {}
        if launched and not fitted:
            for h in ('166', '167'):
                try:
                    clean[h] = call(h, 'cleanup')
                except Exception as e:
                    clean[h] = dict(error=str(e))
        write('final_status.json', dict(status='failed', error=type(error).__name__ + ': ' + str(error),
                                       cleanup=clean, native_fit=fitted,
                                       active_roles='healthy P166/D167 retained for diagnosis' if fitted else 'see cleanup',
                                       public8000='old gateway stopped; full contract not accepted',
                                       no_old_weight_fallback=True))
        raise


if __name__ == '__main__':
    action = sys.argv[1]
    plan = json.loads(PLAN.read_text())
    if action == 'workflow':
        workflow(plan)
    else:
        host = sys.argv[2]
        functions = dict(snapshot=snapshot, stop=stop_old, launch=launch, native=native, poll=poll, cleanup=cleanup)
        value = functions[action](host, plan)
        if value is not None:
            print(json.dumps(value))
