"""Bounded native PD profiling diagnostic; existing unique controller owns it.

No kernels, installed service code, or request semantics are patched. The eight
output tokens below define a diagnostic fixture, not a product restriction.
"""
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PLAN = json.loads((ROOT / 'functional_plan.json').read_text())
spec = importlib.util.spec_from_file_location('owned_native', PLAN['reuse_library'])
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
m.ROOT = ROOT
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def fetch(url, body=None, timeout=10):
    data = None if body is None else json.dumps(body, ensure_ascii=False).encode()
    request = urllib.request.Request(url, data=data, headers={'content-type': 'application/json'})
    try:
        with OPENER.open(request, timeout=timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()


m.fetch = fetch


def owner_guard(host):
    if host == '166':
        owner = json.loads(Path('/data/tiankuan/wio/glm52-pd/controller-owner.json').read_text())
        assert owner['run_id'] == PLAN['run_id'] and owner['status'] == 'running'
        assert m.live(owner['owner'])


def installed_guard():
    code = 'import pathlib,hashlib,json;print(json.dumps({p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest() for p in %r}))' % list(PLAN['installed_sources'])
    raw = subprocess.check_output(['docker', 'exec', 'glm52-single', m.PYTHON, '-c', code], text=True, timeout=20)
    assert json.loads(raw) == PLAN['installed_sources'], 'installed profiler sources changed'


def guard(host):
    owner_guard(host)
    installed_guard()
    root = PLAN['retire_roles'][host]
    assert m.live(root), 'resident role identity changed; no signals'
    members = m.tree(root['pid'], m.catalogue())
    owners, raw = m.device_owners()
    assert len(owners) == 16 and owners.issubset({p['pid'] for p in members}), 'foreign device owner; no signals'
    m.assert_idle(PLAN['ports'][host])
    status, _ = fetch('http://127.0.0.1:%s/health' % PLAN['ports'][host])
    assert status == 200
    result = dict(root=root, members=members, device_owners=sorted(owners), npu_raw=raw, idle=True)
    m.write('retire_guard_%s.json' % host, result)
    return result


def retire(host):
    row = guard(host)  # Revalidate immediately before any signal.
    m.stop_tree(row['members'], row['root'])
    assert not m.device_owners()[0], 'remaining device owners; no launch'
    return dict(host=host, owned_native_stopped=True)


def launch(host):
    assert not m.device_owners()[0]
    env = json.loads((ROOT / PLAN['baseline_env'][host]).read_text())['environment']
    env.update(PLAN['environment'][host])
    env['PYTHONPATH'] = ':'.join(p for p in env.get('PYTHONPATH', '').split(':') if '/glm52-pd/deploy/plugins/' not in p)
    command = ['docker', 'exec', '-d']
    for key, value in sorted(env.items()):
        command += ['-e', key + '=' + value]
    command += ['glm52-single', m.PYTHON, str(Path(__file__).resolve()), 'native', host]
    subprocess.run(command, check=True, timeout=30)
    m.write('launch_%s.json' % host, dict(command=command, launch_exit=0, readiness='unknown'))
    return dict(host=host, launch_exit=0)


def call(host, action, timeout=160):
    argv = ['/usr/bin/python3', str(Path(__file__).resolve()), action, host]
    if host == '167':
        argv = ['ssh', '-o', 'BatchMode=yes', 'root@172.16.10.167'] + argv
    p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    assert p.returncode == 0, '%s/%s: %s' % (host, action, p.stderr[-5000:])
    return json.loads(p.stdout)


def counters():
    status, raw = fetch('http://172.16.10.167:9900/metrics')
    assert status == 200
    return {line.rsplit(' ', 1)[0]: float(line.rsplit(' ', 1)[1])
            for line in raw.decode().splitlines() if not line.startswith('#') and 'external_prefix_cache' in line}


def decode_event(raw):
    if raw == '[DONE]':
        return None
    value = json.loads(raw)
    if 'error' in value:
        raise RuntimeError('in-band stream error: ' + raw)
    return value


def fixture(label):
    prompt = '请阅读以下动态请求并简短回答。\n' + ('推理框架需要保持请求语义和有效输出。 ' * 192) + '\n问题：P和D分别负责什么？'
    body = dict(model='glm-53', messages=[dict(role='user', content=prompt)],
                max_tokens=8, min_tokens=8, temperature=0, ignore_eos=True,
                return_token_ids=True, cache_salt=PLAN['run_id'] + '-' + label)
    helper = dict(body, stream=False, max_tokens=1, min_tokens=1,
                  kv_transfer_params=dict(do_remote_decode=True, do_remote_prefill=False))
    m.write(label + '_request.json', body)
    before = counters()
    started = time.monotonic_ns()
    ps, raw = fetch('http://172.16.10.166:9081/v1/chat/completions', helper, timeout=120)
    p_end = time.monotonic_ns()
    (ROOT / (label + '_P.raw')).write_bytes(raw)
    value = json.loads(raw)
    assert ps == 200 and value['usage']['completion_tokens'] == 1, value
    kv = value['kv_transfer_params']
    assert kv['do_remote_prefill'] and kv['remote_host'] == '172.16.10.166'
    assert kv['remote_engine_id'].startswith(PLAN['engine_ids']['166'] + '-')
    assert kv['remote_port'] == 36000 and kv['remote_dcp_size'] == 16 and kv['remote_pcp_size'] == 1
    assert kv['remote_block_ids'] and all(kv['remote_block_ids'])
    db = dict(body, stream=True, stream_options=dict(include_usage=True), kv_transfer_params=kv)
    request = urllib.request.Request('http://172.16.10.167:9900/v1/chat/completions',
              data=json.dumps(db, ensure_ascii=False).encode(), headers={'content-type': 'application/json'})
    events = []
    # Preserve every received SSE line and its host-monotonic arrival time.
    # D dispatch follows P export immediately; validation/export overhead is
    # inside the measured PD client interval, with no user pause between roles.
    with OPENER.open(request, timeout=120) as response, (ROOT / (label + '_D.sse')).open('xb') as stream:
        assert response.status == 200
        for line in response:
            arrived = time.monotonic_ns()
            stream.write(line)
            if line.startswith(b'data: '):
                event = decode_event(line[6:].decode().strip())
                events.append(dict(arrived_ns=arrived, value=event))
    ended = time.monotonic_ns()
    m.write(label + '_events.json', events)
    chunks = [e for e in events if e['value'] is not None]
    texts = [e for e in chunks if any(any(c.get('delta', {}).get(k) for k in ('content', 'reasoning', 'reasoning_content')) for c in e['value'].get('choices', []))]
    usages = [e['value']['usage'] for e in chunks if e['value'].get('usage')]
    assert usages and usages[-1]['completion_tokens'] == 8 and texts
    assert usages[-1]['prompt_tokens'] == value['usage']['prompt_tokens']
    assert events[-1]['value'] is None, 'missing DONE'
    assert any(c.get('finish_reason') == 'length' for e in chunks for c in e['value'].get('choices', []))
    after = counters()
    delta = {k: v-before.get(k, 0) for k, v in after.items()}
    assert any('hits' in k and v == value['usage']['prompt_tokens'] for k, v in delta.items()), delta
    first = texts[0]['arrived_ns']
    last = texts[-1]['arrived_ns']
    result = dict(label=label, prompt_tokens=value['usage']['prompt_tokens'], output_tokens=8,
                  helper_internal_tokens=1, P_client_wall_s=(p_end-started)/1e9,
                  PD_client_ttft_s=(first-started)/1e9, D_dispatch_to_first_s=(first-p_end)/1e9,
                  PD_client_wall_s=(ended-started)/1e9, stream_text_chunks=len(texts),
                  client_text_span_s=(last-first)/1e9, external_KV_delta=delta,
                  token_TPOT_ms=None, token_TPOT_reason='MTP may emit multiple tokens in one SSE chunk; audit token IDs before token latency attribution',
                  semantic_final_answer_accepted=False, formal_SLA_accepted=False)
    m.write(label + '_result.json', result)
    return result


def profile_control(role, action):
    host, port = {'P': ('166', 9081), 'D': ('167', 9900)}[role]
    started = time.monotonic_ns()
    status, raw = fetch('http://172.16.10.%s:%s/%s_profile' % (host, port, action), {}, timeout=300)
    m.write('%s_%s_profile.json' % (role, action), dict(status=status, body=raw.decode(errors='replace'), client_wall_s=(time.monotonic_ns()-started)/1e9))
    assert status == 200, (role, action, status, raw)


def workflow():
    launched = fitted = False
    active = []
    try:
        checks = {h: call(h, 'guard') for h in ('166', '167')}
        m.write('before_restart.json', {h: {k: v for k, v in row.items() if k != 'npu_raw'} for h, row in checks.items()})
        for h in ('166', '167'):
            call(h, 'retire')
        for h in ('166', '167'):
            call(h, 'snapshot')
        launched = True
        for h in ('166', '167'):
            call(h, 'launch')
        deadline = time.monotonic() + 1800
        while True:
            rows = {h: call(h, 'poll') for h in ('166', '167')}
            m.write('readiness.json', rows)
            if any(row['failure'] for row in rows.values()):
                raise RuntimeError('initialization failure; inspect actual native logs')
            if all(row['ready'] for row in rows.values()):
                fitted = True
                break
            if time.monotonic() > deadline:
                raise TimeoutError('native readiness deadline')
            time.sleep(10)
        for h in ('166', '167'):
            status, raw = fetch('http://172.16.10.%s:%s/openapi.json' % (h, PLAN['ports'][h]))
            assert status == 200 and {'/start_profile', '/stop_profile'} <= set(json.loads(raw)['paths'])
        warm = fixture('warmoff')
        off = fixture('measureoff')
        for role in ('P', 'D'):
            active.append(role)  # stop also when partial start returns an error
            profile_control(role, 'start')
        on = fixture('profileon')
        for role in reversed(active[:]):
            profile_control(role, 'stop')
            active.remove(role)
        for h in ('166', '167'):
            m.assert_idle(PLAN['ports'][h]) if h == '166' else call(h, 'posthealth')
        m.write('final_status.json', dict(status='completed', warmup=warm, profiler_off=off, profiler_on=on,
             trace_coverage='pending actual all-rank device trace audit', full_API_contract=False, SLA_accepted=False, performance_gain=None))
    except BaseException as error:
        m.write('failure.json', dict(error=type(error).__name__ + ': ' + str(error), fitted=fitted))
        raise
    finally:
        errors = []
        for role in reversed(active):
            try:
                profile_control(role, 'stop')
            except Exception as error:
                errors.append(dict(role=role, error=str(error)))
        m.write('profile_cleanup.json', dict(unresolved=errors, models_retained=fitted))
        if launched and not fitted:
            for h in ('166', '167'):
                call(h, 'cleanup')


def posthealth(host):
    m.assert_idle(PLAN['ports'][host])
    status, _ = fetch('http://127.0.0.1:%s/health' % PLAN['ports'][host])
    assert status == 200
    return dict(host=host, health=status, idle=True)


if __name__ == '__main__':
    action = sys.argv[1]
    if action == 'workflow':
        workflow()
    else:
        host = sys.argv[2]
        functions = dict(guard=guard, retire=retire, launch=launch, snapshot=lambda h: m.snapshot(h, PLAN),
                         native=lambda h: m.native(h, PLAN), poll=lambda h: m.poll(h, PLAN),
                         cleanup=lambda h: m.cleanup(h, PLAN), posthealth=posthealth)
        value = functions[action](host)
        if value is not None:
            print(json.dumps(value))
