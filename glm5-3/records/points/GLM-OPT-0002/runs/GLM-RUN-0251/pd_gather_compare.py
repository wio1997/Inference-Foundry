"""Bounded same-resident-process A/B under the existing unique controller.

Only one D restart; stock/candidate switch at globally idle request boundaries.
No new profile, tuning scan, kernel change, or large workload. The shim is a
diagnostic comparison mechanism; it is not a product performance deployment.
"""
import hashlib
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
    raw = None if body is None else json.dumps(body, ensure_ascii=False).encode()
    req = urllib.request.Request(url, data=raw, headers={'content-type': 'application/json'})
    try:
        with OPENER.open(req, timeout=timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()


m.fetch = fetch


def rpc(host, action, *args):
    argv = ['/usr/bin/python3', str(Path(__file__).resolve()), action, host, *map(str, args)]
    if host == '167':
        argv = ['ssh', '-o', 'BatchMode=yes', 'root@172.16.10.167', *argv]
    p = subprocess.run(argv, capture_output=True, text=True, timeout=300 if action == 'micro' else 160)
    assert p.returncode == 0, (host, action, p.stderr[-6000:])
    return json.loads(p.stdout)


def source_digest():
    code = 'import pathlib,hashlib;print(hashlib.sha256(pathlib.Path(%r).read_bytes()).hexdigest())' % PLAN['target_source']
    return subprocess.check_output(['docker', 'exec', 'glm52-single', m.PYTHON, '-c', code], text=True).strip()


def guard(host, fresh=False):
    root = m.new_owner(host) if fresh else PLAN['resident_roles'][host]
    assert root and m.live(root), 'resident identity changed'
    members = m.tree(root['pid'], m.catalogue())
    owners, raw = m.device_owners()
    assert len(owners) == 16 and owners.issubset({p['pid'] for p in members}), 'foreign device owner'
    m.assert_idle(PLAN['ports'][host])
    status, _ = fetch('http://127.0.0.1:%d/health' % PLAN['ports'][host])
    assert status == 200
    if host == '167':
        assert source_digest() == (PLAN['shim_sha256'] if fresh else PLAN['original_sha256'])
    return dict(root=root, members=members, device_owners=sorted(owners), idle=True)


def replace_source(kind):
    before = PLAN['original_sha256'] if kind == 'shim' else PLAN['shim_sha256']
    assert source_digest() == before, 'source identity changed; no overwrite'
    artifact = ROOT / ('prepare_finalize_shim.py' if kind == 'shim' else 'prepare_finalize_original.py')
    expected = PLAN['shim_sha256'] if kind == 'shim' else PLAN['original_sha256']
    assert hashlib.sha256(artifact.read_bytes()).hexdigest() == expected
    code = 'import pathlib,hashlib; p=pathlib.Path(%r); b=pathlib.Path(%r).read_bytes(); assert hashlib.sha256(p.read_bytes()).hexdigest()==%r; p.write_bytes(b); print(hashlib.sha256(p.read_bytes()).hexdigest())' % (PLAN['target_source'], str(artifact), before)
    got = subprocess.check_output(['docker', 'exec', 'glm52-single', m.PYTHON, '-c', code], text=True).strip()
    assert got == expected
    m.write('source_%s.json' % kind, dict(before=before, after=got, target=PLAN['target_source']))
    return dict(source=got)


def switch(mode):
    guard('167', fresh=True)
    assert mode in (0, 1, 2, 3)
    fd = os.open(ROOT / 'gather_mode.bin', os.O_WRONLY)
    try:
        assert os.pwrite(fd, bytes([mode]), 0) == 1
        os.fsync(fd)
    finally:
        os.close(fd)
    return dict(mode=mode, idle=True)


def launch(host):
    assert host == '167' and not m.device_owners()[0]
    env = json.loads((ROOT / PLAN['baseline_env'][host]).read_text())['environment']
    env.update(PLAN['environment'][host])
    env['PYTHONPATH'] = ':'.join(p for p in env.get('PYTHONPATH', '').split(':') if '/glm52-pd/deploy/plugins/' not in p)
    command = ['docker', 'exec', '-d']
    for key, value in sorted(env.items()): command += ['-e', key + '=' + value]
    command += ['glm52-single', m.PYTHON, str(Path(__file__).resolve()), 'native', host]
    subprocess.run(command, check=True, timeout=30)
    m.write('launch_167.json', dict(command=command, launch_exit=0, readiness='unknown'))
    return dict(launch_exit=0)


def counters():
    status, raw = fetch('http://172.16.10.167:9900/metrics')
    assert status == 200
    return {line.rsplit(' ', 1)[0]: float(line.rsplit(' ', 1)[1])
            for line in raw.decode().splitlines() if not line.startswith('#') and 'external_prefix_cache' in line}


def request(label, complete=False):
    if complete:
        messages = [dict(role='system', content='直接给出答案。若需要思考，思考应简短。'),
                    dict(role='user', content='1+1等于几？最终答案只写数字。')]
        body = dict(model='glm-53', messages=messages, max_tokens=96, temperature=0, return_token_ids=True, thinking_token_budget=0)
    else:
        prompt = '请阅读以下动态请求并简短回答。\n' + ('推理框架需要保持请求语义和有效输出。 ' * 192) + '\n问题：P和D分别负责什么？'
        body = dict(model='glm-53', messages=[dict(role='user', content=prompt)],
                    max_tokens=8, min_tokens=8, temperature=0, ignore_eos=True, return_token_ids=True)
    body['cache_salt'] = PLAN['run_id'] + '-' + label
    m.write(label + '_request.json', body)
    helper = dict(body, stream=False, max_tokens=1, min_tokens=1,
                  kv_transfer_params=dict(do_remote_decode=True, do_remote_prefill=False))
    before = counters(); started = time.monotonic_ns()
    ps, raw = fetch('http://172.16.10.166:9081/v1/chat/completions', helper, timeout=120)
    p_end = time.monotonic_ns(); (ROOT / (label + '_P.raw')).write_bytes(raw)
    value = json.loads(raw)
    assert ps == 200 and value['usage']['completion_tokens'] == 1, value
    kv = value['kv_transfer_params']
    assert kv['do_remote_prefill'] and kv['remote_host'] == '172.16.10.166'
    assert kv['remote_engine_id'].startswith('glm53-P-run249-') and kv['remote_port'] == 36000
    assert kv['remote_dcp_size'] == 16 and kv['remote_pcp_size'] == 1 and all(kv['remote_block_ids'])
    req = urllib.request.Request('http://172.16.10.167:9900/v1/chat/completions',
          data=json.dumps(dict(body, stream=True, stream_options=dict(include_usage=True), kv_transfer_params=kv), ensure_ascii=False).encode(),
          headers={'content-type': 'application/json'})
    events = []
    with OPENER.open(req, timeout=180) as response, (ROOT / (label + '_D.sse')).open('xb') as stream:
        assert response.status == 200
        for line in response:
            arrived = time.monotonic_ns(); stream.write(line)
            if line.startswith(b'data: '):
                text = line[6:].decode().strip(); event = None if text == '[DONE]' else json.loads(text)
                events.append(dict(arrived_ns=arrived, value=event))
                if event and 'error' in event:
                    m.write(label + '_events.json', events)
                    raise RuntimeError('in-band SSE error: ' + text)
    ended = time.monotonic_ns(); m.write(label + '_events.json', events)
    assert events and events[-1]['value'] is None, 'missing DONE'
    usages = [e['value']['usage'] for e in events if e['value'] and e['value'].get('usage')]
    assert usages and usages[-1]['prompt_tokens'] == value['usage']['prompt_tokens']
    ids = []; arrivals = []; content = ''; reasons = []
    for e in events:
        for c in (e['value'] or {}).get('choices', []):
            ts = c.get('token_ids') or c.get('delta', {}).get('token_ids') or []
            if ts:
                ids.extend(ts); arrivals.append(dict(arrived_ns=e['arrived_ns'], count=len(ts)))
            content += c.get('delta', {}).get('content') or ''
            if c.get('finish_reason'): reasons.append(c['finish_reason'])
    n = usages[-1]['completion_tokens']
    assert len(ids) == n and arrivals and reasons
    after = counters(); delta = {k: v-before.get(k, 0) for k, v in after.items()}
    assert any('hits' in k and v == value['usage']['prompt_tokens'] for k, v in delta.items()), delta
    if not complete:
        assert n == 8 and ids == [785,1196,374,10156,264,3405,304,8452] and reasons[-1] == 'length'
    semantic = bool(complete and reasons[-1] == 'stop' and re.fullmatch(r'\s*2[。.!！]?\s*', content))
    result = dict(label=label, complete=complete, semantic_accepted=semantic, final_content=content,
                  token_ids=ids, chunks=[x['count'] for x in arrivals], completion_tokens=n,
                  prompt_tokens=value['usage']['prompt_tokens'], finish_reason=reasons[-1],
                  external_KV_delta=delta, P_wall_s=(p_end-started)/1e9,
                  PD_ttft_s=(arrivals[0]['arrived_ns']-started)/1e9,
                  D_first_s=(arrivals[0]['arrived_ns']-p_end)/1e9,
                  PD_wall_s=(ended-started)/1e9, D_wall_s=(ended-p_end)/1e9,
                  TPOT_ms=(arrivals[-1]['arrived_ns']-arrivals[0]['arrived_ns'])/1e6/(n-1) if n>1 else None,
                  formal_SLA_accepted=False, profiler_active=False)
    m.write(label + '_result.json', result); return result


def recovery_guard(host):
    if host == '166': return guard(host)
    root = PLAN['resident_roles'][host]
    assert m.live(root) and source_digest() == PLAN['original_sha256']
    members = m.tree(root['pid'],m.catalogue()); owners,raw=m.device_owners()
    assert len(owners)==16 and owners.issubset({p['pid'] for p in members})
    status,metrics=fetch('http://127.0.0.1:9900/metrics')
    assert status==200
    return dict(root=root,members=members,device_owners=sorted(owners),metrics=metrics.decode(),reason='Owned Run250 rank12 diagnostic assertion; no new requests accepted here')


def micro():
    assert not m.device_owners()[0] and source_digest()==PLAN['original_sha256']
    env=json.loads((ROOT/PLAN['baseline_env']['167']).read_text())['environment'];env.update(PLAN['environment']['167'])
    command=['docker','exec']
    for key,value in sorted(env.items()):command+=['-e',key+'='+value]
    command+=['glm52-single',m.PYTHON,str(ROOT/'check_finalize_hccl.py'),str(ROOT/'prepare_finalize_original.py'),str(ROOT/'prepare_finalize_candidate.py'),str(ROOT/'micro_checks'),str(ROOT/'namespace_owner_167.json')]
    with (ROOT/'micro.stdout.log').open('wb') as out,(ROOT/'micro.stderr.log').open('wb') as err:
        try: p=subprocess.run(command,stdout=out,stderr=err,timeout=240)
        except subprocess.TimeoutExpired:
            m.cleanup('167',PLAN)
            m.write('micro_execution.json',dict(exit_code=None,timed_out=True,passed=False));return dict(passed=False,timed_out=True)
    result=json.loads((ROOT/'micro_checks/result.json').read_text()) if (ROOT/'micro_checks/result.json').exists() else dict(passed=False)
    result['inner_exit_code']=p.returncode
    m.write('micro_execution.json',result)
    if m.device_owners()[0]: m.cleanup('167',PLAN)
    assert not m.device_owners()[0]
    return result


def workflow():
    owner=json.loads(Path('/data/tiankuan/wio/glm52-pd/controller-owner.json').read_text())
    assert owner['run_id']==PLAN['run_id'] and owner['status']=='running' and m.live(owner['owner'])
    mutated=False; launched=False; fitted=False; results=[]
    try:
        m.write('recovery_guards.json',{h:rpc(h,'recoveryguard') for h in ('166','167')})
        rpc('167','recoveryretire')
        probe=rpc('167','micro');m.write('micro_result.json',probe)
        if probe['passed']:rpc('167','install');mutated=True
        rpc('167','launch');launched=True
        deadline=time.monotonic()+1800
        while True:
            row=rpc('167','poll');m.write('readiness.json',row)
            if row['failure']:raise RuntimeError('D recovery initialization failed')
            if row['ready']:fitted=True;break
            if time.monotonic()>deadline:raise TimeoutError('D recovery readiness')
            time.sleep(10)
        if not probe['passed']:
            result=request('stock_recovery_warm');m.write('final_status.json',dict(status='stock_recovered_candidate_rejected',micro=probe,result=result));return
        for mode,name in enumerate(('A1','B1','A2','B2')):
            rpc('166','guard');rpc('167','switch',mode)
            request(name+'_warm')
            witness=rpc('167','witness',mode);m.write(name+'_witness.json',witness)
            assert len(witness)==16 and {x['rank'] for x in witness}==set(range(16))
            assert all(x['mode']==mode and (not mode%2 or (x['actual_hidden_bitwise_equal'] and x['input_bytes_equal'] and x['all_ranks_bytes_equal'])) for x in witness)
            for n in range(2):results.append(request(name+'_measure%d'%n))
            results.append(request(name+'_complete',complete=True));m.write('comparison_results.json',results)
        m.write('final_status.json',dict(status='completed',results=results,micro=probe,Current=None,full_API_contract=False,formal_SLA=False))
    except BaseException as error:
        m.write('failure.json',dict(error=repr(error),launched=launched,fitted=fitted,results=results));raise
    finally:
        restored={}
        if mutated:
            if fitted:
                try:restored['mode']=rpc('167','switch',0)
                except Exception as e:restored['mode_error']=repr(e)
            elif launched:
                try:restored['cleanup']=rpc('167','cleanup')
                except Exception as e:restored['cleanup_error']=repr(e)
            try:restored['source']=rpc('167','restore')
            except Exception as e:restored['source_error']=repr(e)
        m.write('restoration.json',restored)
        assert not any(k.endswith('_error') for k in restored),restored


def action(name, host, args):
    if name == 'recoveryguard': return recovery_guard(host)
    if name == 'micro': return micro()
    if name == 'recoveryretire':
        row=recovery_guard(host);m.stop_tree(row['members'],row['root']);assert not m.device_owners()[0];return dict(owned_failed_D_stopped=True)
    if name == 'guard': return guard(host)
    if name == 'retire':
        row = guard(host); m.stop_tree(row['members'], row['root']); assert not m.device_owners()[0]
        return dict(stopped=True)
    if name == 'install': return replace_source('shim')
    if name == 'restore': return replace_source('original')
    if name == 'launch': return launch(host)
    if name == 'native': return m.native(host, PLAN)
    if name == 'poll': return m.poll(host, PLAN)
    if name == 'cleanup': return m.cleanup(host, PLAN)
    if name == 'switch': return switch(int(args[0]))
    if name == 'witness': return [json.loads(f.read_text()) for f in sorted((ROOT/'witnesses').glob('mode%s_rank*.json' % args[0]))]
    raise ValueError(name)


if __name__ == '__main__':
    if sys.argv[1] == 'workflow': workflow()
    else: print(json.dumps(action(sys.argv[1], sys.argv[2], sys.argv[3:])))
