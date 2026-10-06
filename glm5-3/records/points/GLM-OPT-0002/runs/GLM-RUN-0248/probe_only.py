"""One direct PD probe of already ready Run246 roles; never restarts models."""
import json
import subprocess
import sys
import time
from pathlib import Path
import pd_functional_run as m

ROOT = Path(__file__).resolve().parent


def metrics():
    status, raw = m.fetch('http://172.16.10.167:9900/metrics')
    assert status == 200
    rows = {}
    for line in raw.decode().splitlines():
        if not line.startswith('#') and 'external_prefix_cache' in line:
            key, value = line.rsplit(' ', 1)
            rows[key] = float(value)
    return rows


def continue_decode(plan):
    raw = (ROOT / 'helper_export.raw').read_bytes()
    assert m.hashlib.sha256(raw).hexdigest() == plan['export_sha256']
    helper = json.loads(raw)
    kv = helper['kv_transfer_params']
    assert kv['remote_engine_id'] == plan['expected_P_engine_id']
    assert kv['remote_host'] == '172.16.10.166' and kv['remote_port'] == 36000
    assert kv['remote_dcp_size'] == 16 and kv['remote_pcp_size'] == 1
    assert kv['remote_block_ids'] and all(kv['remote_block_ids'])
    p_log = ROOT.parent / 'GLM-RUN-0246/native_166.log'
    assert plan['expected_P_engine_id'] in p_log.read_text(), 'runtime identity absent from actual native P log'
    prompt = '请阅读以下动态请求并简短回答。\n' + ('推理框架需要保持请求语义和有效输出。 ' * 192) + '\n问题：P和D分别负责什么？'
    body = dict(model='glm-53', messages=[dict(role='user', content=prompt)], max_tokens=32,
                min_tokens=32, temperature=0, ignore_eos=True, return_token_ids=True,
                cache_salt=plan['cache_salt'], stream=False, kv_transfer_params=kv)
    m.write('decoder_request.json', body)
    before = metrics()
    m.write('D_metrics_before.json', before)
    started = time.monotonic_ns()
    status, decoded_raw = m.fetch('http://172.16.10.167:9900/v1/chat/completions', body, timeout=300)
    (ROOT / 'decoder.raw').write_bytes(decoded_raw)
    after = metrics()
    m.write('D_metrics_after.json', after)
    value = json.loads(decoded_raw)
    assert status == 200, value
    assert value['usage']['completion_tokens'] == 32
    assert value['usage']['prompt_tokens'] == helper['usage']['prompt_tokens']
    choice = value['choices'][0]
    assert choice['finish_reason'] != 'error' and len(choice.get('token_ids') or []) == 32
    assert any(choice['message'].get(k) for k in ['content', 'reasoning', 'reasoning_content'])
    delta = {k: v-before.get(k, 0) for k, v in after.items()}
    m.write('D_external_metrics_delta.json', delta)
    assert any('hits' in k and v > 0 for k, v in delta.items()), 'external native KV hit not established'
    result = dict(P_prompt_tokens=helper['usage']['prompt_tokens'], D_prompt_tokens=value['usage']['prompt_tokens'],
                  internal_P_tokens=1, effective_D_tokens=32, D_status=status,
                  external_KV_metrics_delta=delta, cached_tokens_body=(value['usage'].get('prompt_tokens_details') or {}).get('cached_tokens'),
                  D_wall_s=(time.monotonic_ns()-started)/1e9, reused_P_export='GLM-RUN-0247',
                  native_PD_valid=True, full_API_contract=False, performance_gain=None)
    m.write('functional_result.json', result)
    return result


def guard(host, plan):
    root = plan['reuse_roles'][host]
    assert m.live(root), 'resident role identity changed'
    owners, raw = m.device_owners()
    members = m.tree(root['pid'], m.catalogue())
    assert len(owners) == 16 and owners.issubset({p['pid'] for p in members})
    if host == '166':
        owner = json.loads(Path('/data/tiankuan/wio/glm52-pd/controller-owner.json').read_text())
        assert owner['status'] == 'running' and owner['run_id'] == plan['run_id']
        assert m.live(owner['owner'])
    spec = json.loads((ROOT / 'controller_spec.json').read_text())
    for source in spec['stages'][0]['sources']:
        assert m.hashlib.sha256(Path(source['path']).read_bytes()).hexdigest() == source['sha256']
    port = plan['ports'][host]
    status, _ = m.fetch('http://127.0.0.1:%d/health' % port)
    status2, models = m.fetch('http://127.0.0.1:%d/v1/models' % port)
    assert status == status2 == 200
    assert any(x['id'] == 'glm-53' for x in json.loads(models)['data'])
    m.assert_idle(port)
    m.write('guard_%s.json' % host, dict(root=root, members=members, npu_owners=sorted(owners),
                                      npu_raw=raw, idle=True, served_model='glm-53'))
    return dict(host=host, live_identity=True, npu_owners_count=len(owners), idle=True)


def call(host, plan):
    argv = ['/usr/bin/python3', str(Path(__file__).resolve()), 'guard', host]
    if host == '167':
        argv = ['ssh', '-o', 'BatchMode=yes', 'root@172.16.10.167'] + argv
    p = subprocess.run(argv, capture_output=True, text=True, timeout=30)
    assert p.returncode == 0, p.stderr[-5000:]
    return json.loads(p.stdout)


def workflow(plan):
    started = time.monotonic_ns()
    try:
        checks = {h: call(h, plan) for h in ['166', '167']}
        m.write('before_probe.json', checks)
        result = continue_decode(plan)
        m.write('final_status.json', dict(status='completed', result=result,
            borrowed_roles='Run246 P166/D167 retained', model_restarts=0,
            public8000='no replacement;full API pending', performance_gain=None,
            wall_s=(time.monotonic_ns()-started)/1e9))
        print(json.dumps(result), flush=True)
    except BaseException as error:
        m.write('final_status.json', dict(status='failed', error=type(error).__name__+': '+str(error),
            borrowed_roles='Run246 roles retained;read health/native logs before next action',
            model_restarts=0, performance_gain=None, Current=None))
        raise


if __name__ == '__main__':
    plan = json.loads((ROOT / 'functional_plan.json').read_text())
    if sys.argv[1] == 'workflow':
        workflow(plan)
    else:
        print(json.dumps(guard(sys.argv[2], plan)))
