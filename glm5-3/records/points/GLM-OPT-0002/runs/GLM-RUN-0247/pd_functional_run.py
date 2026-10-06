"""Native PD probe client only; no model launch, restart or cleanup."""
import hashlib
import json
import os
import re
import signal
import subprocess
import time
import urllib.request
import urllib.error
from pathlib import Path
ROOT = Path(__file__).resolve().parent


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
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(req, timeout=timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()


def assert_idle(port):
    status, raw = fetch('http://127.0.0.1:%d/metrics' % port)
    assert status == 200
    samples = [line for line in raw.decode().splitlines() if not line.startswith('#')
               and re.match(r'vllm:num_requests_(running|waiting)(\{|\s)', line)]
    assert samples and all(float(line.rsplit(' ', 1)[1]) == 0 for line in samples), samples


def signal_verified(p, sig):
    if live(p):
        os.kill(p['pid'], sig)


def probe(plan):
    prompt = '请阅读以下动态请求并简短回答。\n' + ('推理框架需要保持请求语义和有效输出。 ' * 192) + '\n问题：P和D分别负责什么？'
    body = dict(model='glm-53', messages=[dict(role='user', content=prompt)], max_tokens=32,
                min_tokens=32, temperature=0, ignore_eos=True, return_token_ids=True,
                cache_salt='glm53-functional-' + plan['run_id'])
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
