"""Render the existing owned Router launch for a verified wide P/D pair."""
import argparse
import ast
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import re


def read(path):
    raw = path.read_bytes()
    return json.loads(raw), sha256(raw).hexdigest()


def render(plan, configuration, prefill, decode, guard_source, template, now):
    epoch = plan['epoch']
    phase = plan['workload']
    assert epoch > 1317 and configuration['epoch'] == epoch
    assert configuration['context_length'] == phase['width_tokens']
    assert configuration['max_concurrent_requests'] == 16 and configuration['queue_size'] == 100
    assert configuration['retries_enabled'] is False and configuration['health_check_endpoint'] == '/ready'
    router_timeout = configuration['request_timeout_seconds']
    assert router_timeout == phase['requested_router_request_timeout_seconds']
    assert router_timeout >= phase['warmup_phase_seconds']
    wait = phase['requested_D_wait_seconds']
    assert wait >= phase['warmup_phase_seconds']
    assert decode['KV_waiting_timeout_actual_by_rank'] == {str(i): wait for i in range(8)}
    assert decode['graph_capture_all8_verified']

    for role, current in (('p', prefill), ('d', decode)):
        receipt = current['handle']
        assert -60 <= (now - datetime.fromisoformat(current['at'])).total_seconds() < 600
        assert current['ready'] == 200 and current['all8_ownership_and_FDs_verified']
        assert receipt['epoch'] == epoch and receipt['role'] == role
        assert current['state']['generation'] == epoch
        assert current['state']['lifetime_seconds'] == phase['minimum_P_D_lifetime_seconds']
        assert receipt['pid'] == current['state']['pid']
        assert all(readings and all(value == 0 for value in readings) for readings in current['idle'].values())

    ppid = prefill['handle']['pid']
    dpid = decode['handle']['pid']
    guard = prefill['actual_workload_guard']
    gpid = guard['pid']
    assert prefill['guard_FD_verified'] and guard['phase'] == 'holding'
    assert all(type(pid) is int and pid > 1 for pid in (ppid, dpid, gpid))
    assert gpid == prefill['guard_PID'] == decode['handle']['guard_PID']
    assert re.fullmatch(r'workload_guard_g[0-9]+\.py', guard_source)

    decode_proof = {key: decode[key] for key in (
        'at', 'ready', 'graph_capture_all8_verified', 'all8_ownership_and_FDs_verified',
        'KV_waiting_timeout_actual_by_rank', 'idle', 'container_ID')}
    decode_proof['handle'] = {key: decode['handle'][key] for key in ('epoch', 'pid', 'role', 'guard_PID')}
    changelist = {
        "D = __DECODE_PROOF__": "D = " + repr(decode_proof),
        'sg_pd_g1317': f'sg_pd_g{epoch}',
        "['epoch'] == 1317": f"['epoch'] == {epoch}",
        "['epoch']==1317": f"['epoch']=={epoch}",
        'pid == 2302205': f'pid == {ppid}',
        "['pid']==2967504": f"['pid']=={dpid}",
        '2227468': str(gpid),
        '{str(i):14400 for i in range(8)}': f'{{str(i):{wait} for i in range(8)}}',
        "'--request-timeout-secs','14400'": f"'--request-timeout-secs','{router_timeout}'",
        "'configured_request_deadline_seconds':14400": f"'configured_request_deadline_seconds':{router_timeout}",
        'workload_guard_g1120.py': guard_source,
        'G1317_SG_ROUTER_': f'G{epoch}_SG_ROUTER_',
    }
    # Render the original template first, then attach the hashed proof as data.
    proof_pattern = "D = __DECODE_PROOF__"
    proof_text = changelist.pop(proof_pattern)
    assert template.count(proof_pattern) == 1
    result = template
    for prior, replacement in changelist.items():
        assert prior in result, prior
        result = result.replace(prior, replacement)
    result = result.replace(proof_pattern, proof_text)
    assert '__DECODE_PROOF__' not in result
    ast.parse(result.split("<<'PY'\n",1)[1].rsplit('\nPY',1)[0])
    return result


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ('bundle', 'prefill', 'decode', 'template', 'output'):
        cli.add_argument('--' + name, type=Path, required=True)
    cli.add_argument('--guard-source', required=True)
    args = cli.parse_args()
    plan, _ = read(args.bundle / 'preparation_manifest.json')
    configuration, digest = read(args.bundle / 'router_window.json')
    assert digest == plan['router_window_sha256']
    p, _ = read(args.prefill)
    d, _ = read(args.decode)
    template = args.template.read_text()
    script = render(plan, configuration, p, d, args.guard_source, template, datetime.now(timezone.utc))
    with args.output.open('x') as stream:
        stream.write(script)


if __name__ == '__main__':
    main()
