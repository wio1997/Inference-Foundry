"""Read actual role readiness for a staged wide campaign without generation."""

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import re
import socket
import subprocess
import urllib.error
import urllib.request

from plan_SG_wide_campaign_leases_g1424 import windows
from wide_initial_gate_context_g1434 import PLAN_BY_WIDTH, load_plan


IMAGE = 'sha256:7da963869ee54984dfeea2445befdf181710ef846448466db2361247759904ec'
GUARD_SOURCE_SHA = '291c7b4944ebbbc48ad5f2becf1b224e1072ca0957f075d1fa0707d25c819183'
QUEUES = {
    'p': ('num_prefill_bootstrap_queue_reqs', 'num_prefill_inflight_queue_reqs'),
    'd': ('num_decode_prealloc_queue_reqs', 'num_decode_transfer_queue_reqs'),
}
ROLES = {'p': 'prefill', 'd': 'decode'}


def read(path):
    return json.loads(path.read_bytes())


def cmdline(pid):
    return Path('/proc', str(pid), 'cmdline').read_bytes().split(b'\0')


def fds(pid):
    return {path.resolve() for path in Path('/proc', str(pid), 'fd').iterdir()}


def docker(*args):
    return subprocess.check_output(['docker', *args], text=True, timeout=12)


def get(port, path):
    with urllib.request.urlopen(f'http://127.0.0.1:{port}{path}', timeout=6) as response:
        assert response.status == 200
        return response.read()


def prove_guard(root, coord, plan, expected_sha, handle):
    state = read(root / f"workload_guard_g{plan['epoch']}" / 'state.json')
    width = plan['workload']['width_tokens']
    assert state['phase'] == 'holding' and state['epoch'] == plan['epoch'] and state['input_tokens'] == width
    assert state['campaign_id'] == f"sg_pd_g{plan['epoch']}_aisbench_{width // 1024}k"
    assert state['pair'] == 'P2203/D2204' and state['preparation_manifest_sha256'] == expected_sha
    assert state['bound_s'] == windows(plan['workload'])['pair_guard_lease_seconds']
    assert datetime.fromisoformat(state['deadline_UTC']) > datetime.now(timezone.utc)
    source = root / 'workload_guard_g1424.py'
    assert sha256(source.read_bytes()).hexdigest() == GUARD_SOURCE_SHA
    pid = state['pid']
    assert pid == handle['guard_PID'] and str(source).encode() in cmdline(pid)
    assert coord / 'deepseek-v41-p3d4-workload.lock' in fds(pid)
    return state


def verify_ledgers(run):
    for kind, expected in (('target', 1384), ('draft', 73)):
        rows = [read(path) for path in (run / 'weight_ledgers').glob(kind + '.*.json')]
        assert len(rows) == 8 and all(row['status'] == 'passed' and row['missing'] == []
                                      and row['parameter_count'] == expected for row in rows)


def verify_decode_graph(run):
    graph = {kind: {'begin': {}, 'end': {}} for kind in ('target', 'draft')}
    label = re.compile(r'TP(\d+) EP(\d+)\] Capture (target|draft) verify NPU graph (begin|end)\. (.*)')
    with (run / 'server.log').open(errors='replace') as stream:
        for line in stream:
            match = label.search(line)
            if match:
                tp, ep, kind, stage, detail = match.groups()
                assert tp == ep
                graph[kind][stage][int(tp)] = detail
    for kind, width in (('target', 6), ('draft', 5)):
        begin, end = graph[kind]['begin'], graph[kind]['end']
        assert set(begin) == set(end) == set(range(8))
        assert all(re.search(rf'backend=full, num_tokens_per_req={width}(?:,|$)', value) for value in begin.values())
        assert all(16 in [int(n) for n in re.search(r'bs=\[([^\]]+)\]', value)[1].split(',')] for value in end.values())
    assert len(list(run.glob('Q_head_norm_policy_install.pid*.json'))) == 8


def scheduler_waits(container, expected):
    target = f'SGLANG_DISAGGREGATION_WAITING_TIMEOUT={expected}'.encode()
    waits = {}
    worker_pids = set()
    for line in docker('top', container, '-eo', 'pid,args').splitlines()[1:]:
        rank = re.search(r'sglang::scheduler_TP(\d+)', line)
        if rank:
            assert rank[1] not in waits
            pid = line.split()[0]
            assert pid not in worker_pids
            worker_pids.add(pid)
            env = Path('/proc', pid, 'environ').read_bytes().split(b'\0')
            assert target in env
            waits[rank[1]] = expected
    assert set(waits) == {str(rank) for rank in range(8)}
    return waits


def observe(root, coord, width, role, with_router):
    epoch, _ = PLAN_BY_WIDTH[width]
    run = root / f'sg_pd_g{epoch}'
    plan, plan_sha = load_plan(run / 'wide_workload_plan.json', width)
    handle = read(run / 'actual_launch_handle.json')
    state = read(run / 'state.json')
    host = socket.gethostname()
    pid = handle['pid']
    assert handle['epoch'] == state['generation'] == epoch and handle['role'] == state['role'] == role
    assert state['pid'] == pid and state['host'] == host and state['task_id'] == root.name
    assert handle['image'] == state['image_id'] == IMAGE
    assert handle['source_manifest_SHA256'] == plan['roles'][role]['deployment_manifest_sha256']
    assert str(run / 'node_controller.py').encode() in cmdline(pid)
    report = {'at': datetime.now(timezone.utc).isoformat(), 'handle': handle, 'state': state,
              'actual_controller_live': True, 'ready': None, 'plan_sha256': plan_sha}
    if not state.get('container_id'):
        return report
    port = 19093 if role == 'p' else 19094
    try:
        get(port, '/ready')
    except (urllib.error.URLError, TimeoutError) as exc:
        return {**report, 'ready_error_type': type(exc).__name__}

    cid = state['container_id']
    container = json.loads(docker('inspect', cid))[0]
    labels = container['Config']['Labels']
    assert container['State']['Running'] and container['Id'] == cid and container['Image'] == IMAGE
    assert labels['codex.controller_pid'] == str(pid) and labels['codex.run'] == run.name
    assert labels['codex.task'] == root.name
    assert state['lifetime_seconds'] == windows(plan['workload'])['minimum_model_role_lifetime_seconds']
    held = fds(pid)
    for rank in range(8):
        resource = coord / f'{host}-npu{rank}'
        owner = read(resource.with_suffix('.owner.json'))
        assert owner['pid'] == pid and owner['state'] == 'claimed' and owner['run_directory'] == str(run)
        assert owner['task_id'] == root.name and resource.with_suffix('.lock') in held
    verify_ledgers(run)
    result = json.loads(get(port, '/server_info'))
    options = result.get('server_args', result)
    assert (options['tp_size'], options['ep_size'], options['pp_size']) == (8, 8, 1)
    assert (options['disaggregation_mode'], options['disaggregation_transfer_backend']) == (ROLES[role], 'ascend')
    assert options['speculative_algorithm'] == 'DSPARK' and options['speculative_num_draft_tokens'] == 6
    assert options['context_length'] == 263296 and options['max_running_requests'] == 16
    assert not options['disable_overlap_schedule'] and not options['disable_radix_cache']
    assert options['chunked_prefill_size'] == (8192 if role == 'p' else 1024)
    selected = ('tp_size', 'ep_size', 'pp_size', 'context_length', 'disaggregation_mode',
                'disaggregation_transfer_backend', 'speculative_algorithm', 'speculative_num_draft_tokens',
                'disable_radix_cache', 'disable_overlap_schedule', 'max_running_requests',
                'chunked_prefill_size', 'cuda_graph_config', 'mem_fraction_static')
    metrics = get(port, '/metrics').decode()
    idle = {key: [float(x) for x in re.findall(r'^sglang:' + key + r'(?:\{[^\n]*\})?\s+([\d.eE+-]+)', metrics, re.M)]
            for key in ('num_running_reqs', 'num_queue_reqs', *QUEUES[role])}
    assert all(values and all(value == 0 for value in values) for values in idle.values())
    report.update(ready=200, all8_ownership_and_FDs_verified=True, idle=idle,
                  server_args={key: options[key] for key in selected})
    if role == 'd':
        assert not with_router and options['disaggregation_decode_enable_radix_cache']
        report['server_args']['disaggregation_decode_enable_radix_cache'] = True
        graph = options['cuda_graph_config']['decode']
        assert graph['backend'] == 'full' and graph['max_bs'] == 16
        verify_decode_graph(run)
        report.update(container_ID=cid, graph_capture_all8_verified=True,
                      KV_waiting_timeout_actual_by_rank=scheduler_waits(cid, plan['workload']['requested_D_wait_seconds']))
    else:
        guard = prove_guard(root, coord, plan, plan_sha, handle)
        bootstrap = json.loads(get(18998, '/route?prefill_dp_rank=-1&prefill_cp_rank=-1&target_tp_rank=-1&target_pp_rank=-1'))
        assert bootstrap['attn_tp_size'] == 8 and bootstrap['kv_layout'] == 'v4,c1:v4,c2:v4'
        assert bootstrap['dsv41_spec_layout']['num_draft_tokens'] == 6
        report.update(container={key: container[key] for key in ('Id', 'Image', 'State')},
                      actual_workload_guard=guard, guard_FD_verified=True, guard_PID=guard['pid'],
                      PD_bootstrap=bootstrap)
        if with_router:
            router = read(run / 'router_handle.json')
            assert router['P_controller_PID'] == pid and router['P_container_ID'] == cid
            assert router['guard_PID'] == guard['pid']
            assert router['configured_request_deadline_seconds'] == plan['workload']['requested_router_request_timeout_seconds']
            router_pid = router['router_PID']
            assert b'sglang_router.launch_router' in cmdline(router_pid)
            assert coord / f'{host}-port18090.lock' in fds(router_pid)
            report['actual_router'] = router
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task-root', type=Path, required=True)
    parser.add_argument('--coord', type=Path, required=True)
    parser.add_argument('--width', type=int, choices=PLAN_BY_WIDTH, required=True)
    parser.add_argument('--role', choices=ROLES, required=True)
    parser.add_argument('--with-router', action='store_true')
    args = parser.parse_args()
    result = observe(args.task_root.resolve(strict=True), args.coord.resolve(strict=True),
                     args.width, args.role, args.with_router)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
