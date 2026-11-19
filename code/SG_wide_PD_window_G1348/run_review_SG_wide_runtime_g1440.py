"""Run or review the fixed three-request P/D test for one wide campaign."""

import argparse
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
import os
from pathlib import Path

from observe_SG_wide_role_g1439 import PLAN_BY_WIDTH, cmdline, load_plan, observe, read


SOURCE_SHA = 'ecfe6de1bad9da27176af6fa8354599a21bc04578c0fed56c92fe47773c6b22b'
QUERY_POLICY_SHA = '35a9ff6a87e299a380104162c49e29cbe1b7b9bcd04b592080a630a17f95771c'
RANKS = set(range(8))
MINIMUM_REMAINING_SECONDS = 905
CONTROL = 'PD_runtime_gate_control_wide'
RUNTIME = 'PD_runtime_gate_wide'


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def utc():
    return datetime.now(timezone.utc)


def write(path, value):
    with path.open('x') as file:
        json.dump(value, file, indent=2)
        file.write('\n')


def bind_router(root, coord, width, decode):
    epoch, _ = PLAN_BY_WIDTH[width]
    run = root / f'sg_pd_g{epoch}'
    plan, _ = load_plan(run / 'wide_workload_plan.json', width)
    prefill = observe(root, coord, width, 'p', True)
    assert prefill['ready'] == decode['ready'] == 200
    assert decode['all8_ownership_and_FDs_verified'] and decode['graph_capture_all8_verified']
    assert decode['handle']['epoch'] == decode['state']['generation'] == epoch and decode['handle']['role'] == 'd'
    router = prefill['actual_router']
    assert router['D_controller_PID'] == decode['handle']['pid'] == decode['state']['pid']
    assert router['D_container_ID'] == decode['container_ID'] == decode['state']['container_id']
    assert router['guard_PID'] == decode['handle']['guard_PID']
    wait = plan['workload']['requested_D_wait_seconds']
    assert decode['KV_waiting_timeout_actual_by_rank'] == {str(rank): wait for rank in RANKS}
    assert decode['idle'] and all(values and all(value == 0 for value in values) for values in decode['idle'].values())
    return prefill, router


def check_remaining(prefill, decode, now):
    assert -60 <= (now-datetime.fromisoformat(decode['at'])).total_seconds() < 600
    guard = datetime.fromisoformat(prefill['actual_workload_guard']['deadline_UTC'])
    roles = [datetime.fromisoformat(role['handle']['at']) + timedelta(seconds=role['state']['lifetime_seconds'])
             for role in (prefill, decode)]
    assert (min(guard, *roles)-now).total_seconds() > MINIMUM_REMAINING_SECONDS


def runtime_module(source, root, run, verify):
    raw = source.read_bytes()
    assert sha256(raw).hexdigest() == SOURCE_SHA
    namespace = {'__name__': 'wide_runtime'}
    exec(compile(raw, str(source), 'exec'), namespace)
    namespace.update(ROOT=root, RUN=run, OUT=run/RUNTIME, verify_epoch=verify)
    return namespace


def check_workers(runtime):
    workers = json.loads(runtime['get'](runtime['ROUTER'] + '/workers'))['workers']
    assert len(workers) == 2
    healthy = {(w['worker_type'], w['url']) for w in workers
               if w['is_healthy'] and w['load'] == 0 and w['model_id'] == runtime['MODEL']}
    assert healthy == set(runtime['BASE'].items())


def review_result(worker):
    launch = read(worker / 'launch_state.json')
    result_path = worker / 'result.json'
    result = read(result_path)
    assert result['phase'] == 'complete_requires_parent_review' and result['math1'] == '91' and result['completed'] == 3
    for field in ('transfer_count_delta_by_P_rank', 'transfer_mb_delta_by_P_rank', 'D_full_graph_counter_delta_by_rank'):
        assert set(map(int, result[field])) == RANKS and min(result[field].values()) > 0
    expected = {'launch_state.json', 'before_prefill.prom', 'before_decode.prom', 'math_request.json',
                'math_response.json', 'completion_request.json', 'fresh_response.sse', 'replay_response.sse',
                'after_prefill.prom', 'after_decode.prom'}
    assert set(result['file_SHA256']) == expected
    assert all(digest(worker / name) == checksum for name, checksum in result['file_SHA256'].items())
    return launch, result, digest(result_path)


def review_query_policy(run):
    rows = [read(path) for path in run.glob('Q_head_norm_policy.*.rank*.json')]
    assert len(rows) == 8 and {row['rank'] for row in rows} == RANKS
    assert all(row['source_SHA256'] == QUERY_POLICY_SHA and row['is_dsv41']
               and not row['configured_q_head_norm'] and row['actual_action'] == 'skip_per_head_RMSNorm' for row in rows)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('task-root', 'coord', 'decode', 'fixed-runtime-source'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--decode-sha256', required=True)
    parser.add_argument('--width', required=True, type=int, choices=PLAN_BY_WIDTH)
    parser.add_argument('action', choices=('check', 'run', 'review'))
    args = parser.parse_args()
    root, coord = (path.resolve(strict=True) for path in (args.task_root, args.coord))
    epoch, _ = PLAN_BY_WIDTH[args.width]
    run = root / f'sg_pd_g{epoch}'
    decode_path = args.decode.resolve(strict=True)
    assert decode_path.is_relative_to(root) and digest(decode_path) == args.decode_sha256
    decode = read(decode_path)
    prefill, router = bind_router(root, coord, args.width, decode)
    control = run / CONTROL

    def verify():
        _, current = bind_router(root, coord, args.width, decode)
        assert current == router
        check_workers(runtime)
        return current

    runtime = runtime_module(args.fixed_runtime_source, root, run, verify)
    check_workers(runtime)
    if args.action in ('check', 'run'):
        check_remaining(prefill, decode, utc())
        assert not control.exists() and not (run / RUNTIME).exists()
        if args.action == 'check':
            print(json.dumps({'verified_epoch':epoch, 'ready_for_three_request_test':True, 'submitted':False}))
            return
        control.mkdir(mode=0o700)
        handle = {'at':utc().isoformat(), 'outer_PID':os.getpid(), 'router':router,
                  'source_sha256':SOURCE_SHA, 'decode_receipt_sha256':args.decode_sha256}
        write(control / 'actual_launch_handle.json', handle)
        try:
            runtime['main']()
        except BaseException:
            write(control / 'terminal.json', {'finished_at':utc().isoformat(), 'passed':False})
            raise
        write(control / 'terminal.json', {'finished_at':utc().isoformat(), 'passed':True})
        return

    handle = read(control / 'actual_launch_handle.json')
    assert read(control / 'terminal.json')['passed'] is True
    assert handle['router'] == router and handle['source_sha256'] == SOURCE_SHA
    try:
        owner = cmdline(handle['outer_PID'])
    except FileNotFoundError:
        owner = []
    assert not any(Path(arg.decode(errors='ignore')).name == Path(__file__).name for arg in owner)
    worker_handle, result, result_sha = review_result(run / RUNTIME)
    assert worker_handle['router'] == router and worker_handle['pid'] == handle['outer_PID']
    print(json.dumps({'at':utc().isoformat(), 'terminal_verified':True, 'raw_files_verified':True,
                      'outer_live':False, 'inner_live':False, 'handle':handle,
                      'result':result, 'result_sha256':result_sha, 'fresh_live_guard':prefill['actual_workload_guard'],
                      'P_all8_runtime_query_policy':review_query_policy(run)}, indent=2))


if __name__ == '__main__':
    main()
