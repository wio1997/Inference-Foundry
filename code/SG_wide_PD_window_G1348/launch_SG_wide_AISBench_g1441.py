"""Launch an admitted wide AISBench stage through the fixed runner."""

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import re
import subprocess
import sys
import urllib.request

from observe_SG_wide_role_g1439 import QUEUES, ROLES
from wide_initial_gate_context_g1434 import PLAN_BY_WIDTH, load_plan


RUNNER_SHA = 'f6182f230048909ebfcc02bcb1ef35f4fe93272d16f1d31cdf8157f0c37fe1e8'
USAGE_SHA = 'df9026cdbb4228219051d97694eb95164009856cebc87858022ebc8dedf60fc4'
CPU_SOURCE_SHA = 'd2279822c9709f358f47cd1ba3533493b21f7b66aded62be125b7bf565f21a7a'
IMAGE = 'sha256:05358d58d23f2679c71e48d93797d5d2a28d72eedc7fb4900479c89955465d76'
CPU_ASSIGNMENTS = {
    65536: '127aeedd9a74e6b6932c20e71eadc4a7de14d776473c84b7a76f490d0e61ff28',
    131072: '7d5222942a7d3448754576c593716a8b029f874c3f729ab577d1c411bc297611',
    262144: '6f771b3b2dc0a8a8e9bd0ef9fec1bf7f9a11523ddf7378186a1f48b705ef2fdd',
}


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def now():
    return datetime.now(timezone.utc)


def stage_contract(gate, plan, plan_sha):
    workload, epoch = plan['workload'], plan['epoch']
    width, phase = workload['width_tokens'], gate['stage']
    assert gate['profile_epoch'] == epoch and gate['input_tokens'] == width
    assert gate['run_id'] == f'sg_pd_g{epoch}_aisbench_{width // 1024}k'
    assert gate['engine'] == 'sglang' and gate['profile_epoch'] == PLAN_BY_WIDTH[width][0]
    assert gate['formal_not_previously_submitted'] is True and gate['cold_mode'] == 'novel_prefixes'
    warm_seconds = workload['warmup_phase_seconds']
    if phase == 'formal':
        assert gate['warmup_only'] is False and gate['P_compiler_cache_verified_after_warmups']
        assert gate['completed_warmups'] == 5 and gate['next_warmup'] is None
        budget = gate['formal_wall_budget']
        assert budget['phase_seconds'] == {'cold':workload['cold_phase_seconds'], 'hot':workload['replay_phase_seconds']}
        step, complete = 'formal', 5
        required = workload['remaining_CPU_and_pair_lease_required_at_formal_submit_seconds']
    else:
        assert gate['warmup_only'] is True
        budget = gate['warmup_wall_budget']
        assert budget['phase_seconds'] == warm_seconds and budget['campaign_seconds'] == workload['warmup_campaign_seconds']
        if phase == 'initial':
            assert gate['warmup_pause_after'] == 3
            step, complete, count = 'initial', 0, 3
        else:
            assert phase == 'remaining' and gate['next_warmup'] in (4, 5)
            step, complete, count = f"warmup{gate['next_warmup']}", gate['next_warmup']-1, 1
            assert gate['completed_warmups'] == complete
        required = count * (warm_seconds+90) + 300
    assert budget['wide_preparation_sha256'] == plan_sha
    return step, complete, required, budget


def check_window(gate, budget, cpu, required, at):
    age = (at-datetime.fromisoformat(gate['validated_at'])).total_seconds()
    assert 0 <= age < 600
    deadlines = budget['resource_deadlines_UTC']
    assert set(deadlines) == {'prefill', 'decode', 'CPU_client', 'pair_workload_lock'}
    assert cpu['conservative_deadline_UTC'] == deadlines['CPU_client']
    assert budget['CPU_client_container_id'] == cpu['container_ID']
    assert min((datetime.fromisoformat(value)-at).total_seconds() for value in deadlines.values()) > required


def read_cpu(client, width):
    bundle = client / 'wide_CPU_full_campaign_g1424'
    script = bundle / 'cpu_client_campaign_window_g1424.py'
    assignment = bundle / f'{width // 1024}k' / 'assignment.json'
    assert digest(script) == CPU_SOURCE_SHA and digest(assignment) == CPU_ASSIGNMENTS[width]
    cmd = [sys.executable, '-B', str(script), 'report', '--assignment', str(assignment)]
    return json.loads(subprocess.check_output(cmd, cwd=client, text=True, timeout=20))


def check_idle(gate):
    for tag, role in ROLES.items():
        with urllib.request.urlopen(gate['roles'][role]['metrics_url'], timeout=12) as response:
            assert response.status == 200
            metrics = response.read().decode()
        for name in ('num_running_reqs', 'num_queue_reqs', *QUEUES[tag]):
            values = [float(value) for value in re.findall(
                r'^sglang:' + name + r'(?:\{[^\n]*\})?\s+([\d.eE+-]+)', metrics, re.M)]
            assert values and all(value == 0 for value in values), (role, name, values)


def runner_argv(root, runner, gate_path, usage, gate, cpu, plan, complete):
    limits = plan['workload']
    command = [sys.executable, '-u', '-B', str(runner), '--root', str(root), '--engine', 'sglang',
               '--container', cpu['container_ID'], '--image-id', IMAGE, '--gate', str(gate_path),
               '--lengths', str(gate['input_tokens']), '--warmup-passes', '5',
               '--phase-timeout', str(limits['warmup_phase_seconds']),
               '--total-timeout', str(limits['warmup_campaign_seconds']),
               '--max-gate-age', '600', '--usage-contract', str(usage), '--run-id', gate['run_id']]
    if complete:
        command.append('--resume')
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('check', 'launch'))
    for path in ('task-root', 'gate', 'usage-contract'):
        parser.add_argument('--' + path, type=Path, required=True)
    parser.add_argument('--gate-sha256', required=True)
    args = parser.parse_args()
    root, gate_path, usage = (path.resolve(strict=True) for path in (args.task_root, args.gate, args.usage_contract))
    client = root / 'client_ds41_g1122'
    assert gate_path.is_relative_to(client) and usage.is_relative_to(client)
    assert digest(gate_path) == args.gate_sha256 and digest(usage) == USAGE_SHA
    gate = json.loads(gate_path.read_bytes())
    assert gate['task_id'] == root.name
    width = gate['input_tokens']
    epoch, _ = PLAN_BY_WIDTH[width]
    plan, plan_sha = load_plan(root/f'sg_pd_g{epoch}'/'wide_workload_plan.json', width)
    step, complete, required, budget = stage_contract(gate, plan, plan_sha)
    cpu = read_cpu(client, width)
    check_window(gate, budget, cpu, required, now())
    runner = client / 'benchmark_pipeline_SG_g1183/run_aisbench_cold_hot_g1340.py'
    assert digest(runner) == RUNNER_SHA
    control = client/'SG_wide_AISBench_control_g1441'/gate['run_id']/step
    assert not control.exists()
    cmd = runner_argv(root, runner, gate_path, usage, gate, cpu, plan, complete)
    preflight = json.loads(subprocess.check_output(cmd + ['--preflight-only'], text=True, cwd=client, timeout=90))
    assert preflight['preflight'] == 'passed' and preflight['benchmark_executed'] is False
    assert preflight['resume_completed_to_skip'] == [f'warmup{i}_{width}' for i in range(1, complete+1)]
    check_idle(gate)
    assert digest(gate_path) == args.gate_sha256 and digest(runner) == RUNNER_SHA and digest(usage) == USAGE_SHA
    check_window(gate, budget, cpu, required, now())
    if args.action == 'check':
        print(json.dumps({'step':step, 'epoch':epoch, 'preflight':preflight, 'submitted':False}))
        return
    control.mkdir(parents=True, exist_ok=False)
    with (control/'runner.log').open('x') as log:
        worker = subprocess.Popen(cmd, cwd=client, stdin=subprocess.DEVNULL, stdout=log,
                                  stderr=subprocess.STDOUT, start_new_session=True)
    handle = {'at':now().isoformat(), 'outer_PID':worker.pid, 'argv':cmd, 'run_id':gate['run_id'],
              'step':step, 'length':width, 'epoch':epoch, 'gate_sha256':args.gate_sha256,
              'completed_warmups_reaudited_and_skipped':preflight['resume_completed_to_skip'],
              'formal_submissions_authorized_by_gate':gate['stage'] == 'formal'}
    with (control/'actual_launch_handle.json').open('x') as out:
        json.dump(handle, out, indent=2)
        out.write('\n')
    print(json.dumps({**handle, 'actual_outer_live':worker.poll() is None}))


if __name__ == '__main__':
    main()
