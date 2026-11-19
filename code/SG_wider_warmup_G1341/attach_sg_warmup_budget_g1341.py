"""Attach a warmup time window to a fresh, already verified SGLang service gate."""
import argparse
import datetime as dt
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace

from run_aisbench_cold_hot_g1340 import validate_warmup_timeout_options, warmup_invocation_count, warmup_timeouts


def read(path, expected_hash=None):
    raw = path.read_bytes()
    digest = sha256(raw).hexdigest()
    assert expected_hash is None or digest == expected_hash, path.name
    return json.loads(raw), digest


def timestamp(value):
    return dt.datetime.fromisoformat(value)


def assert_fresh(stamp, now, *, maximum_age=600):
    assert -60 <= (now - timestamp(stamp)).total_seconds() < maximum_age


def attach(base_path, cpu_path, phase_seconds, campaign_seconds, now):
    gate, base_sha = read(base_path)
    cpu, cpu_sha = read(cpu_path)
    assert_fresh(gate['validated_at'], now)
    assert_fresh(cpu['reported_at_UTC'], now)
    assert gate['engine'] == 'sglang' and gate['workload_guard_actual_verified']
    assert gate['stage'] in ('initial', 'remaining') and gate['warmup_only']
    for flag in ('ready', 'pd_transfer_verified', 'configuration_verified', 'resources_verified'):
        assert gate[flag] is True
    hashes = gate['source_receipts_sha256']
    p, _ = read(base_path.with_name('prefill.json'), hashes['P_readiness'])
    d, _ = read(base_path.with_name('decode.json'), hashes['D_readiness'])
    smoke, _ = read(base_path.with_name('runtime.json'), hashes['PD_runtime'])
    assert smoke['terminal_verified'] and smoke['raw_files_verified'] and not (smoke['outer_live'] or smoke['inner_live'])
    router = smoke['handle']['router']
    assert p['actual_router'] == router
    assert d['graph_capture_all8_verified']

    deadlines = {}
    for role, proof, loaded_container, controller in (
        ('prefill', p, p['container']['Id'], router['P_controller_PID']),
        ('decode', d, d['container_ID'], router['D_controller_PID']),
    ):
        assert_fresh(proof['at'], now)
        assert proof['ready'] == 200 and proof['all8_ownership_and_FDs_verified']
        assert proof['handle']['epoch'] == gate['profile_epoch']
        assert proof['handle']['pid'] == controller == gate['roles'][role]['controller_pid']
        assert loaded_container == gate['roles'][role]['container_id']
        assert loaded_container == router[('P' if role == 'prefill' else 'D') + '_container_ID']
        assert all(values and all(value == 0 for value in values) for values in proof['idle'].values())
        lifetime = proof['state']['lifetime_seconds']
        assert type(lifetime) is int and lifetime > 0
        deadlines[role] = timestamp(proof['handle']['at']) + dt.timedelta(seconds=lifetime)

    guard = p['actual_workload_guard']
    assert p['guard_FD_verified'] and p['guard_PID'] == guard['pid'] and guard['phase'] == 'holding'
    original_guard = smoke['fresh_live_guard']
    assert original_guard['phase'] == 'holding' and original_guard['pid'] == router['guard_PID']
    assert guard['pair'] == original_guard['pair'] and guard['bound_s'] > 0
    if guard['pid'] != router['guard_PID']:
        assert guard['previous_guard_PID'] == router['guard_PID'] and guard['router_PID'] == router['router_PID']
        assert guard['prefill_container_ID'] == router['P_container_ID']
        assert guard['decode_container_ID'] == router['D_container_ID']
    deadlines['pair_workload_lock'] = timestamp(guard['at']) + dt.timedelta(seconds=guard['bound_s'])
    deadlines['CPU_client'] = timestamp(cpu['conservative_deadline_UTC'])

    gate['warmup_wall_budget'] = {
        'phase_seconds': phase_seconds, 'campaign_seconds': campaign_seconds,
        'decode_waiting_seconds_by_rank': d['KV_waiting_timeout_actual_by_rank'],
        'router_request_timeout_seconds': router['configured_request_deadline_seconds'],
        'resource_deadlines_UTC': {key: value.isoformat() for key, value in deadlines.items()},
        'CPU_client_container_id': cpu['container_ID'],
        'base_service_gate_sha256': base_sha, 'CPU_client_lease_sha256': cpu_sha,
    }
    resume = gate['stage'] == 'remaining'
    args = SimpleNamespace(phase_timeout=phase_seconds, total_timeout=campaign_seconds,
                           lengths=[gate['input_tokens']], warmup_passes=5, resume=resume)
    assert validate_warmup_timeout_options(gate, args)
    completed = gate['completed_warmups'] if resume else 0
    next_count = warmup_invocation_count(gate, completed, args.warmup_passes)
    # The runner verifies elapsed campaign time against the original manifest.
    warmup_timeouts(gate, now, campaign_seconds, next_count)
    return gate


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--base-gate', type=Path, required=True)
    cli.add_argument('--cpu-lease', type=Path, required=True)
    cli.add_argument('--phase-seconds', type=int, required=True)
    cli.add_argument('--campaign-seconds', type=int, required=True)
    cli.add_argument('--output', type=Path, required=True)
    args = cli.parse_args()
    assert args.output.parent.resolve() == args.base_gate.parent.resolve()
    result = attach(args.base_gate, args.cpu_lease, args.phase_seconds, args.campaign_seconds,
                    dt.datetime.now(dt.timezone.utc))
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps({'gate_sha256': sha256(args.output.read_bytes()).hexdigest(),
                      'requests_submitted': False, 'stage': result['stage']}))


if __name__ == '__main__':
    main()
