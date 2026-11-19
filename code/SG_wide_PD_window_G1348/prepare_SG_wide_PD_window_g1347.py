"""Prepare a timed SGLang P/D source bundle from the fixed G1317 execution configuration."""
import argparse
import ast
from hashlib import sha256
import json
from pathlib import Path

WIDE_INPUTS = (65536, 131072, 262144)
PHASE_CLEANUP_SECONDS = 90
RESOURCE_MARGIN_SECONDS = 300


def digest(raw):
    return sha256(raw).hexdigest()


def json_bytes(value):
    return (json.dumps(value, indent=2, ensure_ascii=False) + '\n').encode()


def read_pinned_files(directory, manifest_digest):
    raw = (directory / 'deployment_source_sha256.json').read_bytes()
    assert digest(raw) == manifest_digest
    expected = json.loads(raw)
    assert len(expected) == 58
    data = {}
    for name, checksum in expected.items():
        file = directory / name
        assert file.resolve().is_relative_to(directory.resolve()) and not file.is_symlink()
        data[name] = file.read_bytes()
        assert digest(data[name]) == checksum, name
    return data


def once(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new)


def budget(args, observed):
    assert args.input_tokens in WIDE_INPUTS
    assert args.epoch > 1317
    assert observed['epoch'] == 1317 and observed['warmup_passes'] == 5
    assert observed['warmup_requests_per_pass'] == 16 and observed['cold_requests'] == 32
    assert observed['output_tokens'] == 1024
    row, = (row for row in observed['scenarios'] if row['input_tokens'] == args.input_tokens)
    assert args.warmup_phase_seconds > row['warmup1_prefill_only_s'], 'Warmup bound is below the existing same-model planning scenario'
    assert min(args.cold_phase_seconds, args.hot_phase_seconds) > row['cold32_prefill_only_s'], 'Cold/replay bounds are below the no-prefix-reuse planning scenario'
    assert args.request_timeout_seconds >= args.warmup_phase_seconds
    assert args.startup_seconds > 0 and args.checkpoint_seconds >= 0
    warmup_window = args.warmup_phase_seconds + PHASE_CLEANUP_SECONDS + RESOURCE_MARGIN_SECONDS
    campaign = 5 * (args.warmup_phase_seconds + PHASE_CLEANUP_SECONDS) + RESOURCE_MARGIN_SECONDS + args.checkpoint_seconds
    formal_window = (args.cold_phase_seconds + args.hot_phase_seconds
                     + 2 * PHASE_CLEANUP_SECONDS + RESOURCE_MARGIN_SECONDS)
    roles = args.startup_seconds + campaign + formal_window + 2 * RESOURCE_MARGIN_SECONDS
    return {
        'width_tokens': args.input_tokens, 'outputs_per_request': 1024, 'concurrency': 16,
        'warmup_passes': 5, 'requests_per_warmup': 16, 'cold_requests': 32, 'replay_requests': 32,
        'warmup_phase_seconds': args.warmup_phase_seconds, 'warmup_campaign_seconds': campaign,
        'cold_phase_seconds': args.cold_phase_seconds, 'replay_phase_seconds': args.hot_phase_seconds,
        'requested_D_wait_seconds': args.request_timeout_seconds,
        'requested_router_request_timeout_seconds': args.request_timeout_seconds,
        'startup_reserve_seconds': args.startup_seconds, 'checkpoint_reserve_seconds': args.checkpoint_seconds,
        'minimum_P_D_lifetime_seconds': roles,
        'remaining_CPU_and_pair_lease_required_at_warmup_submit_seconds': warmup_window,
        'remaining_CPU_and_pair_lease_required_at_formal_submit_seconds': formal_window,
        'current_CPU_fresh_lifetime_seconds': 70000,
        'current_CPU_fresh_lifetime_could_cover_formal_window': 70000 > formal_window,
        'assumptions': 'Wall-time allowances apply to the observed 32K scenario and pessimistic no-prefix-reuse replay. Longer-context runtime is as yet unmeasured.',
    }


def render_role(original, role, epoch, plan):
    updated = dict(original)
    suffix = f'g{epoch}'
    run = f'sg_pd_{suffix}'
    controller = original['node_controller.py'].decode().replace('sg_pd_g1317', run).replace("'-g1317'", f"'-{suffix}'")
    controller = once(controller, 'generation=1317', f'generation={epoch}')
    lifetime = plan['minimum_P_D_lifetime_seconds']
    controller = once(controller, 'LIFETIME=108000', f'LIFETIME={lifetime}')
    controller = once(controller, 'exec sleep 108100', f'exec sleep {lifetime+100}')
    updated['node_controller.py'] = controller.encode()
    entry = original['run_pd.sh'].decode()
    assert entry.count('/task/sg_pd_g1317') == 2
    updated['run_pd.sh'] = entry.replace('/task/sg_pd_g1317', '/task/' + run).encode()
    prior = json.loads(original['candidate.json'])
    assert prior['role'] == role
    candidate = json.loads(original['candidate.json'])
    candidate['environment'] = {key: value.replace('sg_pd_g1317', run) for key, value in prior['environment'].items()}
    if role == 'd':
        assert candidate['environment']['SGLANG_DISAGGREGATION_WAITING_TIMEOUT'] == '14400'
        candidate['environment']['SGLANG_DISAGGREGATION_WAITING_TIMEOUT'] = str(plan['requested_D_wait_seconds'])
    assert candidate['argv'] == prior['argv']
    assert candidate['environment']['SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT'] == '3600'
    assert {k: v for k, v in candidate.items() if k != 'environment'} == {k: v for k, v in prior.items() if k != 'environment'}
    updated['candidate.json'] = json_bytes(candidate)
    assert {path for path in original if original[path] != updated[path]} == {'node_controller.py', 'run_pd.sh', 'candidate.json'}
    for path, raw in updated.items():
        if path.endswith('.py'):
            ast.parse(raw, filename=path)
    return updated, digest(json_bytes(candidate['argv']))


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--base', type=Path, required=True)
    cli.add_argument('--scenario', type=Path, required=True)
    cli.add_argument('--output', type=Path, required=True)
    for key in ('epoch', 'input-tokens', 'warmup-phase-seconds', 'cold-phase-seconds',
                'hot-phase-seconds', 'request-timeout-seconds', 'startup-seconds', 'checkpoint-seconds'):
        cli.add_argument('--' + key, type=int, required=True)
    args = cli.parse_args()
    source = (args.base / 'preparation_manifest.json').read_bytes()
    original = json.loads(source)
    assert original['epoch'] == 1317 and original['P_and_D_model_arg_vectors_byte_equivalent']
    observed_raw = args.scenario.read_bytes()
    plan = budget(args, json.loads(observed_raw))
    roles = {}
    results = {'epoch': args.epoch, 'derived_from_actual_epoch': 1317, 'workload': plan,
               'parent_manifest_sha256': digest(source), 'planning_scenario_sha256': digest(observed_raw),
               'P_bootstrap_seconds': 3600, 'roles': {},
               'scope': 'Time-window candidate files only. Actual launch, ownership, same-cohort quality, and all warmup/scored outputs require live checks.'}
    for role in ('p', 'd'):
        baseline = original['roles'][role]['deployment_manifest_SHA256']
        files = read_pinned_files(args.base / role, baseline)
        updated, argv_sha = render_role(files, role, args.epoch, plan)
        manifest = json_bytes({name: digest(raw) for name, raw in updated.items()})
        updated['deployment_source_sha256.json'] = manifest
        roles[role] = updated
        results['roles'][role] = {'verified_source_files': len(files), 'original_manifest_sha256': baseline,
                                  'deployment_manifest_sha256': digest(manifest), 'unchanged_model_argv_sha256': argv_sha,
                                  'changed_files': ['candidate.json', 'node_controller.py', 'run_pd.sh']}
    router = json_bytes({'epoch': args.epoch, 'context_length': args.input_tokens,
        'max_concurrent_requests': 16, 'queue_size': 100, 'request_timeout_seconds': args.request_timeout_seconds,
        'retries_enabled': False, 'health_check_endpoint': '/ready'})
    results['router_window_sha256'] = digest(router)
    args.output.mkdir(exist_ok=False)
    (args.output / 'router_window.json').write_bytes(router)
    for role, files in roles.items():
        for name, raw in files.items():
            path = args.output / role / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
    (args.output / 'preparation_manifest.json').write_bytes(json_bytes(results))
    print(json.dumps({'epoch': args.epoch, 'width': args.input_tokens, 'lifetime': plan['minimum_P_D_lifetime_seconds'],
                      'D_wait': args.request_timeout_seconds, 'formal_lease': plan['remaining_CPU_and_pair_lease_required_at_formal_submit_seconds']}))


if __name__ == '__main__':
    main()
