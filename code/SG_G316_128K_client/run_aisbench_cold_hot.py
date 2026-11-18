"""Host-side bounded AISBench runner. Invoke detached; no SSH or NPU commands."""
import argparse
import ast
import datetime as dt
import fcntl
import json
import os
import signal
from pathlib import Path
import subprocess
import time
import urllib.error
import urllib.request

from audit_aisbench import atomic, audit_phase, audit_run, sha


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def request(url, method='GET', timeout=60):
    try:
        with urllib.request.urlopen(urllib.request.Request(url, method=method), timeout=timeout) as r:
            return {'status': r.status, 'body': r.read().decode(errors='replace'), 'at': utc()}
    except urllib.error.HTTPError as e:
        return {'status': e.code, 'body': e.read().decode(errors='replace'), 'at': utc()}


def config_value(node):
    if isinstance(node, ast.Constant): return node.value
    if isinstance(node, ast.Name): return node.id
    if isinstance(node, ast.List): return [config_value(x) for x in node.elts]
    if isinstance(node, ast.Dict): return {config_value(k): config_value(v) for k, v in zip(node.keys, node.values)}
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'dict':
        return {k.arg: config_value(k.value) for k in node.keywords}
    raise ValueError(f'Unsupported configuration node {ast.dump(node)}')


def read_config(path):
    return {n.targets[0].id: config_value(n.value) for n in ast.parse(path.read_text()).body
            if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)}


def warmup_plan(gate, passes):
    assert passes >= 2, 'At least two full-prefill warmups required'
    if gate.get('cold_mode', 'flush') == 'flush':
        return ['warmup'] * passes
    phases = gate.get('warmup_dataset_phases')
    assert isinstance(phases, list) and len(phases) == passes, 'Novel-prefix mode requires one explicit independent dataset phase per warmup pass'
    assert len(set(phases)) == len(phases), 'Novel-prefix warmups must use distinct dataset phases'
    assert all(isinstance(s, str) and s.startswith('warmup') and s.replace('_', '').isalnum() for s in phases)
    return phases


def validate_warmup_resume(root, out, current, prepared, usage_contract, now=None):
    """Read-only recovery gate. Accept completed unscored warmups, never resend them."""
    assert usage_contract, 'Resume requires independently verified server-usage evidence'
    old = json.loads((out / 'run_manifest.json').read_text())
    state = json.loads((out / 'state.json').read_text())
    assert state['phase'] in ['failed_requires_review','warmup_diagnostics_complete_pending_compile_audit','warmup_paused_for_P_snapshot'], 'Only unscored diagnostic terminal states can resume'
    assert 'AISBench failed rc=' not in state.get('error', ''), 'A failed CLI is not a completed warmup audit recovery'
    for key in ['engine', 'run_id', 'client_container', 'client_container_id', 'image_id', 'lengths',
                'warmup_passes', 'warmup_dataset_phases', 'warmup_dataset_sha256',
                'dataset_manifest_sha256', 'formal_dataset_sha256', 'cold_mode', 'cold_prefix_audit_sha256', 'usage_contract']:
        assert old[key] == current[key], f'Resume immutable identity/assets mismatch: {key}'
    assert old.get('client_attempts') == current.get('client_attempts') == 1
    for key in ['proxy_url', 'model_id', 'max_model_len', 'cold_mode', 'warmup_dataset_phases']:
        assert old['gate'].get(key) == current['gate'].get(key), f'Resume service configuration changed: {key}'
    for role in ['prefill', 'decode']:
        for key in ['container_id', 'started_at', 'base_url']:
            assert old['gate']['roles'][role][key] == current['gate']['roles'][role][key], f'Resume actual role identity changed: {role}.{key}'
    for path in out.iterdir():
        assert not path.name.startswith(('cold_', 'hot_', 'before_cold_', 'after_cold_', 'before_hot_', 'after_hot_')), 'Scored artifacts exist; this warmup-only resume is forbidden'
    ledger_path = root / 'client_vllm_g42' / f"formal_submissions_{old['engine']}.json"
    if ledger_path.exists():
        target_hashes = {prepared['formal', length]['dataset_sha256'] for length in current['lengths']}
        previous = json.loads(ledger_path.read_text()).get('submissions', [])
        assert not any(entry['dataset_sha256'] in target_hashes for entry in previous), 'This formal dataset already claimed; preserve possibly submitted status and forbid replay'
    first_length = old['lengths'][0]
    allowed = [f'warmup{i + 1}_{first_length}' for i in range(old['warmup_passes'])]
    commands = {p.name.removesuffix('_command.json'): p for p in out.glob('*_command.json')}
    assert commands and set(commands).issubset(allowed), 'Only initial unscored warmup prefix can resume'
    completed = []
    found_gap = False
    for i, name in enumerate(allowed):
        if name not in commands:
            found_gap = True
            continue
        assert not found_gap, 'Completed warmups must form a contiguous prefix'
        dataset_phase = old['warmup_dataset_phases'][i]
        command = json.loads(commands[name].read_text())
        assert command['requests'] == 16 and command['dataset_phase'] == dataset_phase
        for key in ['dataset_sha256', 'config_sha256']:
            assert command[key] == prepared[dataset_phase, first_length][key], f'Completed warmup {key} changed'
        assert (out / f'before_{name}_server_snapshot.json').is_file(), 'Original warmup before-snapshot missing'
        audited = audit_phase(out / name, first_length, requests=16, usage_contract=usage_contract)
        assert audited['fully_validated'] is True, 'Resume must finish the actual raw/usage audit first'
        completed.append({'name': name, 'length': first_length, 'dataset_phase': dataset_phase,
                          'command_sha256': sha(commands[name]), 'audit': audited,
                          'missing_after_snapshot': not (out / f'after_{name}_server_snapshot.json').exists()})
    started = dt.datetime.fromisoformat(old['benchmark_started_at'].replace('Z', '+00:00'))
    now = now or dt.datetime.now(dt.timezone.utc)
    elapsed = (now - started).total_seconds()
    limit = min(current['total_timeout_seconds'], old.get('total_timeout_seconds', 21600), 21600)
    assert 0 <= elapsed < limit - 60, 'Original campaign lifecycle bound expired; never reset it on resume'
    return {'old_manifest': old, 'old_manifest_sha256': sha(out / 'run_manifest.json'),
            'old_state': state, 'completed': completed, 'elapsed_before_resume_s': elapsed,
            'total_timeout_seconds': limit, 'original_started_at': old['benchmark_started_at'],
            'deadline_utc': (started + dt.timedelta(seconds=limit)).isoformat()}


def compile_cache():
    base=Path('<TASK_ROOT>/sg_pd_g181')
    roots=[base/'cache/triton',base/'cache/ascend',base/'cache/torch_extensions',base/'home/.cache/vllm',base/'home/.cache/torch',base/'tmp/torchinductor_root']
    return {str(q.relative_to(base)):{'bytes':q.stat().st_size,'mtime_ns':q.stat().st_mtime_ns} for root in roots for q in root.rglob('*') if q.is_file() and not q.name.endswith('.lock')}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True, type=Path, help='Absolute HOST task root mapped to /task')
    p.add_argument('--engine', required=True, choices=['vllm', 'sglang'])
    p.add_argument('--container', required=True)
    p.add_argument('--image-id', required=True, help='Exact validated AISBench image ID')
    p.add_argument('--gate', type=Path)
    p.add_argument('--lengths', nargs='+', type=int, default=[32768, 65536, 131072, 262144])
    p.add_argument('--warmup-passes', type=int, default=2)
    p.add_argument('--phase-timeout', type=int, default=5400)
    p.add_argument('--total-timeout', type=int, default=21600)
    p.add_argument('--max-gate-age', type=int, default=1800)
    p.add_argument('--usage-contract', type=Path)
    p.add_argument('--run-id')
    p.add_argument('--resume', action='store_true', help='Explicitly recover completed initial warmups only; scored runs cannot resume')
    p.add_argument('--preflight-only', action='store_true')
    a = p.parse_args()
    assert not a.resume or a.run_id, '--resume requires an explicit --run-id'
    root = a.root.resolve(strict=True)
    assert a.root.is_absolute(), '--root must be explicit absolute host directory'
    os.chdir(root)
    assert a.warmup_passes >= 2, 'At least two same-length full-prefill warmups required'
    assert all(n in [32768, 65536, 131072, 262144] for n in a.lengths)
    assert len(set(a.lengths)) == len(a.lengths)
    gate_path = a.gate or root / f'service_gate_{a.engine}.json'
    gate = json.loads(gate_path.read_text())
    assert gate['schema'] == 1 and gate['task_id'] == root.name and gate['engine'] == a.engine
    for key in ['ready', 'pd_transfer_verified', 'configuration_verified', 'resources_verified']:
        assert gate[key] is True, f'Service gate missing {key}'
    age = (dt.datetime.now(dt.timezone.utc) - dt.datetime.fromisoformat(gate['validated_at'].replace('Z', '+00:00'))).total_seconds()
    assert 0 <= age <= a.max_gate_age, f'Stale gate age {age}s'
    assert set(gate['roles']) == {'prefill', 'decode'}
    assert gate['max_model_len'] >= max(a.lengths) + 1024
    cold_mode = gate.get('cold_mode', 'flush')
    assert cold_mode in ['flush', 'novel_prefixes']
    warmup_phases = warmup_plan(gate, a.warmup_passes)
    for r in gate['roles'].values():
        assert r['ready'] is True and r['container_id'] and r['started_at']
        if cold_mode == 'flush':
            assert r['flush_url'] and r['flush_method'] in ['POST', 'GET']
            assert r['flush_success_regex'], 'Require reviewed actual reset response expression'
    assert gate['model_id'] == 'glm53flash'
    client_dir = root / 'client_vllm_g42'
    client_dir.mkdir(exist_ok=True)
    lock_file = (client_dir / f'{a.engine}_runner.lock').open('a+')
    fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    client = json.loads(subprocess.check_output(['docker', 'inspect', a.container], text=True))[0]
    assert client['State']['Running'] and client['Image'] == a.image_id
    assert client['HostConfig']['Devices'] in [None, []], 'Client must have no NPU devices'
    nano_cpus = int(client['HostConfig'].get('NanoCpus', 0))
    cpuset = client['HostConfig'].get('CpusetCpus', '')
    cpu_ids = set()
    for part in cpuset.split(',') if cpuset else []:
        bounds = [int(x) for x in part.split('-')]
        cpu_ids.update(range(bounds[0], bounds[-1] + 1))
    assert (0 < nano_cpus <= 4_000_000_000) or (0 < len(cpu_ids) <= 4), 'Client CPU bound unverified'
    assert client['HostConfig'].get('Privileged', False) is False
    mounts = {m['Destination']: m for m in client['Mounts']}
    assert Path(mounts['/task']['Source']).resolve() == root
    assert mounts['/models/glm53']['RW'] is False
    assets_state = json.loads((client_dir / 'assets_state.json').read_text())
    assert assets_state['phase'] == 'complete'
    dataset_manifest = json.loads((client_dir / 'dataset_manifest.json').read_text())
    assert dataset_manifest['client_concurrency'] == 16 and dataset_manifest['output_tokens'] == 1024
    prepared = {}
    for length in a.lengths:
        for phase, count in [(s, 16) for s in dict.fromkeys(warmup_phases)] + [('formal', 32)]:
            record = next(x for x in dataset_manifest['profiles'] if x['phase'] == phase and x['input_tokens'] == length)
            assert record['requests'] == count and record['min_actual'] == record['max_actual'] == length
            path = client_dir / 'datasets' / f'{phase}_{length}.jsonl'
            assert sha(path) == record['sha256']
            config = root / 'client_configs_sg_g199' / f'{a.engine}_{phase}_{length}.py'
            cfg = read_config(config)
            m = cfg['models'][0]
            assert m['type'] == 'VLLMCustomAPI' and m['model'] == gate['model_id']
            assert m['stream'] is True and m['batch_size'] == 16 and m['max_out_len'] == 1024
            assert m['retry'] == 1 and m['request_rate'] == 0
            assert m['generation_kwargs'] == {'temperature': 0.0, 'ignore_eos': True, 'min_tokens': 1024}
            assert f"http://{m['host_ip']}:{m['host_port']}" == gate['proxy_url'].rstrip('/')
            assert cfg['datasets'][0]['path'] == f'/task/client_vllm_g42/datasets/{phase}_{length}.jsonl'
            prepared[phase, length] = {'dataset_sha256': record['sha256'], 'config_sha256': sha(config)}
        if cold_mode == 'novel_prefixes':
            phase_hashes = [prepared[s, length]['dataset_sha256'] for s in warmup_phases + ['formal']]
            assert len(set(phase_hashes)) == len(phase_hashes), 'Warmup passes/formal must have independent dataset SHA'
    if cold_mode == 'novel_prefixes':
        assert gate['formal_not_previously_submitted'] is True
        prefix_audit_path = root / gate['cold_prefix_audit']
        prefix_audit = json.loads(prefix_audit_path.read_text())
        assert prefix_audit['verified'] is True and prefix_audit['prefix_tokens'] >= 128
        assert prefix_audit['formal_first_prefixes_unique'] is True
        assert prefix_audit['warmup_first_prefixes_unique'] is True
        assert prefix_audit['no_formal_warmup_prefix_collision'] is True
        assert prefix_audit['tokenizer_sha256'] == dataset_manifest['tokenizer_sha256']
        for phase_name in ['formal'] + list(dict.fromkeys(warmup_phases)):
            for n in a.lengths:
                expected = prepared[phase_name, n]['dataset_sha256']
                assert any(e['sha256'] == expected for e in prefix_audit['datasets']), 'Dataset absent from prefix audit'
    # Read-only route preflight. Never assume admin reset routes exist.
    if cold_mode == 'flush':
        for role, r in gate['roles'].items():
            schema = request(r['base_url'].rstrip('/') + '/openapi.json')
            assert schema['status'] == 200, f'{role} OpenAPI unavailable; reviewed reset route required'
            from urllib.parse import urlsplit
            route = urlsplit(r['flush_url']).path
            assert r['flush_method'].lower() in json.loads(schema['body'])['paths'].get(route, {}), f'{role} reset route missing'
    run_id = a.run_id or dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    assert run_id and Path(run_id).name == run_id and run_id not in ['.', '..'], 'Run ID must be a single directory name'
    out = client_dir / 'runs' / a.engine / run_id
    state_path = out / 'state.json'
    started = time.monotonic()
    run_manifest = {'schema': 1, 'engine': a.engine, 'run_id': run_id, 'host_runner_pid': os.getpid(),
                    'client_container': a.container, 'client_container_id': client['Id'], 'image_id': client['Image'],
                    'lengths': a.lengths, 'warmup_passes': a.warmup_passes, 'gate': gate, 'gate_sha256': sha(gate_path),
                    'warmup_dataset_phases': warmup_phases,
                    'warmup_dataset_sha256': {str(n): {s: prepared[s, n]['dataset_sha256'] for s in dict.fromkeys(warmup_phases)} for n in a.lengths},
                    'dataset_manifest_sha256': sha(client_dir / 'dataset_manifest.json'),
                    'formal_dataset_sha256': {str(n): prepared['formal', n]['dataset_sha256'] for n in a.lengths},
                    'cpu_only_client': True, 'benchmark_started_at': utc(),
                    'total_timeout_seconds': a.total_timeout, 'phase_timeout_seconds': a.phase_timeout,
                    'client_attempts': 1, 'client_attempts_semantics': 'Pinned BaseAPIModel uses range(self.retry); retry=1 sends one attempt, no retry',
                    'cold_mode': cold_mode,
                    'cold_prefix_audit_sha256': sha(prefix_audit_path) if cold_mode == 'novel_prefixes' else None,
                    'cache_protocol': 'Three independent full-shape warmup passes, final two compile-stable, then cold32 and identical formal32 hot replay. Flush mode resets both roles; novel_prefixes mode verifies unsubmitted unique prefixes and never claims reset.',
                    'usage_contract': str(a.usage_contract) if a.usage_contract else None}
    resume_info = None
    if a.resume:
        assert out.is_dir(), 'Explicit resume run directory is missing'
        resume_info = validate_warmup_resume(root, out, run_manifest, prepared, a.usage_contract)
        top = subprocess.check_output(['docker', 'top', a.container, '-eo', 'pid,args'], text=True)
        assert not any('ais_bench' in line for line in top.splitlines()[1:]), 'Client still has an active AISBench process; no resume'
        a.total_timeout = resume_info['total_timeout_seconds']
        a.phase_timeout = min(a.phase_timeout, resume_info['old_manifest'].get('phase_timeout_seconds', 5400))
        started = time.monotonic() - resume_info['elapsed_before_resume_s']
    else:
        assert not out.exists(), 'Run exists; use explicit --resume only for audited unscored warmups'
    if a.preflight_only:
        print(json.dumps({'preflight': 'passed', 'engine': a.engine, 'benchmark_executed': False,
                          'resume_completed_to_skip': [x['name'] for x in resume_info['completed']] if resume_info else []}))
        return 0
    completed_resume = {}
    if resume_info:
        resume_id = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        atomic(out / f'resume_prior_state_{resume_id}.json', resume_info['old_state'])
        for item in resume_info['completed']:
            completed_resume[item['name']] = item
            audit_path = out / f"{item['name']}_audit.json"
            if not audit_path.exists():
                atomic(audit_path, item['audit'])
        record = {'schema': 1, 'at': utc(), 'run_id': run_id, 'engine': a.engine,
                  'host_runner_pid': os.getpid(), 'original_manifest_sha256': resume_info['old_manifest_sha256'],
                  'original_benchmark_started_at': resume_info['original_started_at'],
                  'deadline_utc': resume_info['deadline_utc'], 'total_timeout_seconds': a.total_timeout,
                  'prior_gate_sha256': resume_info['old_manifest']['gate_sha256'],
                  'refreshed_gate_sha256': sha(gate_path), 'refreshed_gate': gate,
                  'same_actual_service_identities_verified': True,
                  'runner_sha256': sha(__file__), 'completed_warmups_reaudited_and_not_resent': resume_info['completed'],
                  'client_top': top, 'scope': 'Initial unscored warmup audit recovery only; no scored artifact/claim accepted'}
        atomic(out / f'resume_manifest_{resume_id}.json', record)
        assert sha(out / 'run_manifest.json') == resume_info['old_manifest_sha256'], 'Original manifest must remain immutable'
    else:
        out.mkdir(parents=True, exist_ok=False)
        atomic(out / 'run_manifest.json', run_manifest)
    def interrupted(signum, frame):
        raise InterruptedError(f'Host runner received signal {signum}; inner bounded CLI status requires inspection')
    signal.signal(signal.SIGTERM, interrupted)

    def stamp(phase, **kw):
        atomic(state_path, {'at': utc(), 'run_id': run_id, 'engine': a.engine, 'phase': phase,
                            'elapsed_s': round(time.monotonic() - started, 3), **kw})
        print(json.dumps({'at': utc(), 'phase': phase, **kw}), flush=True)

    def snapshot(tag, posthoc=False):
        values = {}
        for role, r in gate['roles'].items():
            health = request(r['base_url'].rstrip('/') + '/health')
            models = request(r['base_url'].rstrip('/') + '/v1/models')
            values[role] = {'health': health, 'models': models}
            if r.get('metrics_url'):
                values[role]['metrics'] = request(r['metrics_url'])
            assert health['status'] == 200 and models['status'] == 200, f'{role} lost HTTP readiness'
            ids = [x['id'] for x in json.loads(models['body'])['data']]
            assert gate['model_id'] in ids, f'{role} served model changed'
        if posthoc:
            assert tag.startswith('after_warmup'), 'Only completed warmup after-snapshots may be posthoc'
            values['_capture_provenance'] = {'posthoc': True, 'at': utc(), 'resume_id': resume_id,
                                             'scope': 'Current role health/metrics, not historical phase-end counters'}
        atomic(out / f'{tag}_server_snapshot.json', values)

    def flush(tag):
        import re
        result = {}
        for role, r in gate['roles'].items():
            item = request(r['flush_url'], r['flush_method'])
            item['accepted'] = item['status'] == 200 and re.search(r['flush_success_regex'], item['body']) is not None
            result[role] = item
            atomic(out / f'{tag}_flush.json', result)
            assert item['accepted'], f'{role} prefix reset rejected; raw response retained'

    def phase(name, length, dataset_phase, count):
        dataset_path = client_dir / 'datasets' / f'{dataset_phase}_{length}.jsonl'
        config_path = root / 'client_configs_sg_g199' / f'{a.engine}_{dataset_phase}_{length}.py'
        assert sha(dataset_path) == prepared[dataset_phase, length]['dataset_sha256'], 'Dataset changed after preflight'
        assert sha(config_path) == prepared[dataset_phase, length]['config_sha256'], 'Configuration changed after preflight'
        atomic(out / f'before_{name}_{length}_decode_compile_cache.json', compile_cache())
        snapshot(f'before_{name}_{length}')
        remaining = a.total_timeout - (time.monotonic() - started)
        assert remaining > 60, 'Total runtime bound reached'
        timeout = max(1, int(min(a.phase_timeout, remaining - 30)))
        target = f'/task/client_vllm_g42/runs/{a.engine}/{run_id}/{name}_{length}'
        cmd = ['docker', 'exec', '-w', '/task', '-e', 'BFCL_PROJECT_ROOT=/task/client_vllm_g42/bfcl', a.container, 'timeout', '--signal=TERM', '--kill-after=30', str(timeout),
               'ais_bench', f'/task/client_configs_sg_g199/{a.engine}_{dataset_phase}_{length}.py', '--mode', 'perf',
               '--num-warmups', '0', '--num-prompts', str(count), '--max-num-workers', '1', '--work-dir', target]
        # The pinned AISBench image rejects response-anomaly in perf mode.
        # Preserve its standard raw details/JSON/CSV and CLI log for auditing.
        meta = {'argv': cmd, 'at': utc(), 'requests': count, 'dataset_phase': dataset_phase, **prepared[dataset_phase, length]}
        atomic(out / f'{name}_{length}_command.json', meta)
        stamp('aisbench_running', subphase=name, input_tokens=length, scored=name in ['cold', 'hot'])
        with (out / f'{name}_{length}_cli.log').open('x') as log:
            proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
            stamp('aisbench_running', subphase=name, input_tokens=length, docker_exec_pid=proc.pid,
                  inner_timeout_seconds=timeout, command_file=f'{name}_{length}_command.json')
            try:
                rc = proc.wait(timeout=timeout + 90)
            except BaseException:
                stamp('interrupted_client_status_unknown', subphase=name, input_tokens=length,
                      docker_exec_pid=proc.pid, inner_timeout_seconds=timeout,
                      note='Do not retry or call terminal merely because SSH/runner detached. Inspect actual client process; inner timeout still bounds it.')
                raise
        stamp('aisbench_phase_terminal', subphase=name, input_tokens=length, exit_code=rc)
        assert rc == 0, f'AISBench failed rc={rc}; outputs retained, no automatic retry'
        assert sha(dataset_path) == prepared[dataset_phase, length]['dataset_sha256'], 'Dataset changed during phase'
        assert sha(config_path) == prepared[dataset_phase, length]['config_sha256'], 'Configuration changed during phase'
        audited = audit_phase(out / f'{name}_{length}', length, count, a.usage_contract)
        atomic(out / f'{name}_{length}_audit.json', audited)
        snapshot(f'after_{name}_{length}')
        after_compile=compile_cache()
        atomic(out / f'after_{name}_{length}_decode_compile_cache.json', after_compile)
        if name in ['cold','hot']:
            before_compile=json.loads((out/f'before_{name}_{length}_decode_compile_cache.json').read_text())
            changes=[k for k,v in after_compile.items() if v!=before_compile.get(k)]
            atomic(out/f'{name}_{length}_compile_audit.json',dict(passed=not changes,changes=changes,scope='D actual compiler-cache metadata before/after; root independently audits P before final acceptance'))
            assert not changes,'D compiler artifacts changed during formal phase; preserve raw results and stop, no automatic replay'


    def claim_formal(length):
        ledger_path = client_dir / f'formal_submissions_{a.engine}.json'
        ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {'submissions': []}
        dataset_sha = prepared['formal', length]['dataset_sha256']
        assert not any(x['dataset_sha256'] == dataset_sha for x in ledger['submissions']), 'Novel formal data already claimed/submitted; preserve failure and use new audited prefixes'
        ledger['submissions'].append({'run_id': run_id, 'claimed_at': utc(), 'dataset_sha256': dataset_sha,
                                      'input_tokens': length, 'service_roles': gate['roles'],
                                      'scope': 'Claim before sending; an interrupted attempt remains possibly submitted'})
        atomic(ledger_path, ledger)

    def cache_delta(length, phase_name):
        import re
        before = json.loads((out / f'before_{phase_name}_{length}_server_snapshot.json').read_text())
        after = json.loads((out / f'after_{phase_name}_{length}_server_snapshot.json').read_text())
        results = {}
        for role, role_gate in gate['roles'].items():
            metric_name = role_gate.get('prefix_hit_counter')
            if not metric_name:
                results[role] = {'verified': False, 'reason': 'No reviewed prefix-hit counter supplied'}
                continue
            def counter(snapshot):
                metrics = snapshot[role].get('metrics', {})
                assert metrics.get('status') == 200, 'Metrics collection failed'
                pattern = re.compile(r'^' + re.escape(metric_name) + r'(?:\{[^\n]*\})?\s+([0-9.eE+\-]+)(?:\s+\d+)?$', re.M)
                values = [float(x) for x in pattern.findall(metrics['body'])]
                assert values, f'Counter {metric_name} absent'
                return sum(values)
            first, last = counter(before), counter(after)
            assert last >= first, 'Counter reset during scored phase'
            results[role] = {'verified': True, 'counter': metric_name, 'before': first, 'after': last, 'delta': last - first}
            if phase_name == 'cold':
                assert last == first, f'{role} cold phase had prefix hits; cannot accept cache-cold claim'
        atomic(out / f'{phase_name}_{length}_cache_audit.json', results)

    try:
        for length in a.lengths:
            for i, warmup_dataset_phase in enumerate(warmup_phases):
                phase_name = f'warmup{i + 1}'
                resume_key = f'{phase_name}_{length}'
                if resume_key in completed_resume:
                    item = completed_resume[resume_key]
                    if item['missing_after_snapshot']:
                        snapshot(f'after_{resume_key}', posthoc=True)
                    stamp('completed_warmup_resumed_without_send', subphase=phase_name,
                          input_tokens=length, scored=False, reaudited=True)
                    continue
                if cold_mode == 'flush':
                    flush(f'before_warmup{i + 1}_{length}')
                phase(phase_name, length, warmup_dataset_phase, 16)
                if gate.get('warmup_pause_after') == i + 1:
                    stamp('warmup_paused_for_P_snapshot', scored=False, input_tokens=length, completed_warmups=i + 1, reason='Root collects P cache snapshot before remaining full-shape warmups')
                    return 0
            # Refuse scored phases if the last two full-input warmups have not stabilized.
            warm_previous=json.loads((out/f'warmup{len(warmup_phases)-1}_{length}_audit.json').read_text())
            warm_last=json.loads((out/f'warmup{len(warmup_phases)}_{length}_audit.json').read_text())
            stability={k:abs(warm_last[k]-warm_previous[k])/max(abs(warm_previous[k]),1e-12) for k in ['duration_s','ttft_mean_ms','tpot_mean_ms']}
            changes={}
            for j in range(max(0,len(warmup_phases)-2),len(warmup_phases)):
                tag=f'warmup{j+1}_{length}'
                first=json.loads((out/f'before_{tag}_decode_compile_cache.json').read_text())
                last=json.loads((out/f'after_{tag}_decode_compile_cache.json').read_text())
                changes[tag]=[k for k,v in last.items() if v!=first.get(k)]
            stable=all(stability[k]<=0.05 for k in ['duration_s','ttft_mean_ms']) and not any(changes.values())
            atomic(out/f'warmup_stability_{length}.json',dict(last_two_relative_changes=stability,threshold=0.05,passed=stable,old_all3_numeric_gate_passed=all(v<=.05 for v in stability.values()),compiler_artifact_changes=changes,scope='G37 preregistered: actual no-new-compilation on D plus duration/TTFT stability. TPOT variation remains reported guardrail, not sole compilation proxy. P cache must be independently checked by root before warmup_only is lifted.'))
            if gate.get('warmup_only',False):
                stamp('warmup_diagnostics_complete_pending_compile_audit',performance_validated=False,readiness_passed=stable)
                return 0
            assert gate.get('P_compiler_cache_verified_after_warmups') is True,'Root P cache evidence required before first formal'
            assert stable,'Direct compile or primary-timing readiness failed; never submit formal'

            if cold_mode == 'flush':
                flush(f'before_cold_{length}')
            else:
                claim_formal(length)
            phase('cold', length, 'formal', 32)
            cache_delta(length, 'cold')
            phase('hot', length, 'formal', 32)
            cache_delta(length, 'hot')
        stamp('all_phases_terminal', performance_validated=False)
        audit_run(out, a.usage_contract)
        return 0
    except BaseException as e:
        current = json.loads(state_path.read_text()) if state_path.exists() else {}
        if current.get('phase') != 'interrupted_client_status_unknown':
            stamp('failed_requires_review', error=f'{type(e).__name__}: {e}', performance_validated=False)
        raise


if __name__ == '__main__':
    raise SystemExit(main())
