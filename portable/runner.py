#!/usr/bin/env python3
"""Run bounded ranks under an operator-provided shared allocation policy.

The operator config is private to the environment and must identify an existing
assignment, ownership registry, shared lock, bounded reservation helper, pinned
source, read-only checkpoint and already-running isolated container. This program
never stops unrelated workloads, changes the image, or acquires extra devices.
"""
import argparse
import datetime
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--world-size', type=int, choices=(2,4,8,16), default=4)
    parser.add_argument('--messages')
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--goal', default='reproducible-release')
    parser.add_argument('--context-tokens', type=int)
    parser.add_argument('--max-new-tokens', type=int)
    parser.add_argument('--cache-mib', type=int)
    parser.add_argument('--kda-backend', choices=('ascendc_installed_decode', 'ascendc'), default='ascendc_installed_decode')
    parser.add_argument('--expected-final')
    parser.add_argument('--restore-after-first-token', action='store_true')
    parser.add_argument('--experimental-fp32-dsa', action='store_true')
    parser.add_argument('--experimental-layer-major-kda', action='store_true', help='opt in to the four-rank bounded dense KDA chunk prefill')
    parser.add_argument('--experimental-kda-prefill40', action='store_true', help='opt in to the tested dense KDA 40-token prefill')
    parser.add_argument('--experimental-sparse-kda-layer4', action='store_true', help='also chunk the tested sparse KDA layer 4')
    parser.add_argument('--rollback-probe', action='store_true', help='run the bounded nonzero-state rollback probe')
    parser.add_argument('--reference-probe', action='store_true', help='run the installed four-rank reference')
    parser.add_argument('--snapshot-probe', action='store_true', help='run the bounded four-rank archive-v2 restore probe')
    parser.add_argument('--cache-equivalence-probe', action='store_true', help='compare default and reusable cache on the fixed four-rank request')
    args = parser.parse_args()
    if not args.experimental_fp32_dsa:
        parser.error('explicit --experimental-fp32-dsa is required')
    world_size = args.world_size
    if args.experimental_kda_prefill40 and (not args.experimental_layer_major_kda or args.experimental_sparse_kda_layer4):
        parser.error('--experimental-kda-prefill40 requires layer-major dense KDA and excludes sparse KDA layer4')
    if args.experimental_sparse_kda_layer4 and not args.experimental_layer_major_kda:
        parser.error('--experimental-sparse-kda-layer4 requires --experimental-layer-major-kda')
    if args.experimental_layer_major_kda and (world_size != 4 or args.kda_backend != 'ascendc' or args.rollback_probe or args.reference_probe or args.snapshot_probe or args.cache_equivalence_probe):
        parser.error('experimental layer-major KDA requires four-rank ascendc text mode')
    timeout_seconds = 1200 if args.cache_equivalence_probe else 900 if (args.snapshot_probe or args.rollback_probe) else 2100
    if args.cache_equivalence_probe and world_size != 4:
        parser.error('cache equivalence probe requires four ranks')
    if args.snapshot_probe and world_size != 4:
        parser.error('snapshot probe requires four ranks')
    if args.reference_probe and world_size != 4:
        parser.error('reference probe requires four ranks')
    if args.rollback_probe and world_size not in (2,4):
        parser.error('rollback probe requires two or four ranks')
    if sum((args.rollback_probe, args.reference_probe, args.snapshot_probe, args.cache_equivalence_probe)) > 1:
        parser.error('select at most one probe mode')
    if args.cache_equivalence_probe and args.kda_backend != 'ascendc_installed_decode':
        parser.error('--cache-equivalence-probe requires --kda-backend ascendc_installed_decode')
    if args.snapshot_probe and args.kda_backend != 'ascendc':
        parser.error('--snapshot-probe requires --kda-backend ascendc')
    if args.rollback_probe and args.kda_backend != 'ascendc':
        parser.error('--rollback-probe requires --kda-backend ascendc')
    if not re.fullmatch(r'[a-z0-9][a-z0-9-]{0,31}', args.run_id):
        parser.error('invalid run ID')
    if args.rollback_probe or args.snapshot_probe or args.cache_equivalence_probe:
        if (args.messages is not None or args.context_tokens is not None or
                args.max_new_tokens is not None or args.cache_mib is not None or
                args.expected_final is not None or args.restore_after_first_token):
            parser.error('text request options are not valid with fixed probe modes')
    else:
        if args.messages is None:
            parser.error('--messages is required for text mode')
        args.context_tokens = 128 if args.context_tokens is None else args.context_tokens
        args.max_new_tokens = 64 if args.max_new_tokens is None else args.max_new_tokens
        args.cache_mib = 0 if args.cache_mib is None else args.cache_mib
        if not (1 <= args.context_tokens <= 128 and 1 <= args.max_new_tokens <= 64 and 0 <= args.cache_mib <= 128):
            parser.error('invalid context, generation or cache bound')
        if args.reference_probe and (args.cache_mib or args.restore_after_first_token or args.kda_backend != 'ascendc_installed_decode'):
            parser.error('reference mode does not support DeepInfer cache, restore or backend selection')
    cfg = read(args.config)
    manifest_path = Path(cfg['base_source_manifest']).resolve()
    manifest_pins = [entry for entry in cfg['pinned_files']
                     if Path(entry['path']).resolve() == manifest_path]
    if len(manifest_pins) != 1 or sha(manifest_path) != manifest_pins[0]['sha256']:
        raise RuntimeError('base source manifest must have exactly one matching SHA256 pin')
    root = Path(cfg['task_root']).resolve()
    if not (args.rollback_probe or args.snapshot_probe or args.cache_equivalence_probe):
        messages = Path(args.messages).resolve()
        if not messages.is_file() or not messages.is_relative_to(root) or messages.stat().st_size > 16384:
            parser.error('message file must be under task root and <=16 KiB')
        input_text = messages.read_text()
    run = root / cfg['run_relative'] / ('bounded-cli-' + args.run_id)
    allocation = cfg['allocation']
    if len(allocation) != world_size or sorted(x['rank'] for x in allocation) != list(range(world_size)) or len({x['die'] for x in allocation}) != world_size or len({(x['card'], x['chip']) for x in allocation}) != world_size:
        parser.error('allocation must contain world-size distinct rank/die/card-chip mappings')
    allocation = sorted(allocation, key=lambda x: x['rank'])
    with open(cfg['shared_lock'], 'a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assignment, ownership = read(cfg['assignment']), read(cfg['ownership'])
        for name in ('task_id', 'generation', 'executor'):
            if assignment[name] != ownership[name] or assignment[name] != cfg[name]:
                raise RuntimeError('assignment/ownership identity mismatch: ' + name)
        allowed_dies = assignment['authorized_physical_dies']
        if (assignment['host'] != os.uname().nodename or not isinstance(allowed_dies, list)
                or len(set(allowed_dies)) != len(allowed_dies)
                or not set(x['die'] for x in allocation).issubset(allowed_dies) or ownership['active_jobs']):
            raise RuntimeError('host, allocation, or ownership mismatch')
        if run.exists():
            raise RuntimeError('run ID already exists')
        inspected = json.loads(subprocess.check_output(['docker', 'inspect', cfg['container']], text=True))[0]
        if inspected['Id'] != cfg['container_id'] or inspected['Image'] != cfg['image_id'] or not inspected['State']['Running']:
            raise RuntimeError('container/image identity or state mismatch')
        if subprocess.check_output(['docker', 'top', cfg['container'], '-eo', 'pid,ppid,args'], text=True).strip().count('\n') != 1:
            raise RuntimeError('container has active processes')
        info = subprocess.check_output(['npu-smi', 'info'], text=True)
        for card in {x['card'] for x in allocation}:
            if f'No running processes found in NPU {card}' not in info:
                raise RuntimeError('allocated card occupied or process listing uncertain')
        for entry in cfg['pinned_files']:
            if sha(entry['path']) != entry['sha256']:
                raise RuntimeError('pinned file mismatch: ' + entry['label'])
        manifest = read(cfg['base_source_manifest'])
        for rel, digest in manifest.items():
            if not rel.startswith('cli:') and sha(Path(cfg['source_host']) / rel) != digest:
                raise RuntimeError('source manifest mismatch: ' + rel)
        for file in ('worker.py', 'launcher.py', 'runner.py') + (('rollback_probe.py',) if args.rollback_probe else ('reference_probe.py',) if args.reference_probe else ('rank_snapshot_probe.py',) if args.snapshot_probe else ('rank_cache_equivalence_probe.py',) if args.cache_equivalence_probe else ()):
            manifest['portable:' + file] = sha(Path(__file__).resolve().parent / file)
        manifest['portable:world-size'] = hashlib.sha256(str(world_size).encode()).hexdigest()
        manifest['portable:layer-major-kda'] = hashlib.sha256(str(args.experimental_layer_major_kda).encode()).hexdigest()
        manifest['portable:sparse-kda-layer4'] = hashlib.sha256(str(args.experimental_sparse_kda_layer4).encode()).hexdigest()
        if args.experimental_kda_prefill40:manifest['portable:kda-prefill40'] = hashlib.sha256(b'True').hexdigest()
        payload = json.dumps(manifest, sort_keys=True, separators=(',', ':')).encode()
        source_digest = hashlib.sha256(payload).hexdigest()
        if args.rollback_probe:
            request = {'run_id': args.run_id, 'mode': 'rollback_probe',
                       'kda_backend': 'ascendc', 'context_tokens': 32, 'cache_mib': 0,
                       'tokens': [154822, 154824, 154826, 25062],
                       'injected_rank': min(2, world_size-1), 'failure_points': ['legacy_stale_preflight', 'head_after_stack', 'rank_input_mismatch'], 'mismatch_token': 25063,
                       'committed_prefix_tokens': 3}
        elif args.cache_equivalence_probe:
            request = {'run_id': args.run_id, 'mode': 'cache_equivalence_probe', 'world_size': 4,
                       'kda_backend': 'ascendc_installed_decode', 'context_tokens': 32, 'generation_cap': 6,
                       'prompt': 'What is 7 plus 6? Reply only with the number.',
                       'expected_generated_ids': [154842, 99366, 154827], 'cache_mib': 128}
        elif args.snapshot_probe:
            request = {'run_id': args.run_id, 'mode': 'snapshot_probe', 'world_size': 4,
                       'kda_backend': 'ascendc', 'context_tokens': 32, 'generation_cap': 4,
                       'prompt': 'What is 8 plus 5? Reply only with the number.',
                       'expected_generated_ids': [154842, 99366, 154827], 'archive_version': 2}
        else:
            prepare = ['docker', 'exec', '-i', '-e', 'GLM_MODEL_DIR=' + cfg['model_container'],
                       '-e', 'HF_HUB_OFFLINE=1', '-e', 'TRANSFORMERS_OFFLINE=1', cfg['container'],
                       'python3', cfg['launcher_container'], 'prepare', '--run-id', args.run_id,
                       '--context-tokens', str(args.context_tokens), '--max-new-tokens', str(args.max_new_tokens),
                       '--cache-mib', str(args.cache_mib), '--kda-backend', args.kda_backend]
            if args.expected_final is not None:
                prepare += ['--expected-final', args.expected_final]
            if args.restore_after_first_token:
                prepare.append('--restore-after-first-token')
            result = subprocess.run(prepare, input=input_text, capture_output=True, text=True, timeout=90, check=True)
            request = json.loads(result.stdout.strip().splitlines()[-1])
            if request['run_id'] != args.run_id or request['messages'] != json.loads(input_text) or request['kda_backend'] != args.kda_backend:
                raise RuntimeError('request builder returned a different request')
            if args.experimental_layer_major_kda:
                request['experimental_layer_major_kda'] = True
                if args.experimental_kda_prefill40:request['experimental_kda_prefill40'] = True
                if args.experimental_sparse_kda_layer4:request['experimental_sparse_kda_layer4'] = True
            if args.reference_probe:
                request.pop('kda_backend')
                request['mode'] = 'reference_probe'
                request['reference_backend'] = 'vllm_ascend_external_launcher'
        helper = Path(cfg['reservation_helper'])
        if sha(helper) != cfg['reservation_helper_sha256'] or sha(cfg['policy']) != cfg['policy_sha256']:
            raise RuntimeError('reservation helper or policy pin mismatch')
        spec = importlib.util.spec_from_file_location('operator_unit_budget', helper)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        reservation = module.reserve(root, lock, args.goal, source_digest, 'bounded-cli-' + args.run_id, workers=world_size, timeout_seconds=timeout_seconds)
        run.mkdir(parents=True, exist_ok=False)
        write(run / 'reservation.json', reservation)
        (run / 'source-manifest.json').write_bytes(payload + b'\n')
        write(run / 'request.json', request)
        port = int(subprocess.check_output(['docker', 'exec', cfg['container'], 'python3', '-c', 'import socket;s=socket.socket();s.bind(("127.0.0.1",0));print(s.getsockname()[1]);s.close()'], text=True))
        if not 1024 <= port <= 65535:
            raise RuntimeError('invalid rendezvous port')
        workers, fatal = [], None
        stamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        try:
            for item in allocation:
                rank, die = item['rank'], item['die']
                (run / f'rank{rank}').mkdir()
                log = (run / f'rank{rank}.log').open('w')
                env = dict(cfg['container_env'])
                env.update(GLM_MODEL_DIR=cfg['model_container'], GLM_SOURCE_DIR=cfg['source_container'], GLM_ASCEND_LIBRARY=cfg['library_container'], GLM_EXPERIMENT_RUN_DIR=cfg['run_container_prefix'] + '/bounded-cli-' + args.run_id, ASCEND_RT_VISIBLE_DEVICES=str(die), RANK=str(rank), WORLD_SIZE=str(world_size), MASTER_ADDR='127.0.0.1', MASTER_PORT=str(port))
                argv = ['docker', 'exec']
                for key, value in env.items():
                    argv += ['-e', key + '=' + value]
                if args.rollback_probe:
                    argv += [cfg['container'], 'timeout', '-k', '10', str(timeout_seconds), 'python3', str(Path(cfg['launcher_container']).with_name('rollback_probe.py')), str(rank), str(die), str(port), 'bounded']
                elif args.cache_equivalence_probe:
                    argv += [cfg['container'], 'timeout', '-k', '10', str(timeout_seconds), 'python3', str(Path(cfg['launcher_container']).with_name('rank_cache_equivalence_probe.py')), str(rank), str(die), str(port)]
                elif args.snapshot_probe:
                    argv += [cfg['container'], 'timeout', '-k', '10', str(timeout_seconds), 'python3', str(Path(cfg['launcher_container']).with_name('rank_snapshot_probe.py')), str(rank), str(die), str(port)]
                elif args.reference_probe:
                    argv += [cfg['container'], 'timeout', '-k', '10', str(timeout_seconds), 'python3', str(Path(cfg['launcher_container']).with_name('reference_probe.py')), str(rank), str(die), str(port)]
                else:
                    argv += [cfg['container'], 'timeout', '-k', '10', str(timeout_seconds), 'python3', cfg['launcher_container'], 'rank', '--rank', str(rank), '--die', str(die), '--port', str(port)]
                proc = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT)
                workers.append((item, proc, log))
                ownership['active_jobs'].append({'job_id': f'bounded-cli-{args.run_id}-rank{rank}', 'host': os.uname().nodename, 'physical_die': die, 'rank': rank, 'docker_exec_pid': proc.pid, 'started_at': stamp, 'command': argv, 'log': str(run / f'rank{rank}.log'), 'stop_condition': f'{timeout_seconds}-second timeout, 180-second mapping barrier'})
                write(cfg['ownership'], ownership)
            deadline = time.monotonic() + 180
            while time.monotonic() < deadline and not all((run / f"rank{x['rank']}" / 'ready').exists() for x, _, _ in workers):
                if any(p.poll() is not None for _, p, _ in workers):
                    break
                time.sleep(.5)
            if len(workers) != world_size or not all((run / f"rank{x['rank']}" / 'ready').exists() for x, _, _ in workers):
                raise RuntimeError('worker failed before mapping barrier')
            top = subprocess.check_output(['docker', 'top', cfg['container'], '-eo', 'pid,ppid,args'], text=True)
            info = subprocess.check_output(['npu-smi', 'info'], text=True)
            (run / 'mapping.txt').write_text(top + '\n' + info)
            for item, proc, log in workers:
                rank, die = item['rank'], item['die']
                worker_name = 'rollback_probe.py' if args.rollback_probe else 'reference_probe.py' if args.reference_probe else 'rank_snapshot_probe.py' if args.snapshot_probe else 'rank_cache_equivalence_probe.py' if args.cache_equivalence_probe else 'worker.py'
                worker_args = [str(rank), str(die), str(port), 'bounded'] if args.rollback_probe else [str(rank), str(die), str(port)]
                rows = [line for line in top.splitlines() if len(line.split(None, 2)) == 3 and line.split(None, 2)[2].startswith('python3 ') and worker_name in line and line.split()[-len(worker_args):] == worker_args]
                if len(rows) != 1:
                    raise RuntimeError('worker PID mapping ambiguous')
                host_pid = int(rows[0].split()[0])
                if not re.search(r'\|\s*' + str(item['card']) + r'\s+' + str(item['chip']) + r'\s*\|\s*' + str(host_pid) + r'\s*\|', info):
                    raise RuntimeError('worker PID-to-die mapping mismatch')
                next(x for x in ownership['active_jobs'] if x['job_id'] == f'bounded-cli-{args.run_id}-rank{rank}')['host_worker_pid'] = host_pid
            write(cfg['ownership'], ownership)
            for item, _, _ in workers:
                (run / f"rank{item['rank']}" / 'go').write_text('host PID/die mapping verified\n')
        except BaseException as error:
            fatal = repr(error)
        finally:
            results = []
            deadline = time.monotonic() + timeout_seconds + 60
            for item, proc, log in workers:
                try:
                    code = proc.wait(timeout=max(0, deadline - time.monotonic()))
                except subprocess.TimeoutExpired:
                    code = None
                log.close()
                ownership = read(cfg['ownership'])
                if code is not None:
                    job_id = f"bounded-cli-{args.run_id}-rank{item['rank']}"
                    ownership['active_jobs'] = [x for x in ownership['active_jobs'] if x['job_id'] != job_id]
                    ownership.setdefault('completed_jobs', []).append({'job_id': job_id, 'exit_code': code})
                    write(cfg['ownership'], ownership)
                results.append({'rank': item['rank'], 'die': item['die'], 'docker_exec_pid': proc.pid, 'exit_code': code})
            errors = []
            for rank in range(world_size):
                try:
                    x = read(run / f'rank{rank}' / 'result.json')
                    if args.rollback_probe:
                        passed = (x['status'] == 'passed' and x['rank_commit_agreement_enabled'] and x['legacy_stale_coordinated'] and x['legacy_stale_tokens_seen'] == 3 and len(x['legacy_stale_state_equal']) == 45 and all(x['legacy_stale_state_equal'].values()) and x['failure_coordinated'] and x['failure_tokens_seen'] == 3 and x['nonzero_kda'] and all(x['prefix_logits_equal']) and len(x['failure_state_equal']) == 45 and all(x['failure_state_equal'].values()) and x['rank_input_mismatch_coordinated'] and x['rank_input_mismatch_before_embedding'] and x['rank_input_mismatch_tokens_seen'] == 3 and len(x['rank_input_mismatch_state_equal']) == 45 and all(x['rank_input_mismatch_state_equal'].values()) and x['retry_logits_equal'] and len(x['retry_state_equal']) == 45 and all(x['retry_state_equal'].values()) and x['request_owners_remaining'] == 0 and x['destroyed'])
                    elif args.cache_equivalence_probe:
                        passed = (x['status'] == 'passed' and x['destroyed'] and x['rank_commit_agreement_enabled'] and x['semantic_pass'] and len(x['logit_equal']) > 0 and all(x['logit_equal']) and x['generated_equal'] and x['prompt_argmax_equal'] and x['cache_after_close']['closed'] and x['cache_after_close']['used_bytes'] == 0 and not x['cache_after_close']['active_lease'] and x['stop_set_reached'] and not x['cap_reached'])
                    elif args.snapshot_probe:
                        passed = (x['status'] == 'passed' and x['destroyed'] and x['rank_commit_agreement_enabled'] and x['archive_schema_version'] == 2 and x['baseline_tokens_equal'] and x['restore_generated_equal'] and len(x['restore_logits_equal']) > 0 and all(x['restore_logits_equal']) and len(x['restore_state_equal']) == 45 and all(x['restore_state_equal'].values()) and x['owners_closed'] and x['stop_set_reached'] and not x['cap_reached'])
                    elif args.reference_probe:
                        passed = (x['status'] == 'passed' and bool(x['checks']) and all(x['checks'].values()) and not x['cap_reached'])
                    else:
                        passed = (all(x[k] for k in ('rank_commit_agreement_enabled', 'semantic_pass', 'rank_agreement', 'stop_set_reached', 'destroyed')) and x['status'] == 'passed' and not x['cap_reached'])
                        if args.experimental_layer_major_kda:
                            passed = passed and x['experimental_layer_major_kda'] and x['chunk_layers'] == ([0,1,2,4] if args.experimental_sparse_kda_layer4 else [0,1,2]) and sum(x['prefill_groups']) == len(x['prompt_ids']) and all(1 <= n <= (40 if args.experimental_kda_prefill40 else 8) for n in x['prefill_groups']) and x.get('experimental_kda_prefill40',False)==args.experimental_kda_prefill40
                    if not passed:
                        errors.append(f'rank{rank} semantic failure')
                except (OSError, ValueError, KeyError, TypeError) as error:
                    errors.append(f'rank{rank}: {error!r}')
            if fatal:
                errors.append('manager failure: ' + fatal)
            if len(results) != world_size or any(item['exit_code'] != 0 for item in results):
                errors.append('worker completion failure or unknown exit status')
            summary = {'semantic_status': 'pass' if not errors else 'fail', 'semantic_errors': errors, 'batch_id': 'cli-' + args.run_id, 'started_at': stamp, 'finished_at': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'port': port, 'fatal': fatal, 'workers': results}
            write(run / 'summary.json', summary)
            print(json.dumps(summary), flush=True)
        if fatal or len(results) != world_size or any(x['exit_code'] != 0 for x in results) or errors:
            raise SystemExit(1)


if __name__ == '__main__':
    main()
