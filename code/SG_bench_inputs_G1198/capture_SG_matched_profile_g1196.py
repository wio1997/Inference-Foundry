"""One bounded SGLang P/D profile workload; never a scored benchmark.

Run on the decode host only after the parent verifies both owned profiling
services, the canonical workload guard, warmup/compiler evidence and PD gates.
Every start/stop attempt is recorded. HTTP success does not prove rank coverage.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import datetime
import hashlib
import json
from pathlib import Path
import socket
import signal
import subprocess
import sys
import time
import urllib.request


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def http(url, body=None, timeout=30):
    request = urllib.request.Request(url, data=None if body is None else json.dumps(body).encode(),
                                     headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        assert response.status == 200
        return response.read().decode()


def host_path(container, requested):
    requested = Path(requested)
    matches = [item for item in container['Mounts']
               if Path(item['Destination']) == requested or Path(item['Destination']) in requested.parents]
    assert matches
    best = max(matches, key=lambda item: len(Path(item['Destination']).parts))
    return Path(best['Source']) / requested.relative_to(best['Destination'])


def main(args):
    assert socket.gethostname() == '<DECODE_HOSTNAME>'
    task = Path('/PRIVATE_TASK_ROOT')
    assert args.output.is_absolute() and not args.output.exists()
    assert args.prefix and Path(args.prefix).name==args.prefix and args.prefix not in ('.','..')
    gate = json.loads(args.gate.read_text())
    assert gate['engine'] == 'sglang' and gate['ready'] and gate['pd_transfer_verified']
    assert gate['profile_epoch'] == 1178 and gate['workload_guard_actual_verified']
    assert gate['all8_D_target_and_draft_graph_captures_verified'] and gate['P_D_last_two_warmups_compile_verified']
    decode = json.loads((task / 'sg_pd_g1178/state.json').read_text())
    assert decode['generation'] == 1178 and decode['container_id'] == gate['roles']['decode']['container_id']
    assert str(task / 'sg_pd_g1178/node_controller.py').encode() in Path('/proc', str(decode['pid']), 'cmdline').read_bytes()
    controller=gate['roles']['decode']
    assert decode['pid']==controller['controller_pid']
    age=time.time()-datetime.datetime.fromisoformat(gate['verified_at']).timestamp()
    assert -60 <= age < 600
    smoke=gate['PD_runtime_proof']
    assert smoke['math1']=='91' and smoke['completed']==3
    assert all(v>0 for v in smoke['transfer_count_delta_by_P_rank'].values())
    assert all(v>0 for v in smoke['D_full_graph_counter_delta_by_rank'].values())
    assert len(smoke['transfer_count_delta_by_P_rank'])==len(smoke['D_full_graph_counter_delta_by_rank'])==8
    assert gate['profile_cohort_distinct_from_formal_and_warmup'] and gate['profiler_export_roots_checked']
    prior=task/'client_ds41_g1122/runs/sglang'/gate['verified_32k_run_id']
    assert json.loads((prior/'state.json').read_text())['phase']=='all_phases_terminal'
    assert json.loads((prior/'audit_results.json').read_text())['all_required_raw_metrics_and_usage_validated']
    client = json.loads(subprocess.check_output(['docker', 'inspect', 'yxy-ds41-aisbench-g1122']))[0]
    assert client['Id'] == 'd8593a995ef7192cfd5fd9e696d652e24b3d02d77253d2b96a3415a4097f9090'
    assert client['Image'] == 'sha256:05358d58d23f2679c71e48d93797d5d2a28d72eedc7fb4900479c89955465d76'
    assert client['State']['Running'] and not client['HostConfig']['Devices']
    top = subprocess.check_output(['docker', 'top', client['Id'], '-eo', 'pid,args'], text=True)
    assert all('sleep 72000' in line for line in top.splitlines()[1:]), 'Current client job must finish before profiling'
    frozen = json.loads(args.inputs.read_text())
    assert sha(args.inputs)=='308d46f5ae05d7d30b8af9b4711b24582fe4b8a0c6fdc76f32473720f7acf4f4'
    assert len(frozen['records'])==16 and frozen['dataset_sha256']=='5cf5dbc8174409a627c8a3439f4b07e0ca99d53ac19c187ed4988eff354d0c00'
    dataset = host_path(client, frozen['dataset_path'])
    assert sha(dataset) == frozen['dataset_sha256']
    rows = [json.loads(line) for line in dataset.read_text().splitlines()]
    assert len(rows) == 16
    for row, expected in zip(rows, frozen['records']):
        assert sha_text(row['question']) == expected['question_sha256'] and row['max_out_len'] == 1024
    config = frozen['configurations']['sglang']
    assert sha(host_path(client, config['path'])) == config['sha256']
    claim = task/'client_ds41_g1122/profile_submitted_sglang_epoch1178.json'
    with claim.open('x') as stream:
        json.dump({'dataset_sha256': frozen['dataset_sha256'], 'output': str(args.output),
                   'at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   'scope': 'One SGLang diagnostic submission; retain even on failure. Not a formal benchmark claim.'}, stream, indent=2)
    # The output must live in the existing CPU client's writable child bind.
    container_output = '/task/client_ds41_g1122/' + args.output.relative_to(task / 'client_ds41_g1122').as_posix()
    assert host_path(client, container_output) == args.output
    args.output.mkdir()
    started = time.monotonic()
    def state(phase, **more):
        value = {'phase': phase, 'pid': __import__('os').getpid(),
                 'at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 'elapsed_s': time.monotonic() - started, **more}
        save(args.output / 'state.json', value)
        print(json.dumps(value), flush=True)
    def snapshots(tag, require_idle=False):
        result = {}
        for role, item in gate['roles'].items():
            assert json.loads(http(item['base_url'] + '/v1/models'))['data'][0]['id'] == gate['model_id']
            http(item['base_url'] + '/ready')
            metrics = http(item['base_url'] + '/metrics')
            if require_idle:
                import re
                for name in ('sglang:num_running_reqs', 'sglang:num_queue_reqs'):
                    values = [float(x) for x in re.findall(r'^' + re.escape(name) + r'(?:\{[^\n]*\})?\s+([0-9.eE+\-]+)', metrics, re.M)]
                    assert values and sum(values) == 0, (role, name, values)
            result[role] = metrics
        save(args.output / (tag + '_metrics.json'), result)
    def control(role, action, body, timeout):
        record = {'role': role, 'action': action, 'at': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'attempts': 1}
        try:
            record.update(ok=True, body=http(gate['roles'][role]['base_url'] + '/' + action, body, timeout))
        except Exception as error:
            record.update(ok=False, error=repr(error))
        save(args.output / (role + '_' + action + '.json'), record)
        return record
    save(args.output / 'manifest.json', {'gate': gate, 'gate_sha256': sha(args.gate),
         'inputs_manifest_sha256': sha(args.inputs), 'dataset_sha256': frozen['dataset_sha256'],
         'config': config, 'CPU_client_id': client['Id'], 'source_sha256': sha(__file__),
         'scope': 'Unscored16/C16/L32768/O1024 diagnostic. Preserve original formal ledgers; HTTP200 never proves trace coverage.'})
    workload = None
    error = None
    started_roles = []
    def interrupted(signum, _frame):
        raise RuntimeError('Profile driver interrupted by signal ' + str(signum))
    signal.signal(signal.SIGTERM, interrupted)
    try:
        snapshots('before', require_idle=True)
        assert json.loads(http(gate['proxy_url'] + '/v1/models'))['data'][0]['id'] == gate['model_id']
        state('starting_both_role_profilers')
        with ThreadPoolExecutor(max_workers=2) as pool:
            def begin(role):
                spec = {'output_dir':'/task/sg_pd_g1178/profiles/'+args.prefix+'/'+role,
                        'activities':['CPU','GPU'],'with_stack':False,'record_shapes':True,
                        'profile_by_stage':False,'merge_profiles':False,'detailed_annotations':True,
                        'profile_id':args.prefix+'-'+role,'profile_prefix':args.prefix+'-'+role}
                save(args.output/(role+'_start_profile_body.json'),spec)
                return control(role,'start_profile',spec,120)
            start_results=list(pool.map(begin,gate['roles']))
        started_roles=[v['role'] for v in start_results if v['ok']]
        assert set(started_roles)==set(gate['roles']),start_results
        command = ['docker', 'exec', '-w', container_output, '-e', 'BFCL_PROJECT_ROOT=/task/client_ds41_g1122/bfcl',
                   client['Id'], 'timeout', '--signal=TERM', '--kill-after=30', str(args.workload_timeout),
                   'ais_bench', config['path'], '--mode', 'perf', '--num-warmups', '0', '--num-prompts', '16',
                   '--max-num-workers', '1', '--work-dir', container_output + '/aisbench']
        with (args.output / 'aisbench.log').open('x') as log:
            workload = subprocess.Popen(command, cwd=task, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            save(args.output / 'actual_workload_handle.json', {'pid': workload.pid, 'argv': command,
                 'at': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'inner_timeout_s': args.workload_timeout,
                 'outer_wait_s': args.workload_timeout + 90, 'cancel_scope': 'Only this CPU process group is bounded by inner timeout; no service reset/kill.'})
            state('profile_workload_running', docker_exec_pid=workload.pid)
            code = workload.wait(timeout=args.workload_timeout + 90)
            assert code == 0, ('AISBench failed; preserve all original evidence', code)
    except BaseException as caught:
        error = repr(caught)
    finally:
        if started_roles:
            state('stopping_both_role_profilers', previous_error=error)
            with ThreadPoolExecutor(max_workers=2) as pool:
                stopped = list(pool.map(lambda role: control(role, 'stop_profile', {}, 900), started_roles))
            if not all(item['ok'] for item in stopped):
                error = error or 'One or both profiler stops not confirmed; inspect existing exporters, never relaunch blindly'
    if error:
        state('failed_requires_review', error=error,
              CPU_exec_still_live=workload is not None and workload.poll() is None,
              all_rank_profile_accepted=False, resources_released=False)
        raise RuntimeError(error)
    snapshots('after', require_idle=True)
    sys.path.insert(0, str(task / 'client_ds41_g1122/benchmark_pipeline_SG_g1183'))
    from audit_aisbench import audit_phase
    report = audit_phase(args.output / 'aisbench', 32768, requests=16,
                         usage_contract=Path(gate['usage_contract']))
    assert report['fully_validated']
    save(args.output / 'workload_raw_audit.json', report)
    state('profile_workload_complete_pending_allrank_review', actual_requests=16,
          all_rank_profile_accepted=False, resources_released=False,
          scope='Parent must inspect all8P/all8D device artifacts, timing/rank coverage, actual logicalwork and compiler/cache deltas. No gain or globalquality claim.')


def sha_text(text):
    return hashlib.sha256(text.encode()).hexdigest()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gate', type=Path, required=True)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prefix', required=True)
    parser.add_argument('--workload-timeout', type=int, default=480, choices=range(60, 901))
    main(parser.parse_args())
