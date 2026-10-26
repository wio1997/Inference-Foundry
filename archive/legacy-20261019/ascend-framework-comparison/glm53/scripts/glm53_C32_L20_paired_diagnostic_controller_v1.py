"""L20 DRAFT detached coordinator/controller; no execution without a frozen review.

prepare  : local SHA/plan validation only (no SSH/HTTP/model).
launch   : refused while execution_authorized is false; otherwise starts a detached
           coordinator in a new session and returns without replaying it.
coordinate: internal; one precheck SSH at a time, then one long CPU43 session.
status   : read-only file/pid state; never touches a remote or replays a launcher.
"""
import argparse
import base64
import gzip
import hashlib
import json
import os
import re
import shlex
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOCKET = '/Users/wio/work/sglang-glm53-pd/ssh-control-l20-foreground-20261019'
APPROVAL = 'results/C32-L20-paired-diagnostic-actual-review-v1.json'
ACTIVE_LOCK = ROOT / 'runs' / 'C32-L20-paired-active.lock'
FRAME_CAP = 128 * 1024 ** 2
REPORT_CAP = 64 * 1024 ** 2


def load(name, relative, digest):
    path = ROOT / relative
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = load('inventory', 'scripts/inventory_readonly.py',
            'e95abea94f79ba05c770e17314b1cf53ad43284c7d6bb1b41afd7de3394a7777')
base.SSH = list(base.SSH)
base.SSH[2] = SOCKET


def atomic(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


PRECHECK = r'''import hashlib,json,pathlib,shutil,socket,subprocess,urllib.request
assert socket.gethostname()==HOSTNAME
def cmd(args,timeout=30):
 r=subprocess.run(args,text=True,capture_output=True,timeout=timeout);return r
inspect=cmd(['docker','inspect',CID,'--format','{{.Id}} {{.State.Running}} {{.Image}}']);assert inspect.returncode==0,inspect.stderr
fields=inspect.stdout.strip().split();assert fields[0]==CID and fields[1]=='true' and fields[2]==IMAGE
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
with opener.open(METRICS_URL,timeout=8) as response:
 status=response.status;raw=response.read(8*1024**2+1)
assert status==200 and len(raw)<=8*1024**2
lines=[x for x in raw.decode().splitlines() if x.startswith(('sglang:num_running_reqs{','sglang:num_queue_reqs{','vllm:num_requests_running{','vllm:num_requests_waiting{'))]
values=[float(x.rsplit(' ',1)[1]) for x in lines];assert values and all(x==0 for x in values)
npu=cmd(['npu-smi','info'],60);assert npu.returncode==0,npu.stderr
rows=[]
for line in npu.stdout.split('Process id',1)[-1].splitlines():
 cols=[x.strip() for x in line.split('|')]
 if len(cols)>=5 and len(cols[1].split())==2 and all(x.isdigit() for x in cols[1].split()) and cols[2].isdigit():
  pid=int(cols[2]);assert CID in (pathlib.Path('/proc')/str(pid)/'cgroup').read_text();rows.append(pid)
assert rows
if EXPECTED_NPU_CONTEXTS is not None:assert len(rows)==EXPECTED_NPU_CONTEXTS
profile_baseline=None
if ROLE=='D':
 if FRAMEWORK=='vllm':
  reader="import pathlib,json;p=pathlib.Path('/run/glm53/profile');t=sorted(p.glob('*/ASCEND_PROFILER_OUTPUT/trace_view.json'));print(json.dumps([{'relative':str(x.relative_to(p)),'bytes':x.stat().st_size,'mtime_ns':x.stat().st_mtime_ns} for x in t]))"
  p=cmd(['docker','exec',CID,'python3','-S','-c',reader],30);assert p.returncode==0,p.stderr
  traces=json.loads(p.stdout);profile_baseline={'existing_traces':traces};assert traces==[]
  p=cmd(['docker','exec',CID,'df','-B1','/run/glm53'],30);assert p.returncode==0,p.stderr
  profile_baseline['df_tail']=p.stdout.splitlines()[-1]
  assert int(p.stdout.splitlines()[-1].split()[3])>=24*1024**3
 else:
  root=pathlib.Path(SHARED_ROOT)/'profile'/RUN_ID;assert not root.exists()
  profile_baseline={'profile_dir':str(root),'exists':False,'free_bytes':shutil.disk_usage(pathlib.Path(SHARED_ROOT)).free}
  assert profile_baseline['free_bytes']>=24*1024**3
scanner="""import json,pathlib
patterns=API_PATTERNS
keys=ENV_KEYS
out=[]
for p in pathlib.Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:args=(p/'cmdline').read_bytes().replace(b'\\0',b' ').decode(errors='replace')
 except OSError:continue
 if not any(x in args for x in patterns):continue
 try:raw=(p/'environ').read_bytes()
 except OSError:continue
 env=dict(x.split(b'=',1) for x in raw.split(b'\\0') if b'=' in x)
 out.append({'pid':int(p.name),'cmdline':args[:600],'selected_environment':{k:env.get(k.encode(),b'').decode() for k in keys}})
print(json.dumps(out[:64]))"""
p=cmd(['docker','exec',CID,'python3','-S','-c',scanner],30);assert p.returncode==0,p.stderr
procs=json.loads(p.stdout);assert procs
if FRAMEWORK=='sglang':
 assert all(x['selected_environment'].get('SGLANG_PROFILE_V2') in ('','0','false','False') for x in procs)
print(json.dumps({'project_id':'glm53','node':NODE,'role':ROLE,'container_id':CID,'name':NAME,'image':IMAGE,'metrics_status':status,'metrics_sha256':hashlib.sha256(raw).hexdigest(),'idle_gauge_count':len(values),'owned_NPU_contexts':len(rows),'expected_owned_NPU_contexts':EXPECTED_NPU_CONTEXTS,'profile_baseline':profile_baseline,'initial_environment_processes':procs,'environment_interpretation_limit':'/proc/environ is the initial environment only; it does not establish an effective TASK_QUEUE_ENABLE value and supports no causal claim','model_requests':0,'SDK_init_calls':0}))
'''


CPU = r'''import base64,gzip,hashlib,json,pathlib,subprocess,socket,time
assert socket.gethostname()=='B7-23U-Atlas900RCKA3-43'
root=pathlib.Path(CPU_ROOT);assert not root.exists();root.mkdir(parents=True)
archive=root/'native-export';archive.mkdir()
image=CPU_IMAGE
p=subprocess.run(['docker','image','inspect',image,'--format','{{.Id}}'],capture_output=True,text=True,timeout=20);assert p.returncode==0 and p.stdout.strip()==image
for name,item in FILES.items():
 assert pathlib.Path(name).name==name
 raw=base64.b64decode(item['base64'],validate=True);assert hashlib.sha256(raw).hexdigest()==item['sha256'];(root/name).write_bytes(raw)
prior=pathlib.Path(PRIOR_REPORT_REMOTE);assert hashlib.sha256(prior.read_bytes()).hexdigest()==PRIOR_REPORT_SHA
dataset=pathlib.Path(DATASET);assert hashlib.sha256((dataset/'manifest.json').read_bytes()).hexdigest()==CLIENT['input_manifest_sha256']
raw=(json.dumps(CLIENT,indent=2)+'\n').encode();(root/'client-settings.json').write_bytes(raw);settings_sha=hashlib.sha256(raw).hexdigest()
output=root/'output';output.mkdir()
args=['docker','run','--rm','--pull=never','--network=host','--pids-limit=512','--memory=16g','--cpus=4','--label','project_id=glm53','--label','run_id='+RUN_ID,'--name',CONTAINER_NAME,'--cidfile',str(root/'client.cid'),'-v',str(root)+':/run/glm53','-v',str(dataset)+':/diagnostic-inputs:ro','-v',str(prior)+':/prior-report.json:ro','-v',str(root/'prior-acceptance.json')+':/prior-acceptance.json:ro','-v','/data/wio/aisbench-offline-43-20261014:/work:ro','--workdir','/run/glm53']
environment={'PYTHONDONTWRITEBYTECODE':'1','PYTHONNOUSERSITE':'1','PYTHONPATH':'/run/glm53:/work/benchmark','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','TOKENIZERS_PARALLELISM':'false','CUDA_VISIBLE_DEVICES':'','ASCEND_RT_VISIBLE_DEVICES':'','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','HF_HOME':'/run/glm53/hf','XDG_CACHE_HOME':'/run/glm53/cache','MPLCONFIGDIR':'/run/glm53/mpl','TORCH_HOME':'/run/glm53/torch','NLTK_DATA':'/work/nltk_data','TIKTOKEN_CACHE_DIR':'/work/tiktoken_cache'}
for k,v in environment.items():args+=['-e',k+'='+v]
args+=['--entrypoint','/work/venv/bin/python',image,'/run/glm53/glm53_C32_L20_paired_diagnostic_client_v1.py','--settings','/run/glm53/client-settings.json','--sha256',settings_sha]
state={'project_id':'glm53','framework':CLIENT['framework'],'run_id':RUN_ID,'client_rc':None,'host_wait_timeout':False,'model_requests':'COUNT_FROM_ACTUAL_GATEWAY_ONLY'}
last=None
with (root/'client-stdout.txt').open('x') as stdout,(root/'client-stderr.txt').open('x') as stderr:
 process=subprocess.Popen(args,stdout=stdout,stderr=stderr,start_new_session=True)
 deadline=time.monotonic()+DEADLINE_SECONDS
 while process.poll() is None:
  progress=output/'diagnostic-progress.json'
  if progress.is_file():
   try:
    value=json.loads(progress.read_text());rows=value.get('rows',[])
    compact={'project_id':'glm53','framework':CLIENT['framework'],'run_id':RUN_ID,'phase':value.get('phase'),'rows':len(rows),'count_and_terminal_successes':sum(bool(x.get('count_and_terminal_pass')) for x in rows),'prefill_count_successes':sum(bool(x.get('prefill_count_pass')) for x in rows),'raw_cache_observation_passes':sum(bool(x.get('cache_observation',{}).get('pass')) for x in rows)}
   except (OSError,ValueError):compact=None
   if compact!=last:
    print('GLM53_L20_PROGRESS='+json.dumps(compact),flush=True);last=compact
  if time.monotonic()>deadline:
   state['host_wait_timeout']=True;print('GLM53_L20_TIMEOUT='+json.dumps({'client_pid':process.pid,'preserved':True}),flush=True);break
  time.sleep(.2)
if not state['host_wait_timeout']:state['client_rc']=process.wait()
artifacts={}
for path in sorted(root.rglob('*')):
 if path.is_file():raw=path.read_bytes();artifacts[str(path.relative_to(root))]={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
message={'state':state,'artifacts':artifacts}
report=output/'diagnostic-result.json'
if report.is_file():
 raw=report.read_bytes();message['report_sha256']=hashlib.sha256(raw).hexdigest()
 if len(raw)<=REPORT_CAP:message['report_gzip_base64']=base64.b64encode(gzip.compress(raw,mtime=0)).decode()
 else:message['report_transport_limitation']='Full report exceeds64MiB; retained on CPU43, separate complete audit required'
else:message['report_transport_limitation']='No diagnostic-result.json produced'
print('GLM53_L20_DONE='+json.dumps(message),flush=True)
'''


def render(source, values):
    pattern = r'\b(?:' + '|'.join(re.escape(key) for key in values) + r')\b'
    code = re.sub(pattern, lambda match: repr(values[match[0]]), source)
    import ast
    ast.parse(code)
    return code


def invoke(node, code, timeout):
    started = time.monotonic()
    pod, hostname = base.NODES[node]
    mapped = base.invoke(['kubectl get pod -n default ' + shlex.quote(pod) +
                          " -o jsonpath='{.status.hostIP}'"], timeout=min(30, timeout))
    assert mapped['rc'] == 0 and mapped['stdout'] == f'172.16.12.{node}'
    code = 'import socket\nassert socket.gethostname()==' + repr(hostname) + '\n' + code
    import ast
    ast.parse(code)
    command = shlex.join(['kubectl', 'exec', '-i', '-n', 'default', pod, '-c', 'hostctl',
                          '--', 'nsenter', '-t', '1', '-m', '--', 'python3', '-'])
    remaining = timeout - (time.monotonic() - started)
    assert remaining > 0, 'node read-only precheck budget exhausted'
    return base.invoke([command], code, timeout=remaining)


def settings_and_sha(path, digest):
    raw = (ROOT / path).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == digest
    settings = json.loads(raw)
    assert settings['project_id'] == 'glm53'
    assert settings['diagnostic_only'] is True
    assert settings['performance_certificate_allowed'] is False
    return settings


def verify_local_prerequisites(settings):
    for relative, digest in settings['prerequisite_sha256'].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == digest, relative
    proof = json.loads((ROOT / settings['input_manifest_proof']).read_bytes())
    assert proof['manifest_sha256'] == settings['input_manifest_sha256']
    client = settings['client_settings']
    assert client['framework'] == settings['framework']
    assert client['run_id'] == settings['run_id']
    assert client['cache_namespace'] == settings['namespace']
    assert client['prior_report_sha256'] == settings['prior_report_sha256']
    assert client['prior_acceptance_sha256'] == settings['prior_acceptance_sha256']
    assert client['profile_settings'] == settings['profile_settings']
    assert client['execution_authorized'] == settings['execution_authorized']
    for name, relative in settings['files'].items():
        raw = (ROOT / relative).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == settings['files_sha256'][name], name


def prepare(path, digest):
    settings = settings_and_sha(path, digest)
    verify_local_prerequisites(settings)
    plan = {'project_id': 'glm53', 'mode': 'prepare_only', 'cohort': settings['cohort'],
            'status': settings['status'], 'execution_authorized': settings['execution_authorized'],
            'run_id': settings['run_id'], 'framework': settings['framework'],
            'namespace': settings['namespace'], 'prior_report_sha256': settings['prior_report_sha256'],
            'input_manifest_sha256': settings['input_manifest_sha256'],
            'logical_requests': settings['logical_requests'],
            'planned_backend_HTTP_attempts': settings['budget']['backend_HTTP_attempts'],
            'planned_all_successful_backend_output_tokens_if_all_pass':
                settings['budget']['all_successful_backend_output_tokens_if_all_pass'],
            'profile_scopes': settings['profile_scopes'],
            'precheck_nodes': [item['node'] for item in settings['precheck_nodes']],
            'model_requests': 0, 'SDK_init_calls': 0, 'remote_touched': False}
    print(json.dumps(plan, indent=2))
    return plan


def acquire_active_lock(cohort):
    if ACTIVE_LOCK.exists():
        owner = json.loads(ACTIVE_LOCK.read_text())
        if owner.get('released') is True:
            ACTIVE_LOCK.unlink()
        else:
            pid = owner.get('pid')
            alive = False
            if isinstance(pid, int):
                try:
                    os.kill(pid, 0)
                    alive = True
                except OSError:
                    alive = False
            raise RuntimeError(f'active cohort lock held: {owner}; alive={alive}; '
                               'inspect status, do not replay')
    value = {'project_id': 'glm53', 'cohort': cohort, 'pid': os.getpid(),
             'started_monotonic_ns': time.monotonic_ns(), 'released': False}
    descriptor = os.open(str(ACTIVE_LOCK), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, 'w') as file:
        json.dump(value, file)
    return value


def release_active_lock(cohort):
    if not ACTIVE_LOCK.exists():
        return
    owner = json.loads(ACTIVE_LOCK.read_text())
    if owner.get('cohort') != cohort:
        return
    owner['released'] = True
    owner['released_monotonic_ns'] = time.monotonic_ns()
    atomic(ACTIVE_LOCK, owner)


def authorize(settings, digest, review_sha):
    # The approval digest is a launch argument, never embedded in the settings
    # whose digest the approval pins. This avoids a self-referential SHA guard.
    raw = (ROOT / APPROVAL).read_bytes()
    assert review_sha and hashlib.sha256(raw).hexdigest() == review_sha
    approval = json.loads(raw)
    assert approval['decision'] == 'APPROVED_BOUNDED_PAIRED_C32_DIAGNOSTIC'
    assert approval['controller_sha256'] == hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    assert approval['settings_sha256'][settings['framework']] == digest
    assert approval['run_ids'][settings['framework']] == settings['run_id']


def launch(path, digest, cohort, review_sha):
    settings = settings_and_sha(path, digest)
    assert settings['cohort'] == cohort
    assert settings['execution_authorized'] is True, 'DRAFT is not authorized'
    authorize(settings, digest, review_sha)
    verify_local_prerequisites(settings)
    run = ROOT / 'runs' / settings['run_id']
    acquire_active_lock(cohort)
    try:
        run.mkdir(parents=True, exist_ok=False)
        (run / 'execution-once.lock').write_text(str(os.getpid()))
        stdout = (run / 'coordinator-stdout.txt').open('x')
        stderr = (run / 'coordinator-stderr.txt').open('x')
        process = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), 'coordinate',
             '--cohort', cohort, '--settings', path, '--sha256', digest,
             '--review-sha256', review_sha],
            stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr, cwd=str(ROOT),
            start_new_session=True, close_fds=True)
        (run / 'coordinator.pid').write_text(str(process.pid))
        print(json.dumps({'project_id': 'glm53', 'cohort': cohort,
                          'coordinator_pid': process.pid, 'run_id': settings['run_id'],
                          'detached': True, 'replay': False}))
    except BaseException:
        release_active_lock(cohort)
        raise


def coordinate(path, digest, cohort, review_sha):
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    settings = settings_and_sha(path, digest)
    assert settings['cohort'] == cohort
    assert settings['execution_authorized'] is True
    authorize(settings, digest, review_sha)
    run = ROOT / 'runs' / settings['run_id']
    assert run.is_dir()
    state = {'project_id': 'glm53', 'cohort': cohort, 'framework': settings['framework'],
             'run_id': settings['run_id'], 'phase': 'PRECHECK', 'prechecks': {},
             'cpu_connection_pid': None, 'cpu_connection_rc': None,
             'done_received': False, 'eof_without_done': False}
    try:
        precheck_deadline = time.monotonic() + settings['precheck_deadline_seconds']
        for item in settings['precheck_nodes']:
            code = render(PRECHECK, {
                'HOSTNAME': base.NODES[item['node']][1], 'NODE': item['node'],
                'CID': item['container_id'], 'NAME': item['container_name'],
                'IMAGE': settings['model_image'], 'ROLE': item['role'],
                'FRAMEWORK': settings['framework'], 'METRICS_URL': item['metrics_url'],
                'SHARED_ROOT': item['shared_root'], 'RUN_ID': settings['run_id'],
                'EXPECTED_NPU_CONTEXTS': item['expected_owned_NPU_contexts'],
                'API_PATTERNS': settings['api_patterns'],
                'ENV_KEYS': settings['environment_keys']})
            remaining = precheck_deadline - time.monotonic()
            assert remaining > 0, 'total precheck budget exhausted'
            result = invoke(item['node'], code, min(90, remaining))
            (run / f"precheck-{item['node']}.json").write_text(
                json.dumps(result, indent=2) + '\n')
            state['prechecks'][str(item['node'])] = result['rc']
            assert result['rc'] == 0, (item['node'], result['stderr'][-2000:])
        state['phase'] = 'CPU_CLIENT'
        payload = {}
        for name, relative in settings['files'].items():
            raw = (ROOT / relative).read_bytes()
            assert hashlib.sha256(raw).hexdigest() == settings['files_sha256'][name]
            payload[name] = {'base64': base64.b64encode(raw).decode(),
                             'sha256': hashlib.sha256(raw).hexdigest()}
        code = render(CPU, {
            'CPU_ROOT': f"/data/wio/inference-foundry/glm53/runs/{settings['run_id']}",
            'RUN_ID': settings['run_id'], 'FILES': payload,
            'CLIENT': settings['client_settings'], 'CPU_IMAGE': settings['cpu_image'],
            'PRIOR_REPORT_REMOTE': settings['prior_report_remote'],
            'PRIOR_REPORT_SHA': settings['prior_report_sha256'],
            'DATASET': settings['dataset_remote'],
            'CONTAINER_NAME': settings['cpu_container_name'],
            'DEADLINE_SECONDS': settings['client_deadline_seconds'],
            'REPORT_CAP': REPORT_CAP})
        pod = base.NODES[43][0]
        mapped = base.invoke(['kubectl get pod -n default ' + shlex.quote(pod) +
                              " -o jsonpath='{.status.hostIP}'"], timeout=30)
        (run / 'mapping-43.json').write_text(json.dumps(mapped) + '\n')
        assert mapped['rc'] == 0 and mapped['stdout'] == '172.16.12.43'
        command = shlex.join(['kubectl', 'exec', '-i', '-n', 'default', pod, '-c', 'hostctl',
                              '--', 'nsenter', '-t', '1', '-m', '--', 'python3', '-'])
        with (run / 'coordinator-cpu-stdout.txt').open('x') as stdout, \
                (run / 'coordinator-cpu-stderr.txt').open('x') as stderr:
            process = subprocess.Popen(base.SSH + [command], stdin=subprocess.PIPE,
                                       stdout=subprocess.PIPE, stderr=stderr,
                                       text=True, encoding='utf-8', start_new_session=True)
            state['cpu_connection_pid'] = process.pid
            process.stdin.write(code + '\n')
            process.stdin.close()
            other = 0
            while True:
                line = process.stdout.readline(FRAME_CAP + 1)
                if not line:
                    break
                stdout.write(line)
                stdout.flush()
                assert len(line) <= FRAME_CAP and line.endswith('\n')
                if line.startswith('GLM53_L20_PROGRESS='):
                    progress = json.loads(line.split('=', 1)[1])
                    atomic(run / 'controller-progress.json', progress)
                elif line.startswith('GLM53_L20_TIMEOUT='):
                    atomic(run / 'controller-timeout.json', json.loads(line.split('=', 1)[1]))
                elif line.startswith('GLM53_L20_DONE='):
                    done = json.loads(line.split('=', 1)[1])
                    state['done_received'] = True
                    if 'report_gzip_base64' in done:
                        packed = base64.b64decode(done['report_gzip_base64'], validate=True)
                        with gzip.GzipFile(fileobj=__import__('io').BytesIO(packed)) as stream:
                            report_raw = stream.read(REPORT_CAP + 1)
                        assert len(report_raw) <= REPORT_CAP
                        assert hashlib.sha256(report_raw).hexdigest() == done['report_sha256']
                        (run / 'diagnostic-result.remote.json').write_bytes(report_raw)
                        state['report_sha256'] = done['report_sha256']
                    compact = {key: value for key, value in done.items()
                               if key != 'report_gzip_base64'}
                    (run / 'controller-done.json').write_text(
                        json.dumps(compact, indent=2) + '\n')
                    state['client_rc'] = (done.get('state') or {}).get('client_rc')
                elif line.strip():
                    if other < 100:
                        state.setdefault('other_lines', []).append(line[:1000])
                        other += 1
            state['cpu_connection_rc'] = process.wait(timeout=60)
        state['phase'] = 'DONE' if state['done_received'] else 'INTERRUPTED_PRESERVE'
        state['eof_without_done'] = not state['done_received']
    except BaseException as error:
        state['phase'] = 'FAILED_PRESERVE'
        state['error'] = repr(error)
        raise
    finally:
        atomic(run / 'controller-rpc-state.json', state)
        if state['done_received']:
            release_active_lock(cohort)
    return state


def status(cohort):
    runs = sorted((ROOT / 'runs').glob('*L20-*'))
    result = {'project_id': 'glm53', 'cohort': cohort, 'mode': 'status_read_only',
              'active_lock': json.loads(ACTIVE_LOCK.read_text()) if ACTIVE_LOCK.exists() else None,
              'runs': {}}
    for run in runs:
        if not run.is_dir():
            continue
        entry = {}
        for name in ['execution-once.lock', 'coordinator.pid', 'controller-progress.json',
                     'controller-rpc-state.json', 'controller-done.json', 'receipt.json']:
            path = run / name
            if path.is_file():
                raw = path.read_bytes()
                entry[name] = {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
                if name.endswith('.json') and len(raw) <= 1024 * 1024:
                    entry[name]['value'] = json.loads(raw)
        result['runs'][run.name] = entry
    print(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare', 'launch', 'coordinate', 'status'])
    parser.add_argument('--cohort', choices=['sglang', 'vllm'])
    parser.add_argument('--settings')
    parser.add_argument('--sha256')
    parser.add_argument('--review-sha256')
    args = parser.parse_args()
    if args.mode in ('launch', 'coordinate'):
        assert Path(SOCKET).is_socket(), 'foreground authenticated control socket missing'
    if args.mode == 'status':
        assert args.cohort
        status(args.cohort)
        return
    assert args.settings and args.sha256
    if args.mode == 'prepare':
        prepare(args.settings, args.sha256)
    elif args.mode == 'launch':
        assert args.cohort
        launch(args.settings, args.sha256, args.cohort, args.review_sha256)
    else:
        assert args.cohort
        coordinate(args.settings, args.sha256, args.cohort, args.review_sha256)


if __name__ == '__main__':
    main()
