#!/usr/bin/env python3
"""Launch-only start of the node33 TP16 W8A8C8 capacity/functionality diagnostic container.

Stages a bounded payload through the user's existing SSH multiplex socket and access pod into the
node33 host mount namespace, asserts the task reservation lease plus host preconditions (weights
receipt identity and per-path stat sizes without payload rehash, pinned image id, free loopback
port, absent container name), runs exactly one `docker run -d` of that pinned image id, and pulls
the receipt back for SHA verification. No fallback, no polling, no other container touched;
a started container is not model readiness.
"""
import argparse
import base64
import datetime
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys

SCHEMA = 'glm53-capacity-launch/v1'
PLAN_NAME = 'capacity-tp16-node33.json'
TOOLS_DIR = Path(__file__).resolve().parent

# Staged on the host first: creates the fresh run dir, stores plan/payload/setup and manifest,
# then runs the launch payload. Writes nothing if the run dir already exists.
BOOTSTRAP = '''import base64,datetime,hashlib,json,pathlib,subprocess,sys
setup=base64.b64decode(SETUP); s=json.loads(setup)
r=pathlib.Path(s['run_dir'])
assert not r.exists(),'run dir exists; refusing to overwrite evidence: '+str(r)
r.mkdir(parents=True)
def put(name,data,sha):
 assert hashlib.sha256(data).hexdigest()==sha,'staged file sha mismatch: '+name
 (r/name).write_bytes(data)
put('setup.json',setup,hashlib.sha256(setup).hexdigest())
put('plan.json',base64.b64decode(s['plan_b64']),s['plan_sha256'])
put('payload.py',base64.b64decode(s['payload_b64']),s['payload_sha256'])
put('start_capacity.py',base64.b64decode(s['source_script_b64']),s['source_script_sha256'])
(r/'manifest.json').write_text(json.dumps({'schema':s['schema'],'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_script':s['source_script'],'source_script_sha256':s['source_script_sha256'],'plan_sha256':s['plan_sha256'],'payload_sha256':s['payload_sha256'],'setup_sha256':hashlib.sha256(setup).hexdigest()},indent=2)+chr(10))
sys.exit(subprocess.call([sys.executable or 'python3',str(r/'payload.py')]))
'''
# Host-side launch payload: asserts preconditions from the plan, then exactly one docker run.
PAYLOAD = '''import datetime,hashlib,json,os,pathlib,socket,subprocess,sys
def utc(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def run(a,t=120): return subprocess.run(a,capture_output=True,text=True,timeout=t)
RUN=pathlib.Path(__file__).resolve().parent
plan=json.loads((RUN/'plan.json').read_text())
setup=json.loads((RUN/'setup.json').read_text())
names=('setup.json','plan.json','payload.py','start_capacity.py','manifest.json','context.json','launch.json')
def emit(status,**extra):
 o={'status':status,'run_dir':str(RUN),'model_ready_checked':False,'sha256':{n:hashlib.sha256((RUN/n).read_bytes()).hexdigest() for n in names if (RUN/n).is_file()}}
 o.update(extra); print(json.dumps(o))
ctx={'host_checked_utc':utc(),'scope':'launch only: preconditions asserted, one container started; model readiness is not checked',
 'client':{k:setup[k] for k in ('client_started_utc','source_script','source_script_sha256','plan_sha256','payload_sha256','git_sha','git_diff_sha256','lease')}}
launch=None
try:
 expected_host=plan.get('target_hostname','B5-5U-Atlas900RCKA3-33')
 assert expected_host in ('B4-33U-Atlas900RCKA3-32','B5-5U-Atlas900RCKA3-33')
 assert os.uname().nodename==expected_host,'Wrong host identity'
 assert plan['serve_argv'][:2]==['vllm','serve'],'serve argv does not start with vllm serve'
 tr=pathlib.Path(plan['remote']['transfer_root']); wm=plan['weights_manifest']
 receipts=json.loads((tr/wm['receipt']).read_text()); ready=json.loads((tr/wm['ready']).read_text())
 ident=hashlib.sha256(json.dumps(receipts,sort_keys=True).encode()).hexdigest()
 assert ident==wm['manifest_sha256']==ready['manifest_sha256'],'weights manifest identity mismatch: '+ident
 weights=pathlib.Path(plan['remote']['weights_dir']); bad=[]; total=0
 assert ready['all_file_payload_sha256_verified'] is True and ready['files']==196 and ready['bytes']==764240832702
 assert weights.is_dir() and '.incoming' not in str(weights.resolve()),'Incomplete weights path'
 for name,exp in receipts.items():
  assert not name.startswith('/') and '..' not in pathlib.PurePosixPath(name).parts,'unsafe receipt path '+name
  p=weights/name; total+=exp['bytes']
  if not p.is_file() or p.stat().st_size!=exp['bytes']: bad.append(name)
 assert not bad and len(receipts)==wm['file_count'] and total==wm['total_bytes'],'weights stat mismatch: %d bad %s'%(len(bad),bad[:5])
 ctx['weights']={'manifest_sha256':ident,'files':len(receipts),'total_bytes':total,'size_mismatch':bad,'payload_rehash':False,'weights_dir':str(weights),'transfer_root':str(tr)}
 ins=run(['docker','image','inspect',plan['image']['ref'],'--format','{{.Id}}']); image_id=ins.stdout.strip()
 assert ins.returncode==0 and image_id==plan['image']['id'],'image id mismatch: '+repr(image_id)+' '+ins.stderr.strip()
 ctx['image']={'ref':plan['image']['ref'],'id':image_id,'executed_by':'image id'}
 ex=run(['docker','inspect',plan['container']['name'],'--format','{{.Id}}'])
 assert ex.returncode!=0 and not ex.stdout.strip(),'container name already exists: '+plan['container']['name']
 s=socket.socket(); s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1); free=False
 try: s.bind((plan['container']['host_ip'],plan['container']['port'])); free=True
 except OSError: free=False
 finally: s.close()
 assert free,'port in use: %s:%d'%(plan['container']['host_ip'],plan['container']['port'])
 ctx['preconditions']={'container_name_absent':True,'port_free':True}
 devs=['/dev/davinci%d'%i for i in plan['devices']['davinci']]+['/dev/'+n for n in plan['devices']['shared']]
 for p in devs: assert pathlib.Path(p).exists(),'missing device node '+p
 ro=[p for p in plan['remote']['ro_paths_required'] if pathlib.Path(p).exists()]
 assert len(ro)==len(plan['remote']['ro_paths_required']),'missing required read-only mount(s)'
 absent=[p for p in plan['remote']['ro_paths_optional'] if not pathlib.Path(p).exists()]
 ro+=[p for p in plan['remote']['ro_paths_optional'] if pathlib.Path(p).exists()]
 ctx['mounts']={'weights_ro':plan['remote']['weights_dir'],'out_rw':str(RUN),'ro_extra':ro,'ro_absent_not_guessed':absent,'devices':devs}
 logrel=plan['remote']['ascend_log_path'][len(plan['remote']['out_mount']):].lstrip('/')
 assert logrel and not pathlib.PurePosixPath(logrel).is_absolute(),'unexpected ascend log path: '+plan['remote']['ascend_log_path']
 (RUN/logrel).mkdir(parents=True,exist_ok=True)
 argv=['docker','run']+plan['docker_argv']+['--name',plan['container']['name'],'-v',plan['remote']['weights_dir']+':'+plan['remote']['weights_mount']+':ro','-v',str(RUN)+':'+plan['remote']['out_mount']]
 for p in ro: argv+=['-v',p+':'+p+':ro']
 for p in devs: argv+=['--device',p]
 if 'runtime_overlay' in plan:
  smi=run(['npu-smi','info'],t=30)
  assert smi.returncode==0 and smi.stdout.count('No running processes found in NPU')==8,'Device process present; inspect before model launch'
  (RUN/'npu-smi-before.txt').write_text(smi.stdout+smi.stderr)
  ctx['npu_idle_precondition']={'empty_physical_cards':8,'raw_sha256':hashlib.sha256((smi.stdout+smi.stderr).encode()).hexdigest()}
  ov=plan['runtime_overlay'];source=pathlib.Path(ov['source'])
  assert str(source)=='/data/wio/glm5-3-vllm-execution/20261017-memcache/loader-fix-v3/site-packages'
  assert hashlib.sha256((source/'memcache_hybrid/__init__.py').read_bytes()).hexdigest()==ov['python_init_sha256']
  assert hashlib.sha256((source.parent/'aliases.json').read_bytes()).hexdigest()==ov['aliases_sha256']
  argv+=['-v',str(source)+':/opt/memcache-runtime:ro']
  original_env=json.loads(subprocess.check_output(['docker','image','inspect',image_id,'--format','{{json .Config.Env}}']))
  ld=next((v.split('=',1)[1] for v in original_env if v.startswith('LD_LIBRARY_PATH=')), '')
  py=next((v.split('=',1)[1] for v in original_env if v.startswith('PYTHONPATH=')), '')
  python_path='/opt/memcache-runtime'+(':'+py if py else '')
  probe=run(['docker','run','--rm','--pull','never','--network','none','--read-only','-e','PYTHONPATH='+python_path,'--entrypoint','python3',image_id,'-c','import importlib.util; s=importlib.util.find_spec("acl"); assert s is not None; print(s.origin)'])
  assert probe.returncode==0,'CANN Python module path unavailable: '+probe.stderr
  ctx['acl_module_preflight']=probe.stdout.strip()
  argv+=['-e','PYTHONPATH='+python_path,'-e','LD_LIBRARY_PATH=/opt/memcache-runtime/memcache_hybrid/lib:/opt/memcache-runtime/memfabric_hybrid/lib:'+ld]
  config=pathlib.Path(plan['memcache_config_source'])
  assert config.is_dir() and hashlib.sha256((config/'mmc-local.conf').read_bytes()).hexdigest()==plan['memcache_config_sha256']
  argv+=['-v',str(config)+':/memcache-config:ro']
  ctx['runtime_overlay']=ov
  ctx['memcache_config_sha256']=plan['memcache_config_sha256']
 for patch in plan.get('runtime_patches',[]):
  source=pathlib.Path(patch['source'])
  legacy_target=patch['target'] in (
   '/vllm-workspace/vllm/vllm/v1/core/sched/scheduler.py',
   '/vllm-workspace/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py',
   '/vllm-workspace/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py',
   '/vllm-workspace/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/metadata.py',
   '/vllm-workspace/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py',
   '/vllm-workspace/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/backend/memcache_backend.py')
  worker_target=patch['target']=='/vllm-workspace/vllm-ascend/vllm_ascend/worker/model_runner_v1.py'
  scheduler_target=(patch['target']=='/vllm-workspace/vllm/vllm/v1/core/sched/scheduler.py' and str(source)=='/data/wio/glm5-3-vllm-execution/20261017-official-pd/execution-patches/p-first-publish-v1/scheduler.py')
  assert (legacy_target and str(source).startswith('/data/wio/glm5-3-vllm-execution/20261017-pd-baseline/')) or (worker_target and str(source).startswith('/data/wio/glm5-3-vllm-execution/20261017-official-pd/execution-patches/')) or scheduler_target
  if scheduler_target:
   originals={patch['target']:'c1db45f3bbad3a875dd8638331b8870c4380ac1ab4734265ba5c863833a82884','/vllm-workspace/vllm/vllm/v1/core/sched/async_scheduler.py':'e586a0ef3c6778be56a93e7f9bb712d4de9de6e7d7fe7e3e1d51dae83ecfc508'}
   assert patch['base_sha256']==originals[patch['target']]
   probe=run(['docker','run','--rm','--pull','never','--network','none','--read-only','--entrypoint','sha256sum',image_id]+list(originals))
   assert probe.returncode==0,'Cannot verify pinned scheduler source: '+probe.stderr
   observed={line.split()[1]:line.split()[0] for line in probe.stdout.splitlines()}
   assert observed==originals,'Scheduler source differs from reviewed base'
   ctx['scheduler_patch_base_sha256']=observed
  if worker_target:
   originals={patch['target']:'568c3049205b9caff5c656a7f5dd4c6e79fb358abe526f89fb9fb3f70a60a43c','/vllm-workspace/vllm/vllm/v1/worker/gpu_model_runner.py':'87c29d08c0bbf66993d8b984811e7325e35e6436242a773feb55ec88f16b2c51'}
   assert patch['base_sha256']==originals[patch['target']]
   probe=run(['docker','run','--rm','--pull','never','--network','none','--read-only','--entrypoint','sha256sum',image_id]+list(originals))
   assert probe.returncode==0,'Cannot verify pinned worker source: '+probe.stderr
   observed={line.split()[1]:line.split()[0] for line in probe.stdout.splitlines()}
   assert observed==originals,'Worker/async-output source differs from reviewed base'
   ctx['worker_patch_base_sha256']=observed
  assert hashlib.sha256(source.read_bytes()).hexdigest()==patch['sha256']
  argv+=['-v',str(source)+':'+patch['target']+':ro']
 ctx['runtime_patches']=plan.get('runtime_patches',[])
 for k,v in plan['env'].items(): argv+=['-e',k+'='+v]
 assert json.dumps(plan['additional_config'],separators=(',',':'))==plan['serve_argv'][-1],'additional-config element mismatch'
 argv+=['--entrypoint','/usr/bin/timeout',image_id,'--signal=TERM','--kill-after=30',str(plan['container']['lifetime_seconds'])]+plan['serve_argv']
 for e in argv: assert e and not any(c.isspace() for c in e),'argv element is not a single token: '+repr(e)
 launch={'argv':argv,'image_id':image_id,'started_host_utc':utc(),'ended_host_utc':None,'docker_run_rc':None,'stdout':None,'stderr':None,'container_id':None,'inspect':None}
 proc=run(argv); launch['docker_run_rc']=proc.returncode; launch['stdout']=proc.stdout; launch['stderr']=proc.stderr
 launch['container_id']=proc.stdout.strip(); launch['ended_host_utc']=utc()
 (RUN/'launch.json').write_text(json.dumps(launch,indent=2)+chr(10))
 if proc.returncode==0:
  ins=run(['docker','inspect',launch['container_id'],'--format','{{.Id}} {{.Image}} {{.Name}} {{.State.Status}} {{.State.ExitCode}}'])
  f=ins.stdout.strip().split()
  fields=dict(zip(['id','image','name','state_status','state_exit_code'],f)) if ins.returncode==0 and len(f)==5 else None
  launch['inspect']={'rc':ins.returncode,'stdout':ins.stdout.strip(),'stderr':ins.stderr.strip(),'fields':fields}
  assert fields and fields['id']==launch['container_id'] and fields['image']==image_id,'post-run inspect mismatch: '+ins.stderr.strip()
 (RUN/'launch.json').write_text(json.dumps(launch,indent=2)+chr(10))
except Exception as exc:
 ctx['failure']='%s: %s'%(type(exc).__name__,exc)
 if launch is not None: (RUN/'launch.json').write_text(json.dumps(launch,indent=2)+chr(10))
 (RUN/'context.json').write_text(json.dumps(ctx,indent=2)+chr(10))
 emit('failed',failure=ctx['failure'])
 raise
(RUN/'context.json').write_text(json.dumps(ctx,indent=2)+chr(10))
emit('launched' if launch['docker_run_rc']==0 else 'docker_run_failed',container_id=launch['container_id'],docker_run_rc=launch['docker_run_rc'])
sys.exit(0 if launch['docker_run_rc']==0 else 1)
'''

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--control', required=True, help='existing SSH multiplex socket (ControlPath) to the gateway')
    p.add_argument('--output', type=Path, required=True, help='new local evidence directory; must not exist')
    p.add_argument('--plan', type=Path, default=TOOLS_DIR / PLAN_NAME, help='launch plan JSON (default: %(default)s)')
    a = p.parse_args()
    if a.output.exists():
        raise SystemExit('refusing to touch existing evidence dir: ' + str(a.output))
    plan_bytes = a.plan.read_bytes()
    plan = json.loads(plan_bytes)
    assert plan.get('schema') == SCHEMA, 'unexpected plan schema: ' + repr(plan.get('schema'))
    for key in ('gateway', 'access_pod', 'reservation', 'remote', 'weights_manifest', 'image', 'container', 'docker_argv', 'devices', 'env', 'serve_argv', 'additional_config', 'timeouts'):
        assert key in plan, 'plan is missing: ' + key
    payload = PAYLOAD.encode()
    ssh = ['ssh', '-S', a.control, '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes',
           '-p', str(plan['gateway']['port']), plan['gateway']['target']]
    a.output.mkdir(parents=True)
    res = plan['reservation']
    got = subprocess.run(ssh + ['kubectl get pod -n %s %s -o json' % (res['namespace'], res['name'])],
                         capture_output=True, text=True, timeout=plan['timeouts']['kubectl_s'])
    assert got.returncode == 0, 'kubectl get failed: ' + got.stderr.strip()
    pod = json.loads(got.stdout)  # parsed in memory only; the pod dump is never written to evidence
    if 'expected_uid' in res:
        assert res['expected_uid'] and pod['metadata']['uid'] == res['expected_uid'], 'fresh reservation UID mismatch'
        if plan.get('state_tail_probe') is True:
            deadline_s = int(pod['spec']['activeDeadlineSeconds'])
            assert 0 < deadline_s <= 1500, 'state diagnostic requires a bounded fresh holder'
            created = datetime.datetime.fromisoformat(pod['metadata']['creationTimestamp'].replace('Z', '+00:00'))
            remaining = (created + datetime.timedelta(seconds=deadline_s) - datetime.datetime.now(datetime.timezone.utc)).total_seconds()
            lifetime = int(plan['container']['lifetime_seconds'])
            assert 900 <= lifetime <= 1200 and remaining >= lifetime + 300, 'stale model lifetime or insufficient cleanup margin; regenerate from fresh receipts'
    assert pod['metadata']['labels'].get('owner') == res['owner_label'], 'task owner mismatch'
    assert pod['status']['phase'] == res['phase'] == 'Running', 'reservation is not Running'
    assert pod['spec']['nodeName'] == res['node_name'], 'reservation is on another node'
    assert int(pod['spec']['containers'][0]['resources']['requests'][res['ascend_resource']]) == res['ascend_request'], 'NPU request mismatch'
    lease = {'name': pod['metadata']['name'], 'uid': pod['metadata']['uid'], 'creation_timestamp': pod['metadata']['creationTimestamp'],
             'ascend_real': pod['metadata'].get('annotations', {}).get(res['ascend_real_annotation']),
             'active_deadline_seconds': pod['spec'].get('activeDeadlineSeconds'),
             'phase': pod['status']['phase'], 'node_name': pod['spec']['nodeName'], 'ascend_request': res['ascend_request']}
    assert lease['uid'] and lease['creation_timestamp'] and lease['ascend_real'], 'lease record incomplete'
    assert {int(x.rsplit('-',1)[1]) for x in lease['ascend_real'].split(',')} == set(range(16))
    try:
        git_sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=TOOLS_DIR, text=True, stderr=subprocess.DEVNULL).strip()
        git_diff_sha = hashlib.sha256(subprocess.check_output(['git', 'diff', 'HEAD'], cwd=TOOLS_DIR, stderr=subprocess.DEVNULL)).hexdigest()
    except Exception as err:
        git_sha = git_diff_sha = 'unavailable: %r' % (err,)
    setup = {'schema': SCHEMA, 'client_started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
             'run_dir': plan['remote']['run_dir'], 'source_script': Path(__file__).name,
             'source_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             'source_script_b64': base64.b64encode(Path(__file__).read_bytes()).decode(),
             'plan_sha256': hashlib.sha256(plan_bytes).hexdigest(), 'payload_sha256': hashlib.sha256(payload).hexdigest(),
             'plan_b64': base64.b64encode(plan_bytes).decode(), 'payload_b64': base64.b64encode(payload).decode(),
             'lease': lease, 'git_sha': git_sha, 'git_diff_sha256': git_diff_sha}
    setup_bytes = json.dumps(setup, indent=2, sort_keys=True, ensure_ascii=False).encode()
    remote = plan['remote']['run_dir']
    host = 'kubectl exec -i -n %s %s -- %s -- ' % (plan['access_pod']['namespace'], plan['access_pod']['name'],
                                                   plan['access_pod']['nsenter'])
    bootstrap = BOOTSTRAP.replace('SETUP', repr(base64.b64encode(setup_bytes).decode()))
    try:
        done = subprocess.run(ssh + [host + 'python3 -'], input=bootstrap, capture_output=True, text=True,
                              timeout=plan['timeouts']['launch_s'])
    except subprocess.TimeoutExpired as err:
        (a.output / 'transport.json').write_text(json.dumps(
            {'command': host + 'python3 -', 'timeout': True, 'error': repr(err)}, indent=2) + '\n')
        raise
    (a.output / 'transport.json').write_text(json.dumps(
        {'command': host + 'python3 -', 'rc': done.returncode, 'stdout': done.stdout, 'stderr': done.stderr}, indent=2) + '\n')
    tail = [ln for ln in done.stdout.splitlines() if ln.strip()]
    try: summary = json.loads(tail[-1]) if tail else {}
    except ValueError: summary = {}
    verified = []
    if summary:
        for name in ('manifest.json', 'plan.json', 'setup.json', 'payload.py', 'start_capacity.py', 'context.json', 'launch.json'):
            pull = subprocess.run(ssh + [host + 'cat ' + shlex.quote(remote + '/' + name)], capture_output=True,
                                  timeout=plan['timeouts']['fetch_s'])
            if pull.returncode or not pull.stdout:
                continue
            (a.output / name).write_bytes(pull.stdout)
            if name in summary.get('sha256', {}):
                assert hashlib.sha256(pull.stdout).hexdigest() == summary['sha256'][name], 'download sha mismatch: ' + name
            verified.append(name)
    for name, want in (('plan.json', plan_bytes), ('payload.py', payload), ('setup.json', setup_bytes)):
        if (a.output / name).exists():
            assert (a.output / name).read_bytes() == want, 'local/remote content mismatch: ' + name
    ok = done.returncode == 0 and summary.get('status') == 'launched'
    if ok: assert len(verified) == 7, 'Incomplete evidence download; inspect existing container before any retry'
    print(json.dumps({'local': str(a.output.resolve()), 'remote': remote, 'ssh_rc': done.returncode,
                      'status': summary.get('status'), 'container_id': summary.get('container_id'),
                      'docker_run_rc': summary.get('docker_run_rc'), 'lease_uid': lease['uid'],
                      'verified': verified, 'launched': ok, 'model_ready_checked': False}))
    if not ok:
        print('launch not confirmed; evidence retained, no retry and no blind change', file=sys.stderr)
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
