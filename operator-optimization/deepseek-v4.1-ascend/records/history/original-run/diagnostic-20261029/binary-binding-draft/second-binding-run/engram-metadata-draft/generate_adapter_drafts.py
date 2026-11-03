"""Generate disabled copies of the previously used owned-container chain.

Local file generation only: NO Docker, SSH, NPU, service, or remote operation.
The generated scripts remain disabled until explicitly reviewed and edited.
"""
from pathlib import Path
import hashlib
import os
import tempfile
import sys

if sys.flags.optimize:raise RuntimeError('optimized Python forbidden during generation')

here=Path(__file__).resolve().parent
old=here.parent
generated={}
expected={
    'start_binding_recapture.py':'3d7a926b0ab982df00e7630bafe1eea46da1656d3228333b4520020dea80a38f',
    'supervise_binding_recapture.py':'036b4bb48ed1db8c1f383e41f0b19f79383470754b1e5cd3b0a1bbeb9b6d73fb',
    'recover_binding.py':'f8cd06cc8fa9479750fd017a062aa682720c36ff38695f24c9849602e6541b50',
    'remote_guard.py':'f475694ae5f660394d60d5239fa6f2ec640d64ab09d67401827f4bfee83383d7'}
for name,want in expected.items():
    if hashlib.sha256((old/name).read_bytes()).hexdigest()!=want:
        raise RuntimeError('previous reviewed source changed: '+name)


def once(text, old, new):
    if text.count(old)!=1:raise RuntimeError('transformation anchor is not unique: '+old[:120])
    return text.replace(old,new,1)

holder_prefix="import sys\nif sys.flags.optimize: raise RuntimeError('optimized Python forbidden')\n"
holder_injection="\nHOLDER_CODE = "+repr(holder_prefix)+" + HOLDER_CODE\n"

def common(text):
    text=text.replace("(HERE/'remote_guard.py').read_text()", "(HERE.parent/'remote_guard.py').read_text()")
    text=text.replace("wio-vllm-v41-binding-diag2-169-20261029", "wio-vllm-v41-engram-REPLACEME")
    text=text.replace("binding-diagnostic-selective2", "engram-diagnostic-REPLACEME")
    text=text.replace("'wio-vllm-v41-engram-REPLACEME'", "'wio-vllm-v41-engram-'+RUN_ID")
    text=text.replace("'/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029/engram-diagnostic-REPLACEME'", "'/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029/engram-diagnostic-'+RUN_ID")
    return text


start=common((old/'start_binding_recapture.py').read_text())
start=once(start,"DIAG = 'wio-vllm-v41-engram-'+RUN_ID", "RUN_ID = os.environ.get('WIO_BINDING_RUN_ID','')\nDIAG = 'wio-vllm-v41-engram-'+RUN_ID")
a=start.index('PROFILE = dict(');b=start.index('\nREMOTE =',a)
start=start[:a]+start[b:]
a=start.index('prepare=r"""');b=start.index('\n"""\nprepared=',a)+4
prepare='''prepare=r"""
from pathlib import Path
import shutil,subprocess,hashlib,json,shlex,os
os.umask(0o077)
root=Path(WORK+REL);root.mkdir(parents=True,exist_ok=True)
src=Path(WORK+'/runtime');dst=root/'runtime';assert not dst.exists(),'Never overwrite previous diagnostic runtime'
frozen=src/'benchmark-20261028/workload/frozen'
expected={'warmup':{'seed':'de3aa10aefe1287fca42ad831cc8bdc27dc0118b6dd2be68f7329f8a39890fc0','requests':'c5fb896a2d9d5fe13c0b27eb1cf92932edb31bf3cd2007de1f8a60b5900c20e8'},'measure1':{'seed':'6dafe45dd0225129ac9e3e79193619a284df4ea410bd48437739c0eb03c029b5','requests':'7863f18d0f0ac87fd158b773f181b9abb81a528245ed3e1d8a1dfcbcdf4dc404','references':'c00806e9438e56bc7485dfba73a36ab7eea3bf08009fe167129d4d03cabb69f4'}}
for phase,hashes in expected.items():
 folder=frozen/phase;files=list(folder.rglob('*.requests.jsonl'));assert len(files)==1,'ambiguous frozen requests'
 checks={'seed':folder/'warmup.json','requests':files[0]}
 if 'references' in hashes:checks['references']=folder/'frozen-vllm-details.jsonl'
 for key,path in checks.items():assert hashlib.sha256(path.read_bytes()).hexdigest()==hashes[key],('frozen fixture mismatch',phase,key)
for name in ('scripts','config'): shutil.copytree(src/name,dst/name)
for name in ('cache','home','tmp','logs','instrumentation','metadata'): (dst/name).mkdir(parents=True,exist_ok=True,mode=0o700)
meta=dst/'metadata';assert str(meta.resolve())==str(meta.absolute()) and not meta.is_symlink(),'metadata path contains symlink'
assert meta.stat().st_uid==BASE_UID and meta.stat().st_mode & 0o077==0,'metadata ownership or mode mismatch'
if (src/'cache').exists():subprocess.run(['cp','-a','--reflink=auto',str(src/'cache')+'/.',str(dst/'cache')],check=True)
text=(src/'scripts/serve.sh').read_text()
assert hashlib.sha256(text.encode()).hexdigest()=='c297201185b5676254af4d7693279f89bfa60f9c76eb340058a78ecf730b7bba'
lines=text.splitlines();indices=[i for i,s in enumerate(lines) if s.startswith('exec /usr/local/python3.12.13/bin/vllm serve ')];assert len(indices)==1
assert '--profiler-config' not in lines[indices[0]],'baseline profiler settings conflict'
assert not any('LD_PRELOAD=' in line or 'PYTHONPATH=' in line for line in lines if not line.lstrip().startswith('#')),'baseline injection environment needs review'
idx=indices[0]
lines[idx:idx]=['export PYTHONPATH='+shlex.quote(APP+'/instrumentation'),
               'export WIO_DIAG_DIR='+shlex.quote(APP+'/metadata'),
               'export WIO_ENGRAM_META_ENABLE=1',
               'export WIO_ENGRAM_RUN_ID='+shlex.quote(RUN_ID),
               'export WIO_ENGRAM_META_PATH='+shlex.quote(APP+'/metadata/engram-metadata.jsonl')]
(dst/'scripts/serve-engram.sh').write_text('\\n'.join(lines)+'\\n')
for name,data in BUNDLE.items():(dst/name).write_text(data)
print(json.dumps({'baseline_script_sha256':hashlib.sha256(text.encode()).hexdigest(),
 'diagnostic_script_sha256':hashlib.sha256((dst/'scripts/serve-engram.sh').read_bytes()).hexdigest(),
 'bundle_sha256':{name:hashlib.sha256(data.encode()).hexdigest() for name,data in BUNDLE.items()},
 'runtime':str(dst),'profiler_enabled_by_adapter':False}))
"""'''
start=start[:a]+prepare+start[b:]
start=start.replace("'\nREL='+repr(REL)+'\nAPP='+repr(APP)+'\nPROFILE='+repr(PROFILE)+'\nBUNDLE='+repr(BUNDLE)","'\nREL='+repr(REL)+'\nAPP='+repr(APP)+'\nRUN_ID='+repr(RUN_ID)+'\nBUNDLE='+repr(BUNDLE)")
# The literal includes escaped newlines in the generator source.
start=start.replace("+'\\nAPP='+repr(APP)+'\\nPROFILE='+repr(PROFILE)+'\\nBUNDLE='+repr(BUNDLE)","+'\\nAPP='+repr(APP)+'\\nRUN_ID='+repr(RUN_ID)+'\\nBASE_UID='+repr(BASE_UID)+'\\nBUNDLE='+repr(BUNDLE)")
start=once(start,"prepared=json.loads(cmd(['docker','exec',STAGE,", "BASE_UID=int(cmd(['docker','exec',baseline['Id'],'python3','-c','import os;print(os.getuid())']).strip())\nprepared=json.loads(cmd(['docker','exec',STAGE,")
start=once(start,"cfg=baseline['HostConfig'];args=", "cfg=baseline['HostConfig'];assert cfg.get('NetworkMode')=='bridge' and cfg.get('IpcMode')=='private' and cfg.get('ReadonlyRootfs') is True and cfg.get('Privileged') is True,'unexpected baseline container mode'\nassert not any(x.startswith(('LD_PRELOAD=','PYTHONPATH=')) and x.split('=',1)[1] for x in baseline['Config']['Env']),'inherited injection environment conflicts'\nargs=")
start=once(start,"assert not any(x.startswith(('LD_PRELOAD=','PYTHONPATH=')) and x.split('=',1)[1] for x in baseline['Config']['Env']),'inherited injection environment conflicts'", "assert not any(x.startswith(('LD_PRELOAD=','PYTHONPATH=')) and x.split('=',1)[1] for x in baseline['Config']['Env']),'inherited injection environment conflicts'\nassert not any(x.startswith('PYTHONOPTIMIZE=') and x.split('=',1)[1] not in ('','0') for x in baseline['Config']['Env']),'optimized Python forbidden'")
start=once(start,"def cmd(a, timeout=60):\n    ensure_owner_fence(fence_fd)", """def cmd(a, timeout=60):
    ensure_owner_fence(fence_fd)
    if 'python3' in a:
        i=a.index('python3')
        if '-c' in a[i+1:]:
            j=a.index('-c',i+1)+1;a=list(a);a[j]='import sys\\nif sys.flags.optimize: raise RuntimeError(\\"optimized Python forbidden\\")\\n'+a[j]""")
start=start.replace('serve-profile.sh','serve-engram.sh')
start=once(start,"diag_id=cmd(args).strip()\nrecord=", """diag_id=cmd(args).strip()
assert re.fullmatch('[0-9a-f]{64}',diag_id),'invalid diagnostic identity'
created=json.loads(cmd(['docker','inspect',diag_id]))[0]
assert created['Id']==diag_id and created['Image']==baseline['Image'] and not created['State']['Running']
assert created['Config']['Labels'].get('wio.task')=='operator-ceiling169-20261029' and created['Config']['Labels'].get('wio.binding_run_id')==RUN_ID
# Any unspecifiable or omitted hardware/resource difference must fail closed.
keys=('NetworkMode','IpcMode','ReadonlyRootfs','Privileged','ShmSize','Devices','DeviceRequests','Ulimits','CapAdd','SecurityOpt','NanoCpus','CpuShares','CpuPeriod','CpuQuota','CpusetCpus','CpusetMems','Memory','MemorySwap','MemoryReservation','PidsLimit','Runtime')
for key in keys:assert created['HostConfig'].get(key)==cfg.get(key),('host configuration mismatch',key)
assert created['Config']['Env']==baseline['Config']['Env'],'container environment changed'
record=""")
a=start.index("    names=['instrumentation/sitecustomize.py','profile_client.py']")
b=start.index('    RUN_ID=os.environ.get',a)
bundle='''    boot=HERE.parent/'engram-audited-bootstrap'
    manifest=json.loads((HERE/'bootstrap-approved-sha256.json').read_text())
    names=('sitecustomize.py','engram_bootstrap.py','audited_observer.py')
    assert manifest.get('status')=='REVIEWED_FOR_THIS_DIAGNOSTIC'
    bundle={}
    for name in names:
        source=(boot/name).read_text()
        assert hashlib.sha256(source.encode()).hexdigest()==manifest['sha256'][name],name
        bundle['instrumentation/'+name]=source
    bundle['engram_client.py']=(HERE/'engram_client.py').read_text()
    assert hashlib.sha256(bundle['engram_client.py'].encode()).hexdigest()==manifest['sha256']['engram_client.py']
    for name,source in bundle.items():compile(source,name,'exec')
'''
start=start[:a]+bundle+start[b:]
start=start.replace(',PROFILE=PROFILE,BUNDLE=bundle',',BUNDLE=bundle')
start=start.replace("(HERE/'launch.json').write_text", "(HERE/('launch-'+RUN_ID+'.json')).write_text")
for filename in ('launch-controller.stdout','launch-controller.stderr'):
    start=start.replace("HERE/'"+filename+"'", "HERE/('"+filename.replace('.',"-'+RUN_ID+'.",1)+"')")
start=once(start,"REMOTE=(HERE.parent/'remote_guard.py').read_text()+'\\n'+REMOTE", "assert hashlib.sha256((HERE.parent/'remote_guard.py').read_bytes()).hexdigest()=='f475694ae5f660394d60d5239fa6f2ec640d64ab09d67401827f4bfee83383d7'\nREMOTE=(HERE.parent/'remote_guard.py').read_text()+"+repr(holder_injection)+"+REMOTE")
start=once(start,"    main()", "    raise SystemExit('DISABLED: review the independent adapter before enabling execution')\n    main()")
generated['engram_adapter_start.py']=start

# Recovery retains the original strict owner/identity + empirical release flow.
recover=common((old/'recover_binding.py').read_text())
recover=once(recover,"DIAG='wio-vllm-v41-engram-'+RUN_ID", "RUN_ID=os.environ.get('WIO_BINDING_RUN_ID','')\nDIAG='wio-vllm-v41-engram-'+RUN_ID")
recover=recover.replace("/engram-diagnostic-REPLACEME'", "/engram-diagnostic-'+RUN_ID")
recover=recover.replace("ROOT.removesuffix(\"-selective2\")", "LOCK_ROOT")
recover=once(recover,"if found:\n    diag_id=found[0]", "own_before=set()\nif found:\n    diag_id=found[0]")
recover=once(recover,"    if d['State']['Running']:run(['docker','stop','--time','45',diag_id],timeout=100)", """    if d['State']['Running']:
        own_before={int(x.strip()) for x in run(['docker','top',diag_id,'-eo','pid']).splitlines() if x.strip().isdigit()}
        if not own_before:raise ValueError('cannot identify own diagnostic processes before stop')
        run(['docker','stop','--time','45',diag_id],timeout=100)
    stopped=json.loads(run(['docker','inspect',diag_id]))[0]
    if stopped['Id']!=diag_id or stopped['State']['Running']:raise ValueError('diagnostic stop not confirmed')""")
recover=once(recover,"elif release.get('pids'):out={'state':'RESTORE_DEFERRED_OTHER_NPU_OWNER','other_pids':sorted(release['pids'])}", "elif release.get('pids') and own_before & release['pids']:out={'state':'RESTORE_DEFERRED_OWNED_WORKERS','pids':sorted(own_before & release['pids'])}\nelif release.get('pids'):out={'state':'RESTORE_DEFERRED_OTHER_NPU_OWNER','other_pids':sorted(release['pids'])}")
recover=recover.replace("('RESTORE_DEFERRED_OTHER_NPU_OWNER','RESTORE_DEFERRED_UNATTRIBUTED_HBM')", "('RESTORE_DEFERRED_OWNED_WORKERS','RESTORE_DEFERRED_OTHER_NPU_OWNER','RESTORE_DEFERRED_UNATTRIBUTED_HBM')")
recover=once(recover,"REMOTE=r'''", "LOCK_ROOT='/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029/binding-diagnostic'\nREMOTE=r'''")
recover=recover.replace('WORK=WORK,ROOT=ROOT,RUN_ID=RUN_ID','WORK=WORK,ROOT=ROOT,LOCK_ROOT=LOCK_ROOT,RUN_ID=RUN_ID')
recover=recover.replace("HERE/'manual-recovery-state.json'", "HERE/('manual-recovery-state-'+RUN_ID+'.json')")
recover=once(recover,"def ssh(code,timeout):", "def ssh(code,timeout):\n    import hashlib\n    assert hashlib.sha256((HERE.parent/'remote_guard.py').read_bytes()).hexdigest()=='f475694ae5f660394d60d5239fa6f2ec640d64ab09d67401827f4bfee83383d7'")
recover=recover.replace("+(HERE.parent/'remote_guard.py').read_text()+'\\n'+code", "+(HERE.parent/'remote_guard.py').read_text()+"+repr(holder_injection)+"+code")
recover=once(recover,"HERE=Path(__file__).resolve().parent", "raise SystemExit('DISABLED: review the independent adapter before enabling execution')\nHERE=Path(__file__).resolve().parent")
generated['engram_adapter_recover.py']=recover

sup=common((old/'supervise_binding_recapture.py').read_text())
sup=once(sup,"DIAG='wio-vllm-v41-engram-'+RUN_ID", "RUN_ID=os.environ.get('WIO_BINDING_RUN_ID','')\nDIAG='wio-vllm-v41-engram-'+RUN_ID")
sup=once(sup,"launch=json.loads((HERE/'launch.json').read_text())", "launch=json.loads((HERE/('launch-'+RUN_ID+'.json')).read_text())")
sup=sup.replace("APP+'/profile_client.py'", "APP+'/engram_client.py'")
sup=sup.replace("if r.returncode:raise RuntimeError(r.stderr[-5000:])", "if r.returncode:raise RuntimeError('no-profiler client failed; see dedicated sanitized progress')")
sup=once(sup,"r=subprocess.run(['docker','exec',DIAG_ID,'python3',APP+'/engram_client.py'],capture_output=True,text=True,timeout=2400)\nprint(r.stdout)\nif r.returncode:raise RuntimeError('no-profiler client failed; see dedicated sanitized progress')", '''import time
r=subprocess.Popen(['docker','exec','-e','WIO_ENGRAM_RUN_ID='+RUN_ID,DIAG_ID,'python3',APP+'/engram_client.py'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,text=True)
deadline=time.monotonic()+2400
try:
 while r.poll() is None:
  if time.monotonic()>deadline:raise TimeoutError('client timeout')
  ensure_owner_fence(fence_fd)
  current_own={int(x.strip()) for x in subprocess.check_output(['docker','top',DIAG_ID,'-eo','pid'],text=True).splitlines() if x.strip().isdigit()}
  current=parse_npu_processes(subprocess.check_output(['docker','exec',STAGE,'npu-smi','info'],text=True))
  if current-current_own:raise RuntimeError('OTHER_NPU_PROCESSES_DURING_CAPTURE')
  time.sleep(5)
 r.communicate(timeout=10)
 if r.returncode:raise RuntimeError('no-profiler client failed; see dedicated sanitized progress')
 print('no-profiler client completed')
finally:
 if r.poll() is None:
  r.terminate()
  try:r.wait(timeout=5)
  except subprocess.TimeoutExpired:r.kill();r.wait(timeout=5)
''')
sup=once(sup,"r=subprocess.run(['docker','exec',DIAG_ID,'python3','-c',check],capture_output=True,text=True,timeout=15) if d['State']['Running'] else None", """r=subprocess.run(['docker','exec',DIAG_ID,'python3','-c',check],capture_output=True,text=True,timeout=15) if d['State']['Running'] else None
own={int(x.strip()) for x in subprocess.check_output(['docker','top',DIAG_ID,'-eo','pid'],text=True).splitlines() if x.strip().isdigit()} if d['State']['Running'] else set()
pids=parse_npu_processes(subprocess.check_output(['docker','exec',STAGE,'npu-smi','info'],text=True))
if pids-own:raise RuntimeError('OTHER_NPU_PROCESSES_DURING_LOAD')""")
sup=once(sup,"if d['State']['Running']:cmd(['docker','stop','--time','45',DIAG_ID],timeout=100)", """own_before=set()
if d['State']['Running']:
    own_before={int(x.strip()) for x in cmd(['docker','top',DIAG_ID,'-eo','pid']).splitlines() if x.strip().isdigit()}
    if not own_before:raise ValueError('cannot identify own diagnostic processes before stop')
    cmd(['docker','stop','--time','45',DIAG_ID],timeout=100)
stopped=json.loads(cmd(['docker','inspect',DIAG_ID]))[0]
if stopped['Id']!=DIAG_ID or stopped['State']['Running']:raise ValueError('diagnostic stop not confirmed')""")
sup=once(sup,"def cmd(args,timeout=60):\n    ensure_owner_fence(fence_fd);out=subprocess.check_output(args,text=True,timeout=timeout);ensure_owner_fence(fence_fd);return out", """def cmd(args,timeout=60):
    ensure_owner_fence(fence_fd)
    if 'python3' in args:
        i=args.index('python3')
        if '-c' in args[i+1:]:
            j=args.index('-c',i+1)+1;args=list(args);args[j]='import sys\\nif sys.flags.optimize: raise RuntimeError("optimized Python forbidden")\\n'+args[j]
    out=subprocess.check_output(args,text=True,timeout=timeout);ensure_owner_fence(fence_fd);return out""")
sup=once(sup,"elif release.get('pids'):out={'state':'WAITING_OTHER_NPU_PROCESSES_BEFORE_RESTORE','other_pids':sorted(release['pids'])}", "elif release.get('pids') and own_before & release['pids']:out={'state':'WAITING_OWNED_WORKERS_BEFORE_RESTORE','pids':sorted(own_before & release['pids'])}\n    elif release.get('pids'):out={'state':'WAITING_OTHER_NPU_PROCESSES_BEFORE_RESTORE','other_pids':sorted(release['pids'])}")
a=sup.index("RAW_PROBE=r'''");b=sup.index("RESTORE=r'''",a)
summary='''SUMMARY=r\'\'\'
import subprocess
code=r"""from pathlib import Path
import collections,json,time
p=Path(APP);m=p/'metadata';counts=collections.Counter();malformed=0
log=m/'engram-metadata.jsonl'
log_exists=log.exists();log_oversize=log_exists and log.stat().st_size>1024*1024
if log_exists and not log_oversize:
 for line in log.open():
  try: row=json.loads(line);counts[str(row.get('event','<missing>'))]+=1
  except (ValueError,TypeError):malformed+=1
progress=json.loads((p/'engram-client-results/progress.json').read_text())
deadline=time.monotonic()+2;statuses=[];status_invalid=0
while True:
 files=list(m.glob('observer-*.json'));statuses=[];status_invalid=0
 if len(files)>64:status_invalid+=1;files=[]
 for f in files:
  try:
   if f.is_symlink() or f.stat().st_size>4096:raise ValueError('status file invalid')
   s=json.loads(f.read_text())
   if s.get('run_id')!=progress.get('run_id'):raise ValueError('status run mismatch')
   statuses.append(s)
  except (OSError,ValueError,TypeError):status_invalid+=1
 if (statuses and not status_invalid and all(s.get('closed_window')==progress.get('window_id') for s in statuses)) or time.monotonic()>=deadline:break
 time.sleep(.1)
closed=sum(s.get('closed_window')==progress.get('window_id') for s in statuses)
print(json.dumps({'progress':progress,'event_counts':dict(counts),'malformed_lines':malformed,'log_exists':log_exists,'log_oversize':log_oversize,
                  'observer_status_file_count':len(files),'valid_statuses':len(statuses),
                  'invalid_statuses':status_invalid,'observed_closed_window':closed,
                  'rank_coverage':'UNKNOWN',
                  'graph_replay_coverage':'UNKNOWN','window_ack_coverage':'UNKNOWN',
                  'log_completeness':'UNKNOWN'}))
"""
r=subprocess.run(['docker','exec',DIAG_ID,'python3','-c','APP='+repr(APP)+'\\n'+code],capture_output=True,text=True,timeout=90)
if r.returncode:raise RuntimeError('metadata summary unavailable')
print(r.stdout)
\'\'\'
'''
sup=sup[:a]+summary+sup[b:]
a=sup.index("    # This image requires offline analysis;");b=sup.index('\nexcept BaseException as exc:',a)
sup=sup[:a]+'''    summary=json.loads(ssh(SUMMARY,timeout=120))
    (HERE/('capture-summary-'+RUN_ID+'.json')).write_text(json.dumps(summary,indent=2)+'\\n')
    save('SUMMARY_SAVED',rank_coverage='UNKNOWN',graph_replay_coverage='UNKNOWN',log_completeness='UNKNOWN')
'''+sup[b:]
sup=sup.replace("print(json.dumps({'capture_client_complete':True,'raw_profile_status':state.get('raw_profile_status','UNKNOWN'),'raw_profile_visibility_complete':state.get('raw_profile_visibility_complete',False),'offline_analysis_completed':False,'restore':restored}))", "print(json.dumps({'capture_client_complete':True,'coverage':'UNKNOWN','restore':restored}))")
sup=once(sup,"def ssh(code,timeout=90):", "def ssh(code,timeout=90):\n    import hashlib\n    assert hashlib.sha256((HERE.parent/'remote_guard.py').read_bytes()).hexdigest()=='f475694ae5f660394d60d5239fa6f2ec640d64ab09d67401827f4bfee83383d7'")
sup=sup.replace("+(HERE.parent/'remote_guard.py').read_text()+'\\n'+code", "+(HERE.parent/'remote_guard.py').read_text()+"+repr(holder_injection)+"+code")
for filename in ('supervisor.json','capture.stdout','supervisor-error.txt','restore.json'):
    sup=sup.replace("HERE/'"+filename+"'", "HERE/('"+filename.replace('.',"-'+RUN_ID+'.",1)+"')")
sup=once(sup,"HERE=Path(__file__).resolve().parent", "raise SystemExit('DISABLED: review the independent adapter before enabling execution')\nHERE=Path(__file__).resolve().parent")
generated['engram_adapter_supervise.py']=sup

generated['engram_adapter_watchdog.py']=(here/'watchdog_heartbeat.template').read_text()
generated['engram_adapter_orchestrate.py']=(here/'orchestrator_heartbeat.template').read_text()

# Reuse the prior CPU-only fence, corrupted identity and recovery simulations
# against the generated copies. The original test source remains untouched.
test=(old/'test_fencing_parser_mock.py').read_text()
test=test.replace('import remote_guard', "sys.path.insert(0,str(Path(__file__).resolve().parent.parent))\nimport remote_guard")
test=test.replace('recover_binding.py','engram_adapter_recover.py').replace('start_binding_recapture.py','engram_adapter_start.py').replace('supervise_binding_recapture.py','engram_adapter_supervise.py').replace('watchdog_binding.py','engram_adapter_watchdog.py')
test=test.replace("env=dict(ROOT='/unused',", "env=dict(LOCK_ROOT='/unused',ROOT='/unused',")
test=test.replace("here.parent.parent/'ownership-check-20261029.raw.json'", "here.parent.parent.parent/'ownership-check-20261029.raw.json'")
test=once(test,"    calls=[];bid={'Id':BASE_ID,'State':{'Running':running}}", "    calls=[];bid={'Id':BASE_ID,'State':{'Running':running}};diagnostic_state={'Running':True}")
test=once(test,"{'Id':DIAG_ID,'State':{'Running':True},'Config':", "{'Id':DIAG_ID,'State':diagnostic_state,'Config':")
test=once(test,"        if args[:3]==['docker','stop','--time']:\n            assert args[-1]==DIAG_ID;return DIAG_ID+'\\n'", "        if args[:2]==['docker','top']:return 'PID\\n999\\n'\n        if args[:3]==['docker','stop','--time']:\n            assert args[-1]==DIAG_ID;diagnostic_state['Running']=False;return DIAG_ID+'\\n'")
(here/'test_engram_adapter_guard_mock.py').write_text(test)

import harden_adapter_drafts as hardener
checked={name:hardener.harden_source(source,name) for name,source in generated.items()}
for name,source in checked.items():
    compile(source,name,'exec')
    fd,tmp=tempfile.mkstemp(prefix='.engram-adapter-',dir=here)
    try:
        with os.fdopen(fd,'w') as f:f.write(source);f.flush();os.fsync(f.fileno())
        os.replace(tmp,here/name)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)
    print(name,hashlib.sha256(source.encode()).hexdigest())
