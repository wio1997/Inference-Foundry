"""Prepare/start one isolated diagnostic copy; original inference files stay intact."""
from pathlib import Path
import datetime
import hashlib
import json
import os
import shlex
import subprocess
import sys

HERE = Path(__file__).resolve().parent
SSH = '/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh'
BASE = 'wio-vllm-v41-repro169-20261028'
STAGE = 'wio-vllm-v41-repro169-stage-20261028'
DIAG = 'wio-vllm-v41-binding-diag169-20261029'
ROOT = '/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-169-20261028'
WORK = '/work/vllm-ascend-reproduce-172-on-169-20261028'
REL = '/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029/binding-diagnostic'
APP = '/data/tiankuan/wio/projects/910c-deepseek-v4.1-flash'
PROFILE = dict(profiler='torch', torch_profiler_dir=APP+'/diagnostic-traces',
               torch_profiler_with_stack=True, torch_profiler_record_shapes=True,
               torch_profiler_with_memory=False, ignore_frontend=False,
               detailed_trace_annotation=True, max_iterations=12)

REMOTE = r'''
import subprocess, json, re, datetime, hashlib, shlex, time
fence_fd=acquire_owner_fence(ROOT+REL)
def cmd(a, timeout=60):
    ensure_owner_fence(fence_fd)
    out=subprocess.check_output(a, text=True, timeout=timeout)
    ensure_owner_fence(fence_fd)
    return out
def save(obj):
    data=json.dumps(obj,indent=2)+'\n'
    code='\n'.join(['from pathlib import Path','import os,tempfile', 'p=Path('+repr(WORK+REL)+')', 'p.mkdir(parents=True,exist_ok=True)', 'data='+repr(data), 'def commit(path):', '    fd,tmp=tempfile.mkstemp(prefix=".launch-",dir=p)', '    try:', '        with os.fdopen(fd,"w") as f:', '            f.write(data); f.flush(); os.fsync(f.fileno())', '        os.replace(tmp,path)', '    finally:', '        if os.path.exists(tmp): os.unlink(tmp)', 'commit(p/'+repr('identity-'+RUN_ID+'.json')+')', 'commit(p/"launch.json")', 'fd=os.open(p,os.O_RDONLY|os.O_DIRECTORY)', 'try: os.fsync(fd)', 'finally: os.close(fd)'])
    cmd(['docker','exec',STAGE,'python3','-c',code])
def npu_pids():
    snapshot=cmd(['docker','exec',STAGE,'npu-smi','info'])
    parse_npu_hbm(snapshot)
    return parse_npu_processes(snapshot),snapshot
baseline=json.loads(cmd(['docker','inspect',BASE]))[0]
assert baseline['Id']=='b00f4d2018e3d495d876b6e396ba9579e78b47a97111b99c27d6678e31039092'
assert baseline['Image']=='sha256:ce42d65b2403992ad2e2b06c61f4f048babb57e455cfe307b018345e36c5e98b'
assert baseline['State']['Running'], 'Baseline must be running at initial guard'
existing=cmd(['docker','ps','-a','--format','{{.Names}}']).splitlines()
assert DIAG not in existing, 'New diagnostic name already exists: inspect, do not overwrite'
own={int(x.strip()) for x in cmd(['docker','top',baseline['Id'],'-eo','pid']).splitlines() if x.strip().isdigit()}
pids,snapshot=npu_pids()
assert pids<=own, ('OTHER_NPU_PROCESSES',sorted(pids-own))
reference=build_empirical_reference(snapshot,HIST1,HIST2,own)
health='import urllib.request,json;op=urllib.request.build_opener(urllib.request.ProxyHandler({}));u="http://127.0.0.1:8172";status=op.open(u+"/health",timeout=10).status;raw=op.open(u+"/metrics",timeout=10).read().decode();rows=[x for x in raw.splitlines() if x.startswith(("vllm:num_requests_running{","vllm:num_requests_waiting{"))];assert status==200 and len(rows)>=4 and all(float(x.rsplit(" ",1)[1])==0 for x in rows),rows;print(json.dumps({"health":status,"idle":rows}))'
idle=json.loads(cmd(['docker','exec',baseline['Id'],'python3','-c',health]))
prepare=r"""
from pathlib import Path
import shutil,subprocess,hashlib,json,shlex
root=Path(WORK+REL);root.mkdir(parents=True,exist_ok=True)
src=Path(WORK+'/runtime');dst=root/'runtime';assert not dst.exists(),'Never overwrite previous diagnostic runtime'
for name in ('scripts','config'): shutil.copytree(src/name,dst/name)
for name in ('cache','home','tmp','logs','diagnostic-traces','instrumentation','metadata'): (dst/name).mkdir(parents=True,exist_ok=True)
if (src/'cache').exists():subprocess.run(['cp','-a','--reflink=auto',str(src/'cache')+'/.',str(dst/'cache')],check=True)
text=(src/'scripts/serve.sh').read_text()
assert hashlib.sha256(text.encode()).hexdigest()=='c297201185b5676254af4d7693279f89bfa60f9c76eb340058a78ecf730b7bba'
lines=text.splitlines();indices=[i for i,s in enumerate(lines) if s.startswith('exec /usr/local/python3.12.13/bin/vllm serve ')];assert len(indices)==1
idx=indices[0]
lines[idx:idx]=['export PYTHONPATH='+shlex.quote(APP+'/instrumentation')+':${PYTHONPATH:-}',
               'export WIO_DIAG_DIR='+shlex.quote(APP+'/metadata'),
               'export WIO_DIAG_HMAC_KEY=wio-synthetic-recapture-169-20261029',
               'export BINDING_TRACE_PATH='+shlex.quote(APP+'/metadata/binding-trace.jsonl'),
               'export LD_PRELOAD='+shlex.quote(APP+'/instrumentation/libbinding_recorder.so')+':${LD_PRELOAD:-}']
lines[idx+5]+=' --profiler-config '+shlex.quote(json.dumps(PROFILE,separators=(',',':')))
(dst/'scripts/serve-profile.sh').write_text('\n'.join(lines)+'\n')
for name,data in BUNDLE.items():(dst/name).write_text(data)
# Build only inside this isolated runtime; no existing framework library changes.
source=Path(WORK+'/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029/binary-binding-draft/recorder.c')
import os
env=dict(os.environ);env['TMPDIR']=str(dst/'tmp')
subprocess.run(['gcc','-Wall','-Wextra','-Werror','-fPIC','-shared','-I/usr/local/Ascend/cann-9.2.0/aarch64-linux/include',str(source),'-o',str(dst/'instrumentation/libbinding_recorder.so'),'-ldl','-pthread'],check=True,env=env)
(root/'recorder-sha256.json').write_text(json.dumps({'source':hashlib.sha256(source.read_bytes()).hexdigest(),'library':hashlib.sha256((dst/'instrumentation/libbinding_recorder.so').read_bytes()).hexdigest()},indent=2)+'\n')
(root/'instrumentation-sha256.json').write_text(json.dumps({name:hashlib.sha256(data.encode()).hexdigest() for name,data in BUNDLE.items()},indent=2)+'\n')
print(json.dumps({'baseline_script_sha256':hashlib.sha256(text.encode()).hexdigest(),'diagnostic_script_sha256':hashlib.sha256((dst/'scripts/serve-profile.sh').read_bytes()).hexdigest(),'runtime':str(dst),'profile':PROFILE}))
"""
prepared=json.loads(cmd(['docker','exec',STAGE,'python3','-c','WORK='+repr(WORK)+'\nREL='+repr(REL)+'\nAPP='+repr(APP)+'\nPROFILE='+repr(PROFILE)+'\nBUNDLE='+repr(BUNDLE)+'\n'+prepare],timeout=300))
cfg=baseline['HostConfig'];args=['docker','create','--name',DIAG,'--label','wio.task=operator-ceiling169-20261029','--label','wio.binding_run_id='+RUN_ID,'--privileged','--read-only','--network','bridge','--ipc','private','--shm-size',str(cfg['ShmSize']),'-p','127.0.0.1:30077:8172','--workdir',APP,'--entrypoint','bash']
for env in baseline['Config']['Env']:args+=['--env',env]
for u in cfg.get('Ulimits') or []:args+=['--ulimit',str(u['Name'])+'='+str(u['Soft'])+':'+str(u['Hard'])]
for cap in cfg.get('CapAdd') or []:args+=['--cap-add',cap]
for option in cfg.get('SecurityOpt') or []:args+=['--security-opt',option]
for m in baseline['Mounts']:
    assert m['Type']=='bind'
    src=m['Source'];dest=m['Destination']
    if dest==APP:src=ROOT+REL+'/runtime'
    elif dest=='/tmp':src=ROOT+REL+'/runtime/tmp'
    else:assert not m['RW'],'Unexpected inherited writable mount: '+dest
    args+=['--mount','type=bind,src='+src+',dst='+dest+(',readonly' if not m['RW'] else '')]
args+=['--mount','type=bind,src='+ROOT+'/runtime/benchmark-20261028,dst=/bench-inputs,readonly']
for dev in cfg.get('Devices') or []:args+=['--device',dev['PathOnHost']+':'+dev['PathInContainer']+':'+dev['CgroupPermissions']]
args+=[baseline['Image'],'-c','exec bash '+APP+'/scripts/serve-profile.sh >'+APP+'/logs/service.log 2>&1']
diag_id=cmd(args).strip()
record={'state':'PREPARED','baseline_name':BASE,'baseline_id':baseline['Id'],'baseline_started_at':baseline['State']['StartedAt'],'diagnostic_name':DIAG,'diagnostic_id':diag_id,'image':baseline['Image'],'prepared':prepared,'idle_before':idle,'npu_before':snapshot,'created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'inference_semantics_unchanged':None,'hbm_reference':reference,'binding_run_id':RUN_ID,'inference_source_modified':False,'rollback':['docker','stop','--time','45',diag_id],'restore':['docker','start',baseline['Id']]}
save(record)
# Repeat guards immediately before the only service interruption.
own={int(x.strip()) for x in cmd(['docker','top',baseline['Id'],'-eo','pid']).splitlines() if x.strip().isdigit()}
pids,current=npu_pids();assert pids<=own,('OTHER_NPU_PROCESSES',sorted(pids-own))
reference=build_empirical_reference(current,HIST1,HIST2,own)
record['hbm_reference']=reference;save(record)
cmd(['docker','exec',baseline['Id'],'python3','-c',health])
verify='\n'.join(['from pathlib import Path','import json', 'p=Path('+repr(WORK+REL)+')', 'ref='+repr(reference), 'expected_run='+repr(RUN_ID), 'baseline='+repr(baseline['Id']), 'diagnostic='+repr(diag_id), 'for name in ("launch.json",'+repr('identity-'+RUN_ID+'.json')+'):', '    x=json.loads((p/name).read_text())', '    assert x.get("binding_run_id")==expected_run and x.get("baseline_id")==baseline and x.get("diagnostic_id")==diagnostic and x.get("hbm_reference")==ref'])
cmd(['docker','exec',STAGE,'python3','-c',verify])
lease=json.loads(cmd(['docker','exec',STAGE,'cat',WORK+REL+'/watchdog-lease.json']))
assert lease.get('binding_run_id')==RUN_ID and 0 <= time.time()-lease.get('epoch',0) < 20, 'recovery watchdog lease missing or stale'
cmd(['docker','stop','--time','45',baseline['Id']],timeout=100)
try:
    release=wait_for_empirical_release(lambda:cmd(['docker','exec',STAGE,'npu-smi','info']),reference)
    if not release['satisfies_policy']:
        record.update(state='WAITING_DEVICE_RELEASE',release_reason=release['reason'],other_pids=sorted(release.get('pids',[])))
        save(record)
        raise RuntimeError('Device release does not meet the observed baseline; not starting diagnostic')
    _,after=npu_pids()
    if not verified_release(after,reference)['satisfies_policy']:raise RuntimeError('Device status changed after release observation')
    cmd(['docker','start',diag_id])
    record.update(state='DIAGNOSTIC_LOADING',npu_after_park=after)
    save(record)
except BaseException as exc:
    record['launch_error']=repr(exc)
    diag=json.loads(cmd(['docker','inspect',diag_id]))[0]
    if diag['State']['Running']:cmd(['docker','stop','--time','45',diag_id],timeout=100)
    release=wait_for_empirical_release(lambda:cmd(['docker','exec',STAGE,'npu-smi','info']),reference)
    if release['satisfies_policy']:
        cmd(['docker','start',baseline['Id']]);record['state']='BASELINE_RESTORING_AFTER_LAUNCH_ERROR'
    elif release.get('pids'):record['state']='WAITING_OTHER_NPU_PROCESSES_BEFORE_RESTORE'
    else:record['state']='RESTORE_DEFERRED_UNATTRIBUTED_HBM'
    save(record)
    raise
print(json.dumps({k:v for k,v in record.items() if k not in ('npu_before','npu_after_park')}))
'''

REMOTE=(HERE/'remote_guard.py').read_text()+'\n'+REMOTE

def main():
    names=['instrumentation/sitecustomize.py','profile_client.py']
    bundle={name:(HERE.parent/name).read_text() for name in names}
    for name,source in bundle.items():compile(source,name,'exec')
    RUN_ID=os.environ.get('WIO_BINDING_RUN_ID','')
    assert len(RUN_ID)==32 and all(c in '0123456789abcdef' for c in RUN_ID), 'unique watchdog run ID required'
    HIST1=Path('/Users/wio/work/169-sglang-deepseek-v41/evidence/001-npu-preflight.txt').read_text()
    HIST2=Path('/Users/wio/work/169-sglang-deepseek-v41/evidence/009-npu-occupancy-current.txt').read_text()
    assert hashlib.sha256(HIST1.encode()).hexdigest()=='fe22dc33030d27b8176492bb5f43cab4055fb6ddbb56df1b26b7a7b0eb1fba93'
    assert hashlib.sha256(HIST2.encode()).hexdigest()=='62886a47b5dd18b7954d79dc903bb37ee10c0a9eb1a27ac4cb49a9be78cdfd4e'
    constants=dict(HIST1=HIST1,HIST2=HIST2,RUN_ID=RUN_ID,BASE=BASE,STAGE=STAGE,DIAG=DIAG,ROOT=ROOT,WORK=WORK,REL=REL,APP=APP,PROFILE=PROFILE,BUNDLE=bundle)
    code='\n'.join(k+'='+repr(v) for k,v in constants.items())+'\n'+REMOTE
    result=subprocess.run([SSH,shlex.join(['python3','-c',code])],capture_output=True,text=True,timeout=600)
    (HERE/'launch-controller.stdout').write_text(result.stdout)
    (HERE/'launch-controller.stderr').write_text(result.stderr)
    if result.returncode:
        print(result.stderr[-5000:]);print(result.stdout[-1500:]);raise SystemExit(result.returncode)
    data=json.loads(result.stdout);(HERE/'launch.json').write_text(json.dumps(data,indent=2)+'\n')
    print(json.dumps(data,indent=2))

if __name__=='__main__':
    if sys.argv[1:] != ['--execute']:
        print('Review draft: no operations. Pass --execute only after reviewing recovery procedure.');raise SystemExit(0)
    main()
