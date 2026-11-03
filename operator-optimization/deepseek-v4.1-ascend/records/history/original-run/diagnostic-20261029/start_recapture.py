"""Prepare/start one isolated diagnostic copy; original inference files stay intact."""
from pathlib import Path
import datetime
import json
import shlex
import subprocess
import sys

HERE = Path(__file__).resolve().parent
SSH = '/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh'
BASE = 'wio-vllm-v41-repro169-20261028'
STAGE = 'wio-vllm-v41-repro169-stage-20261028'
DIAG = 'wio-vllm-v41-ceiling-diag169-20261029'
ROOT = '/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-169-20261028'
WORK = '/work/vllm-ascend-reproduce-172-on-169-20261028'
REL = '/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029'
APP = '/data/tiankuan/wio/projects/910c-deepseek-v4.1-flash'
PROFILE = dict(profiler='torch', torch_profiler_dir=APP+'/diagnostic-traces',
               torch_profiler_with_stack=True, torch_profiler_record_shapes=True,
               torch_profiler_with_memory=False, ignore_frontend=False,
               detailed_trace_annotation=True, max_iterations=12)

REMOTE = r'''
import subprocess, json, re, datetime, hashlib, shlex
def cmd(a, timeout=60):
    return subprocess.check_output(a, text=True, timeout=timeout)
def save(obj):
    code='from pathlib import Path;p=Path('+repr(WORK+REL)+');p.mkdir(parents=True,exist_ok=True);(p/"launch.json").write_text('+repr(json.dumps(obj,indent=2)+'\n')+')'
    cmd(['docker','exec',STAGE,'python3','-c',code])
def npu_pids():
    snapshot=cmd(['docker','exec',STAGE,'npu-smi','info'])
    pids=set()
    for row in snapshot.splitlines():
        c=[x.strip() for x in row.split('|')]
        if len(c)>4 and re.fullmatch(r'\d+\s+\d+',c[1]) and c[2].isdigit(): pids.add(int(c[2]))
    assert pids or 'Process id' in snapshot, 'NPU process table unavailable'
    return pids,snapshot
baseline=json.loads(cmd(['docker','inspect',BASE]))[0]
assert baseline['Id']=='b00f4d2018e3d495d876b6e396ba9579e78b47a97111b99c27d6678e31039092'
assert baseline['Image']=='sha256:ce42d65b2403992ad2e2b06c61f4f048babb57e455cfe307b018345e36c5e98b'
assert baseline['State']['Running'], 'Baseline must be running at initial guard'
existing=cmd(['docker','ps','-a','--format','{{.Names}}']).splitlines()
assert DIAG not in existing, 'New diagnostic name already exists: inspect, do not overwrite'
own={int(x.strip()) for x in cmd(['docker','top',BASE,'-eo','pid']).splitlines() if x.strip().isdigit()}
pids,snapshot=npu_pids()
assert pids<=own, ('OTHER_NPU_PROCESSES',sorted(pids-own))
health='import urllib.request,json;op=urllib.request.build_opener(urllib.request.ProxyHandler({}));u="http://127.0.0.1:8172";status=op.open(u+"/health",timeout=10).status;raw=op.open(u+"/metrics",timeout=10).read().decode();rows=[x for x in raw.splitlines() if x.startswith(("vllm:num_requests_running{","vllm:num_requests_waiting{"))];assert status==200 and len(rows)>=4 and all(float(x.rsplit(" ",1)[1])==0 for x in rows),rows;print(json.dumps({"health":status,"idle":rows}))'
idle=json.loads(cmd(['docker','exec',BASE,'python3','-c',health]))
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
               'export WIO_DIAG_HMAC_KEY=wio-synthetic-recapture-169-20261029']
lines[idx+3]+=' --profiler-config '+shlex.quote(json.dumps(PROFILE,separators=(',',':')))
(dst/'scripts/serve-profile.sh').write_text('\n'.join(lines)+'\n')
for name,data in BUNDLE.items():(dst/name).write_text(data)
(root/'instrumentation-sha256.json').write_text(json.dumps({name:hashlib.sha256(data.encode()).hexdigest() for name,data in BUNDLE.items()},indent=2)+'\n')
print(json.dumps({'baseline_script_sha256':hashlib.sha256(text.encode()).hexdigest(),'diagnostic_script_sha256':hashlib.sha256((dst/'scripts/serve-profile.sh').read_bytes()).hexdigest(),'runtime':str(dst),'profile':PROFILE}))
"""
prepared=json.loads(cmd(['docker','exec',STAGE,'python3','-c','WORK='+repr(WORK)+'\nREL='+repr(REL)+'\nAPP='+repr(APP)+'\nPROFILE='+repr(PROFILE)+'\nBUNDLE='+repr(BUNDLE)+'\n'+prepare],timeout=300))
cfg=baseline['HostConfig'];args=['docker','create','--name',DIAG,'--label','wio.task=operator-ceiling169-20261029','--privileged','--read-only','--network','bridge','--ipc','private','--shm-size',str(cfg['ShmSize']),'-p','127.0.0.1:30077:8172','--workdir',APP,'--entrypoint','bash']
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
record={'state':'PREPARED','baseline_name':BASE,'baseline_id':baseline['Id'],'baseline_started_at':baseline['State']['StartedAt'],'diagnostic_name':DIAG,'diagnostic_id':diag_id,'image':baseline['Image'],'prepared':prepared,'idle_before':idle,'npu_before':snapshot,'created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'inference_semantics_unchanged':True,'inference_source_modified':False,'rollback':['docker','stop','--time','45',DIAG],'restore':['docker','start',BASE]}
save(record)
# Repeat guards immediately before the only service interruption.
own={int(x.strip()) for x in cmd(['docker','top',BASE,'-eo','pid']).splitlines() if x.strip().isdigit()}
pids,_=npu_pids();assert pids<=own,('OTHER_NPU_PROCESSES',sorted(pids-own))
cmd(['docker','exec',BASE,'python3','-c',health])
cmd(['docker','stop','--time','45',BASE],timeout=100)
try:
    pids,after=npu_pids()
    if pids:
        record.update(state='WAITING_OTHER_NPU_PROCESSES_AFTER_PARK',other_pids=sorted(pids))
        save(record)
        # Restoration would conflict with the new owner; do not start either model.
        raise RuntimeError('External NPU tasks appeared after own baseline was parked; record requires delayed restore')
    cmd(['docker','start',DIAG])
    record.update(state='DIAGNOSTIC_LOADING',npu_after_park=after)
    save(record)
except BaseException as exc:
    record['launch_error']=repr(exc)
    diag=json.loads(cmd(['docker','inspect',DIAG]))[0]
    if diag['State']['Running']:cmd(['docker','stop','--time','45',DIAG],timeout=100)
    pids,_=npu_pids()
    if not pids:
        cmd(['docker','start',BASE]);record['state']='BASELINE_RESTORING_AFTER_LAUNCH_ERROR'
    else:record['state']='WAITING_OTHER_NPU_PROCESSES_BEFORE_RESTORE'
    save(record)
    raise
print(json.dumps({k:v for k,v in record.items() if k not in ('npu_before','npu_after_park')}))
'''

def main():
    names=['instrumentation/sitecustomize.py','profile_client.py']
    bundle={name:(HERE/name).read_text() for name in names}
    for name,source in bundle.items():compile(source,name,'exec')
    constants=dict(BASE=BASE,STAGE=STAGE,DIAG=DIAG,ROOT=ROOT,WORK=WORK,REL=REL,APP=APP,PROFILE=PROFILE,BUNDLE=bundle)
    code='\n'.join(k+'='+repr(v) for k,v in constants.items())+'\n'+REMOTE
    result=subprocess.run([SSH,shlex.join(['python3','-c',code])],capture_output=True,text=True,timeout=600)
    (HERE/'launch-controller.stdout').write_text(result.stdout)
    (HERE/'launch-controller.stderr').write_text(result.stderr)
    if result.returncode:
        print(result.stderr[-5000:]);print(result.stdout[-1500:]);raise SystemExit(result.returncode)
    data=json.loads(result.stdout);(HERE/'launch.json').write_text(json.dumps(data,indent=2)+'\n')
    print(json.dumps(data,indent=2))

if __name__=='__main__':main()
