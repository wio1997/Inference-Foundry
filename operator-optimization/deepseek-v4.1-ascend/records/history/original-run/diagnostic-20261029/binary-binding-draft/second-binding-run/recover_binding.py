"""Reviewable recovery helper for the owned binding diagnostic. Not run automatically."""
from pathlib import Path
import json, os, shlex, subprocess, sys, time

if __name__ != '__main__' or sys.argv[1:] != ['--execute']:
    print('Review draft: no operations. Use --execute only to recover the owned binding diagnostic.')
    raise SystemExit(0)

HERE=Path(__file__).resolve().parent
SSH='/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh'
BASE='wio-vllm-v41-repro169-20261028'
BASE_ID='b00f4d2018e3d495d876b6e396ba9579e78b47a97111b99c27d6678e31039092'
DIAG='wio-vllm-v41-binding-diag2-169-20261029'
STAGE='wio-vllm-v41-repro169-stage-20261028'
RUN_ID=os.environ.get('WIO_BINDING_RUN_ID','')
assert len(RUN_ID)==32 and all(c in '0123456789abcdef' for c in RUN_ID), 'unique watchdog run ID required'
ROOT='/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029/binding-diagnostic-selective2'
WORK='/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029/binding-diagnostic-selective2'
REMOTE=r'''
import json,subprocess
fence_fd=acquire_owner_fence(ROOT.removesuffix("-selective2"))
def run(a,timeout=60):
    ensure_owner_fence(fence_fd);out=subprocess.check_output(a,text=True,timeout=timeout);ensure_owner_fence(fence_fd);return out
def saved_identity():
    for name in ('launch.json','identity-'+RUN_ID+'.json'):
        ensure_owner_fence(fence_fd)
        r=subprocess.run(['docker','exec',STAGE,'cat',WORK+'/'+name],capture_output=True,text=True,timeout=30)
        if r.returncode:continue
        try:x=json.loads(r.stdout)
        except (ValueError,TypeError):continue
        if x.get('baseline_id')==BASE_ID and x.get('diagnostic_name')==DIAG and x.get('binding_run_id')==RUN_ID:return x
    return None
launch=saved_identity()
b=json.loads(run(['docker','inspect',BASE_ID]))[0];assert b['Id']==BASE_ID
rows=run(['docker','ps','-a','--no-trunc','--format','{{.ID}}\t{{.Names}}']).splitlines()
found=[]
for row in rows:
    cols=row.split('\t')
    if len(cols)!=2:raise ValueError('malformed docker container list')
    if cols[1]==DIAG:found.append(cols[0])
if len(found)>1:raise ValueError('ambiguous diagnostic container name')
if found:
    diag_id=found[0]
    d=json.loads(run(['docker','inspect',diag_id]))[0]
    if d['Id']!=diag_id or d['Config']['Labels'].get('wio.task')!='operator-ceiling169-20261029' or d['Config']['Labels'].get('wio.binding_run_id')!=RUN_ID:raise ValueError('diagnostic identity or owner label mismatch')
    if launch is not None and launch.get('diagnostic_id')!=diag_id:raise ValueError('diagnostic identity does not match launch record')
    if d['State']['Running']:run(['docker','stop','--time','45',diag_id],timeout=100)
b=json.loads(run(['docker','inspect',BASE_ID]))[0];assert b['Id']==BASE_ID
# With no identity or diagnostic, an already running original baseline can be
# checked; a stopped baseline requires a valid saved empirical reference.
if b['State']['Running']:
    print(json.dumps({'state':'BASELINE_ALREADY_RUNNING','identity_source':'saved' if launch else 'labels_or_missing'}));raise SystemExit(0)
reference=launch.get('hbm_reference') if launch else None
release=wait_for_empirical_release(lambda:run(['docker','exec',STAGE,'npu-smi','info']),reference)
if release['satisfies_policy']:run(['docker','start',BASE_ID]);out={'state':'BASELINE_RESTORING'}
elif release.get('pids'):out={'state':'RESTORE_DEFERRED_OTHER_NPU_OWNER','other_pids':sorted(release['pids'])}
else:out={'state':'RESTORE_DEFERRED_UNATTRIBUTED_HBM','reason':release['reason']}
out['identity_source']='saved' if launch else 'labels_or_missing'
print(json.dumps(out))
'''
HEALTH=r'''
import json,subprocess,urllib.request
b=json.loads(subprocess.check_output(['docker','inspect',BASE_ID],text=True))[0];assert b['Id']==BASE_ID
check='import urllib.request;op=urllib.request.build_opener(urllib.request.ProxyHandler({}));print(op.open("http://127.0.0.1:8172/health",timeout=5).status)'
r=subprocess.run(['docker','exec',BASE_ID,'python3','-c',check],capture_output=True,text=True,timeout=15) if b['State']['Running'] else None
print(json.dumps({'running':b['State']['Running'],'health':int(r.stdout.strip()) if r and r.returncode==0 else None}))
'''

def ssh(code,timeout):
    pre='\n'.join(k+'='+repr(v) for k,v in dict(BASE=BASE,BASE_ID=BASE_ID,DIAG=DIAG,STAGE=STAGE,WORK=WORK,ROOT=ROOT,RUN_ID=RUN_ID).items())+'\n'
    r=subprocess.run([SSH,shlex.join(['python3','-c',pre+(HERE/'remote_guard.py').read_text()+'\n'+code])],capture_output=True,text=True,timeout=timeout)
    if r.returncode:raise RuntimeError(r.stderr[-2000:]+r.stdout[-1000:])
    return json.loads(r.stdout)

def save(obj):
    obj['checked_unix']=time.time()
    (HERE/'manual-recovery-state.json').write_text(json.dumps(obj,indent=2)+'\n')
    print(json.dumps(obj,ensure_ascii=False),flush=True)

try:
    result=ssh(REMOTE,400);save(result)
    if result['state'] in ('RESTORE_DEFERRED_OTHER_NPU_OWNER','RESTORE_DEFERRED_UNATTRIBUTED_HBM'):raise SystemExit(2)
    deadline=time.monotonic()+2400
    while time.monotonic()<deadline:
        try:
            health=ssh(HEALTH,30)
            if health['running'] and health['health']==200:
                save({'state':'BASELINE_HEALTHY','health':health});raise SystemExit(0)
        except Exception as e:save({'state':'BASELINE_HEALTH_PENDING','error':repr(e)})
        time.sleep(20)
    save({'state':'BASELINE_HEALTH_TIMEOUT'});raise SystemExit(3)
except Exception as e:
    save({'state':'RECOVERY_ERROR','error':repr(e)});raise
