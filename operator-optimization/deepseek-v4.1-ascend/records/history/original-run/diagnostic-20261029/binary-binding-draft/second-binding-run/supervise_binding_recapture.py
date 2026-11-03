"""Bounded supervisor: wait for this diagnostic, capture once, restore baseline."""
from pathlib import Path
import datetime
import json
import os
import shlex
import subprocess
import time
import traceback
import urllib.error
import sys
if __name__ != '__main__' or sys.argv[1:] != ['--execute']:
    print('Review draft: no operations. Pass --execute only for an existing owned diagnostic.');raise SystemExit(0)

HERE=Path(__file__).resolve().parent
SSH='/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh'
BASE='wio-vllm-v41-repro169-20261028'
STAGE='wio-vllm-v41-repro169-stage-20261028'
DIAG='wio-vllm-v41-binding-diag2-169-20261029'
APP='/data/tiankuan/wio/projects/910c-deepseek-v4.1-flash'
ROOT='/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-169-20261028'
WORK='/work/vllm-ascend-reproduce-172-on-169-20261028'
REL='/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029/binding-diagnostic-selective2'
launch=json.loads((HERE/'launch.json').read_text())
RUN_ID=os.environ.get('WIO_BINDING_RUN_ID','')
assert len(RUN_ID)==32 and all(c in '0123456789abcdef' for c in RUN_ID)
assert launch['binding_run_id']==RUN_ID and launch['diagnostic_name']==DIAG
DIAG_ID=launch['diagnostic_id']
REFERENCE=launch.get('hbm_reference')
state={'state':'WAITING_DIAGNOSTIC_HEALTH','diagnostic_id':DIAG_ID,'capture_attempted':False,'baseline_restored':False}
def save(label, **fields):
    state.update(state=label,updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),**fields)
    (HERE/'supervisor.json').write_text(json.dumps(state,indent=2)+'\n')
    print(label,flush=True)
def ssh(code,timeout=90):
    prefix='BASE='+repr(BASE)+'\nSTAGE='+repr(STAGE)+'\nDIAG='+repr(DIAG)+'\nDIAG_ID='+repr(DIAG_ID)+'\nRUN_ID='+repr(RUN_ID)+'\nREFERENCE='+repr(REFERENCE)+'\nAPP='+repr(APP)+'\nROOT='+repr(ROOT)+'\nWORK='+repr(WORK)+'\nREL='+repr(REL)+'\n'
    result=subprocess.run([SSH,shlex.join(['python3','-c',prefix+(HERE/'remote_guard.py').read_text()+'\n'+code])],capture_output=True,text=True,timeout=timeout)
    if result.returncode:raise RuntimeError(result.stderr[-4500:]+result.stdout[-1500:])
    return result.stdout
STATUS=r'''
import json,subprocess
d=json.loads(subprocess.check_output(['docker','inspect',DIAG_ID],text=True))[0];assert d['Id']==DIAG_ID and d['Config']['Labels'].get('wio.binding_run_id')==RUN_ID
check='import urllib.request,json;op=urllib.request.build_opener(urllib.request.ProxyHandler({}));u="http://127.0.0.1:8172";print(json.dumps({"health":op.open(u+"/health",timeout=5).status}))'
r=subprocess.run(['docker','exec',DIAG_ID,'python3','-c',check],capture_output=True,text=True,timeout=15) if d['State']['Running'] else None
print(json.dumps({'running':d['State']['Running'],'exit_code':d['State']['ExitCode'],'started_at':d['State']['StartedAt'],'health':json.loads(r.stdout)['health'] if r and r.returncode==0 else None}))
'''
CAPTURE=r'''
import json,re,subprocess
fence_fd=acquire_owner_fence(ROOT+'/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029/binding-diagnostic')
ensure_owner_fence(fence_fd)
d=json.loads(subprocess.check_output(['docker','inspect',DIAG_ID],text=True))[0];assert d['Id']==DIAG_ID and d['Config']['Labels'].get('wio.binding_run_id')==RUN_ID and d['State']['Running']
own={int(x.strip()) for x in subprocess.check_output(['docker','top',DIAG_ID,'-eo','pid'],text=True).splitlines() if x.strip().isdigit()}
smi=subprocess.check_output(['docker','exec',STAGE,'npu-smi','info'],text=True);pids=parse_npu_processes(smi)
assert pids and pids<=own,('OTHER_NPU_PROCESSES_BEFORE_CAPTURE',sorted(pids-own))
ensure_owner_fence(fence_fd)
r=subprocess.run(['docker','exec',DIAG_ID,'python3',APP+'/profile_client.py'],capture_output=True,text=True,timeout=2400)
print(r.stdout)
if r.returncode:raise RuntimeError(r.stderr[-5000:])
'''
RAW_PROBE=r'''
import subprocess
r=subprocess.run(['docker','exec',DIAG_ID,'python3',APP+'/raw_profile_probe.py',APP],capture_output=True,text=True,timeout=90)
if r.returncode:raise RuntimeError(r.stderr[-3000:])
print(r.stdout)
'''
SUMMARY=r'''
import subprocess
code=r"""from pathlib import Path
import collections,hashlib,json
p=Path(APP);m=p/'metadata';counts=collections.Counter();errors=[];schedules=[];labels=collections.Counter()
for file in m.glob('events-*.jsonl'):
    for line in file.open():
        try:row=json.loads(line)
        except json.JSONDecodeError:continue
        counts[row['kind']]+=1
        if row['kind']=='instrumentation_error':errors.append(row)
        if row.get('label'):labels[row['label']]+=1
        if row.get('schedule'):schedules.append({'pid':row['pid'],'kind':row['kind'],'label':row.get('label'),'schedule':row['schedule']})
progress=json.loads((p/'profile-client-results/progress.json').read_text())
infos=[]
for file in (p/'diagnostic-traces').glob('*/profiler_info*.json'):
    infos.append({'path':str(file),'data':json.loads(file.read_text())})
out={'progress':progress,'metadata_event_counts':dict(counts),'metadata_labels':dict(labels),'instrumentation_errors':errors[:100],'metadata_file_count':len(list(m.glob('events-*.jsonl'))),'profiler_info_count':len(infos),'profiler_info':infos,'schedule_samples':schedules[:10],'trace_root':str(p/'diagnostic-traces')}
print(json.dumps(out))
"""
r=subprocess.run(['docker','exec',DIAG_ID,'python3','-c','APP='+repr(APP)+'\n'+code],capture_output=True,text=True,timeout=90)
if r.returncode:raise RuntimeError(r.stderr[-3000:])
print(r.stdout)
'''
EXPORT_STATUS=r'''
import subprocess
code='from pathlib import Path;import json;p=Path('+repr(APP+'/diagnostic-traces')+');roots=[x for x in p.glob("*_ascend_pt") if x.is_dir()];print(json.dumps({"worker_roots":len(roots),"kernel_details":sum((x/"ASCEND_PROFILER_OUTPUT/kernel_details.csv").is_file() for x in roots),"trace_view":sum((x/"ASCEND_PROFILER_OUTPUT/trace_view.json").is_file() for x in roots)}))'
r=subprocess.run(['docker','exec',DIAG_ID,'python3','-c',code],capture_output=True,text=True,timeout=30)
if r.returncode:raise RuntimeError(r.stderr[-1500:])
print(r.stdout)
'''
RESTORE=r'''
import subprocess,json,re,datetime
fence_fd=acquire_owner_fence(ROOT+'/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029/binding-diagnostic')
def cmd(args,timeout=60):
    ensure_owner_fence(fence_fd);out=subprocess.check_output(args,text=True,timeout=timeout);ensure_owner_fence(fence_fd);return out
baseline=json.loads(cmd(['docker','inspect',BASE]))[0];d=json.loads(cmd(['docker','inspect',DIAG_ID]))[0]
assert baseline['Id']=='b00f4d2018e3d495d876b6e396ba9579e78b47a97111b99c27d6678e31039092'
assert d['Id']==DIAG_ID and d['Config']['Labels'].get('wio.task')=='operator-ceiling169-20261029' and d['Config']['Labels'].get('wio.binding_run_id')==RUN_ID
if d['State']['Running']:cmd(['docker','stop','--time','45',DIAG_ID],timeout=100)
baseline=json.loads(cmd(['docker','inspect',BASE]))[0];assert baseline['Id']=='b00f4d2018e3d495d876b6e396ba9579e78b47a97111b99c27d6678e31039092'
if baseline['State']['Running']:out={'state':'BASELINE_ALREADY_RUNNING'}
else:
    release=wait_for_empirical_release(lambda:cmd(['docker','exec',STAGE,'npu-smi','info']),REFERENCE)
    if release['satisfies_policy']:cmd(['docker','start',baseline['Id']]);out={'state':'BASELINE_RESTORING'}
    elif release.get('pids'):out={'state':'WAITING_OTHER_NPU_PROCESSES_BEFORE_RESTORE','other_pids':sorted(release['pids'])}
    else:out={'state':'RESTORE_DEFERRED_UNATTRIBUTED_HBM','reason':release['reason']}
out['time']=datetime.datetime.now(datetime.timezone.utc).isoformat()
save='from pathlib import Path;Path('+repr(WORK+REL+'/restore.json')+').write_text('+repr(json.dumps(out,indent=2)+'\n')+')'
cmd(['docker','exec',STAGE,'python3','-c',save]);print(json.dumps(out))
'''

error=None
try:
    save('WAITING_DIAGNOSTIC_HEALTH')
    deadline=time.monotonic()+2400
    while time.monotonic()<deadline:
        snapshot=json.loads(ssh(STATUS,timeout=30));state['last_health']=snapshot
        (HERE/'supervisor.json').write_text(json.dumps(state,indent=2)+'\n')
        if not snapshot['running']:raise RuntimeError('Diagnostic container exited: '+str(snapshot))
        if snapshot['health']==200:break
        time.sleep(20)
    else:raise TimeoutError('Diagnostic health did not become ready within 40 minutes')
    save('CAPTURE_RUNNING',capture_attempted=True)
    output=ssh(CAPTURE,timeout=2500);(HERE/'capture.stdout').write_text(output)
    save('CAPTURE_COMPLETE')
    # This image requires offline analysis; never wait for online export.
    state['export_incomplete_requires_offline_parse']=True
    save('OFFLINE_EXPORT_REQUIRED',raw_profile_path=WORK+REL+'/runtime/diagnostic-traces')
    try:
        raw=json.loads(ssh(RAW_PROBE,timeout=120))
        (HERE/'raw-profile-readiness.json').write_text(json.dumps(raw,indent=2)+'\n')
        save(raw['status'],raw_profile_status=raw['status'],raw_profile_visibility_complete=(raw['status']=='RAW_VISIBILITY_STABLE_CLUE'),raw_profile_readiness=str(HERE/'raw-profile-readiness.json'))
    except Exception as raw_error:
        (HERE/'raw-profile-readiness.json').write_text(json.dumps({'status':'RAW_CHECK_ERROR','error':repr(raw_error)},indent=2)+'\n')
        save('RAW_CHECK_ERROR',raw_profile_status='RAW_CHECK_ERROR',raw_profile_visibility_complete=False,raw_error=repr(raw_error))
    summary=json.loads(ssh(SUMMARY,timeout=120))
    (HERE/'capture-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    save('EXPORT_SUMMARY_SAVED',summary_path=str(HERE/'capture-summary.json'))
except BaseException as exc:
    error=exc
    (HERE/'supervisor-error.txt').write_text(traceback.format_exc())
    save('CAPTURE_FAILED',error=repr(exc))
finally:
    try:
        restored=json.loads(ssh(RESTORE,timeout=400))
        (HERE/'restore.json').write_text(json.dumps(restored,indent=2)+'\n')
        save(restored['state'],capture_error=None if error is None else repr(error))
        if restored['state'] not in ('BASELINE_RESTORING','BASELINE_ALREADY_RUNNING'):raise RuntimeError('Baseline recovery deferred: '+str(restored))
    except BaseException as restore_exc:
        save('RESTORE_ERROR',restore_error=repr(restore_exc))
        raise
if error:raise error
print(json.dumps({'capture_client_complete':True,'raw_profile_status':state.get('raw_profile_status','UNKNOWN'),'raw_profile_visibility_complete':state.get('raw_profile_visibility_complete',False),'offline_analysis_completed':False,'restore':restored}))
