"""Generate review drafts from the previously tested owned-container workflow. Never launches."""
from pathlib import Path

here=Path(__file__).resolve().parent
parent=here.parent
old=(parent/'start_recapture.py').read_text()
old=old.replace("import json\n", "import json\nimport os\n")
old=old.replace("DIAG = 'wio-vllm-v41-ceiling-diag169-20261029'", "DIAG = 'wio-vllm-v41-binding-diag169-20261029'")
old=old.replace("REL = '/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029'", "REL = '/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029/binding-diagnostic'")
old=old.replace("'export WIO_DIAG_HMAC_KEY=wio-synthetic-recapture-169-20261029']", "'export WIO_DIAG_HMAC_KEY=wio-synthetic-recapture-169-20261029',\n               'export BINDING_TRACE_PATH='+shlex.quote(APP+'/metadata/binding-trace.jsonl'),\n               'export LD_PRELOAD='+shlex.quote(APP+'/instrumentation/libbinding_recorder.so')+':${LD_PRELOAD:-}']")
old=old.replace("lines[idx+3]+=' --profiler-config '", "lines[idx+5]+=' --profiler-config '")
old=old.replace("import subprocess, json, re, datetime, hashlib, shlex", "import subprocess, json, re, datetime, hashlib, shlex, time")
old=old.replace("cmd(['docker','stop','--time','45',BASE],timeout=100)", "lease=json.loads(cmd(['docker','exec',STAGE,'cat',WORK+REL+'/watchdog-lease.json']))\nassert lease.get('binding_run_id')==RUN_ID and 0 <= time.time()-lease.get('epoch',0) < 20, 'recovery watchdog lease missing or stale'\ncmd(['docker','stop','--time','45',BASE],timeout=100)")
old=old.replace("'inference_semantics_unchanged':True", "'inference_semantics_unchanged':None,'binding_run_id':RUN_ID")
old=old.replace("constants=dict(BASE=BASE,", "RUN_ID=os.environ.get('WIO_BINDING_RUN_ID','')\n    assert len(RUN_ID)==32 and all(c in '0123456789abcdef' for c in RUN_ID), 'unique watchdog run ID required'\n    constants=dict(RUN_ID=RUN_ID,BASE=BASE,")
needle="for name,data in BUNDLE.items():(dst/name).write_text(data)"
insert='''for name,data in BUNDLE.items():(dst/name).write_text(data)
# Build only inside this isolated runtime; no existing framework library changes.
source=Path(WORK+'/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029/binary-binding-draft/recorder.c')
import os
env=dict(os.environ);env['TMPDIR']=str(dst/'tmp')
subprocess.run(['gcc','-Wall','-Wextra','-Werror','-fPIC','-shared','-I/usr/local/Ascend/cann-9.2.0/aarch64-linux/include',str(source),'-o',str(dst/'instrumentation/libbinding_recorder.so'),'-ldl','-pthread'],check=True,env=env)
(root/'recorder-sha256.json').write_text(json.dumps({'source':hashlib.sha256(source.read_bytes()).hexdigest(),'library':hashlib.sha256((dst/'instrumentation/libbinding_recorder.so').read_bytes()).hexdigest()},indent=2)+'\\n')'''
assert needle in old
old=old.replace(needle,insert)
old=old.replace("bundle={name:(HERE/name).read_text() for name in names}","bundle={name:(HERE.parent/name).read_text() for name in names}")
old=old.replace("if __name__=='__main__':main()", "if __name__=='__main__':\n    if sys.argv[1:] != ['--execute']:\n        print('Review draft: no operations. Pass --execute only after reviewing recovery procedure.');raise SystemExit(0)\n    main()")
assert 'export BINDING_TRACE_PATH=' in old and "lines[idx+5]" in old
(here/'start_binding_recapture.py').write_text(old)
cap=(parent/'run_capture_and_restore.py').read_text()
cap=cap.replace("DIAG='wio-vllm-v41-ceiling-diag169-20261029'", "DIAG='wio-vllm-v41-binding-diag169-20261029'")
cap=cap.replace("REL='/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029'", "REL='/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029/binding-diagnostic'")
# The old script executes on import: require explicit opt-in before any state read or network operation.
cap=cap.replace("import urllib.error\n", "import urllib.error\nimport sys\nif __name__ != '__main__' or sys.argv[1:] != ['--execute']:\n    print('Review draft: no operations. Pass --execute only for an existing owned diagnostic.');raise SystemExit(0)\n")
(here/'supervise_binding_recapture.py').write_text(cap)
for name in ('start_binding_recapture.py','supervise_binding_recapture.py'):
    compile((here/name).read_text(),name,'exec')
print('Review drafts generated; no launch or network activity')

# Apply one remote lock/parser to start, capture and rollback. The guarded
# remote process retains the open lock FD until it exits, including on errors.
import re
start=here/'start_binding_recapture.py';s=start.read_text()
s=s.replace('import subprocess, json, re, datetime, hashlib, shlex, time\ndef cmd', 'import subprocess, json, re, datetime, hashlib, shlex, time\nfence_fd=acquire_owner_fence(ROOT+REL)\ndef cmd')
a=s.index('def save(obj):',s.index("REMOTE = r'''"));b=s.index('def npu_pids():',a)
s=s[:a]+'''def save(obj):
    data=json.dumps(obj,indent=2)+'\\n'
    code='\\n'.join(['from pathlib import Path','import os', 'p=Path('+repr(WORK+REL)+')', 'p.mkdir(parents=True,exist_ok=True)', 'data='+repr(data), 'identity=p/'+repr('identity-'+RUN_ID+'.json'), 'temporary=p/("launch.tmp."+str(os.getpid()))', 'backup=p/("identity.tmp."+str(os.getpid()))', 'if not identity.exists():', '    backup.write_text(data); os.replace(backup,identity)', 'temporary.write_text(data)', 'os.replace(temporary,p/"launch.json")'])
    cmd(['docker','exec',STAGE,'python3','-c',code])
''' +s[b:]
a=s.index('def npu_pids():',s.index("REMOTE = r'''"));b=s.index('baseline=json.loads',a)
s=s[:a]+'''def npu_pids():
    snapshot=cmd(['docker','exec',STAGE,'npu-smi','info'])
    return parse_npu_processes(snapshot),snapshot
''' +s[b:]
s=s.replace("'--label','wio.task=operator-ceiling169-20261029','--privileged'", "'--label','wio.task=operator-ceiling169-20261029','--label','wio.binding_run_id='+RUN_ID,'--privileged'")
s=s.replace("cmd(['docker','top',BASE,'-eo','pid'])", "cmd(['docker','top',baseline['Id'],'-eo','pid'])")
s=s.replace("cmd(['docker','exec',BASE,'python3','-c',health])", "cmd(['docker','exec',baseline['Id'],'python3','-c',health])")
s=s.replace("cmd(['docker','stop','--time','45',BASE],timeout=100)", "cmd(['docker','stop','--time','45',baseline['Id']],timeout=100)")
s=s.replace("cmd(['docker','start',DIAG])", "cmd(['docker','start',diag_id])")
s=s.replace("cmd(['docker','inspect',DIAG])", "cmd(['docker','inspect',diag_id])")
s=s.replace("cmd(['docker','stop','--time','45',DIAG],timeout=100)", "cmd(['docker','stop','--time','45',diag_id],timeout=100)")
s=s.replace("cmd(['docker','start',BASE]);record", "cmd(['docker','start',baseline['Id']]);record")
s=s.replace("def main():\n", "REMOTE=(HERE/'remote_guard.py').read_text()+'\\n'+REMOTE\n\ndef main():\n")
compile(s,str(start),'exec');start.write_text(s)

supervisor=here/'supervise_binding_recapture.py';s=supervisor.read_text()
# No data-changing path can run concurrently with the remote startup or recovery.
s=s.replace("import json,re,subprocess\nd=json.loads", "import json,re,subprocess\nfence_fd=acquire_owner_fence(ROOT+REL)\nd=json.loads",1)
s=s.replace("'APP='+repr(APP)+'\\nWORK='+repr(WORK)", "'APP='+repr(APP)+'\\nROOT='+repr(ROOT)+'\\nWORK='+repr(WORK)")
s=s.replace("APP='/data/tiankuan/wio/projects/910c-deepseek-v4.1-flash'", "APP='/data/tiankuan/wio/projects/910c-deepseek-v4.1-flash'\nROOT='/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-169-20261028'")
s=s.replace("prefix+code])", "prefix+(HERE/'remote_guard.py').read_text()+'\\n'+code])")
a=s.index("smi=subprocess.check_output(['docker','exec',STAGE,'npu-smi','info']",s.index("CAPTURE=r'''"));b=s.index("assert pids and pids<=own",a)
s=s[:a]+"smi=subprocess.check_output(['docker','exec',STAGE,'npu-smi','info'],text=True);pids=parse_npu_processes(smi)\n"+s[b:]
s=s.replace("['docker','top',DIAG,'-eo','pid']", "['docker','top',DIAG_ID,'-eo','pid']")
s=s.replace("['docker','exec',DIAG,'python3',APP+'/profile_client.py']", "['docker','exec',DIAG_ID,'python3',APP+'/profile_client.py']")
s=s.replace("import subprocess,json,re,datetime\ndef cmd", "import subprocess,json,re,datetime\nfence_fd=acquire_owner_fence(ROOT+REL)\ndef cmd")
s=s.replace("cmd(['docker','inspect',DIAG])", "cmd(['docker','inspect',DIAG_ID])")
s=s.replace("cmd(['docker','stop','--time','45',DIAG],timeout=100)", "cmd(['docker','stop','--time','45',DIAG_ID],timeout=100)")
a=s.index("snapshot=cmd(['docker','exec',STAGE,'npu-smi','info'])",s.index("RESTORE=r'''"));b=s.index("if baseline['State']['Running']",a)
s=s[:a]+"snapshot=cmd(['docker','exec',STAGE,'npu-smi','info']);pids=parse_npu_processes(snapshot)\n"+s[b:]
s=s.replace("cmd(['docker','start',BASE]);out", "cmd(['docker','start',baseline['Id']]);out")
# Snapshot must be refreshed after the stopped diagnostic before baseline state decision.
s=s.replace("snapshot=cmd(['docker','exec',STAGE,'npu-smi','info']);pids=parse_npu_processes(snapshot)\nif baseline", "snapshot=cmd(['docker','exec',STAGE,'npu-smi','info']);pids=parse_npu_processes(snapshot)\nbaseline=json.loads(cmd(['docker','inspect',BASE]))[0];assert baseline['Id']=='b00f4d2018e3d495d876b6e396ba9579e78b47a97111b99c27d6678e31039092'\nif baseline")
compile(s,str(supervisor),'exec');supervisor.write_text(s)
print('Generated drafts use shared strict parser, flock and container IDs')
# Include host ROOT in every supervisor SSH preamble and use immutable IDs on reads.
s=supervisor.read_text()
needle="+'\\nAPP='+repr(APP)+'\\nWORK='+repr(WORK)"
assert needle in s
s=s.replace(needle,"+'\\nAPP='+repr(APP)+'\\nROOT='+repr(ROOT)+'\\nWORK='+repr(WORK)")
s=s.replace("['docker','inspect',DIAG]", "['docker','inspect',DIAG_ID]")
s=s.replace("['docker','exec',DIAG,'python3'", "['docker','exec',DIAG_ID,'python3'")
compile(s,str(supervisor),'exec');supervisor.write_text(s)
s=start.read_text().replace("'rollback':['docker','stop','--time','45',DIAG],'restore':['docker','start',BASE]", "'rollback':['docker','stop','--time','45',diag_id],'restore':['docker','start',baseline['Id']]")
compile(s,str(start),'exec');start.write_text(s)
# A stage-container subprocess holds the common fence; check its liveness before
# and after owner operations. PID-empty alone is not evidence of HBM release.
s=start.read_text()
s=s.replace("def cmd(a, timeout=60):\n    return subprocess.check_output(a, text=True, timeout=timeout)", "def cmd(a, timeout=60):\n    ensure_owner_fence(fence_fd)\n    out=subprocess.check_output(a, text=True, timeout=timeout)\n    ensure_owner_fence(fence_fd)\n    return out")
s=s.replace("return parse_npu_processes(snapshot),snapshot", "parse_npu_hbm(snapshot)\n    return parse_npu_processes(snapshot),snapshot")
needle="    pids,after=npu_pids()\n    if pids:"
assert needle in s
s=s.replace(needle,"    pids,after=npu_pids()\n    deadline=time.monotonic()+60\n    while not pids and not verified_release(after)['proven_empty'] and time.monotonic()<deadline:\n        time.sleep(5);pids,after=npu_pids()\n    if not pids and not verified_release(after)['proven_empty']:\n        record.update(state='WAITING_UNATTRIBUTED_HBM',release_reason='UNATTRIBUTED_HBM')\n        save(record)\n        raise RuntimeError('No device PIDs but HBM is not proven released; do not start diagnostic')\n    if pids:")
needle="    pids,_=npu_pids()\n    if not pids:\n        cmd(['docker','start',baseline['Id']]);record['state']='BASELINE_RESTORING_AFTER_LAUNCH_ERROR'\n    else:record['state']='WAITING_OTHER_NPU_PROCESSES_BEFORE_RESTORE'"
assert needle in s
s=s.replace(needle,"    pids,restore_snapshot=npu_pids()\n    release=verified_release(restore_snapshot)\n    if release['proven_empty']:\n        cmd(['docker','start',baseline['Id']]);record['state']='BASELINE_RESTORING_AFTER_LAUNCH_ERROR'\n    elif pids:record['state']='WAITING_OTHER_NPU_PROCESSES_BEFORE_RESTORE'\n    else:record['state']='RESTORE_DEFERRED_UNATTRIBUTED_HBM'")
compile(s,str(start),'exec');start.write_text(s)

s=supervisor.read_text()
s=s.replace("def cmd(args,timeout=60):return subprocess.check_output(args,text=True,timeout=timeout)", "def cmd(args,timeout=60):\n    ensure_owner_fence(fence_fd);out=subprocess.check_output(args,text=True,timeout=timeout);ensure_owner_fence(fence_fd);return out")
s=s.replace("d=json.loads(subprocess.check_output(['docker','inspect',DIAG_ID],text=True))[0];assert d['Id']==DIAG_ID and d['State']['Running']", "ensure_owner_fence(fence_fd)\nd=json.loads(subprocess.check_output(['docker','inspect',DIAG_ID],text=True))[0];assert d['Id']==DIAG_ID and d['State']['Running']")
s=s.replace("r=subprocess.run(['docker','exec',DIAG_ID,'python3',APP+'/profile_client.py']", "ensure_owner_fence(fence_fd)\nr=subprocess.run(['docker','exec',DIAG_ID,'python3',APP+'/profile_client.py']")
s=s.replace("if baseline['State']['Running']:out={'state':'BASELINE_ALREADY_RUNNING'}\nelif pids:out={'state':'WAITING_OTHER_NPU_PROCESSES_BEFORE_RESTORE','other_pids':sorted(pids)}\nelse:cmd(['docker','start',baseline['Id']]);out={'state':'BASELINE_RESTORING'}", "if baseline['State']['Running']:out={'state':'BASELINE_ALREADY_RUNNING'}\nelif pids:out={'state':'WAITING_OTHER_NPU_PROCESSES_BEFORE_RESTORE','other_pids':sorted(pids)}\nelif not verified_release(snapshot)['proven_empty']:out={'state':'RESTORE_DEFERRED_UNATTRIBUTED_HBM'}\nelse:cmd(['docker','start',baseline['Id']]);out={'state':'BASELINE_RESTORING'}")
compile(s,str(supervisor),'exec');supervisor.write_text(s)
# A two-snapshot empirical memory envelope is built before parking and saved in
# both current and immutable identity records, so recovery can apply it too.
s=start.read_text()
s=s.replace('import datetime\nimport json\nimport os', 'import datetime\nimport hashlib\nimport json\nimport os',1)
needle="    constants=dict(RUN_ID=RUN_ID,BASE=BASE,"
insert="""    HIST1=Path('/Users/wio/work/169-sglang-deepseek-v41/evidence/001-npu-preflight.txt').read_text()
    HIST2=Path('/Users/wio/work/169-sglang-deepseek-v41/evidence/009-npu-occupancy-current.txt').read_text()
    assert hashlib.sha256(HIST1.encode()).hexdigest()=='fe22dc33030d27b8176492bb5f43cab4055fb6ddbb56df1b26b7a7b0eb1fba93'
    assert hashlib.sha256(HIST2.encode()).hexdigest()=='62886a47b5dd18b7954d79dc903bb37ee10c0a9eb1a27ac4cb49a9be78cdfd4e'
    constants=dict(HIST1=HIST1,HIST2=HIST2,RUN_ID=RUN_ID,BASE=BASE,"""
assert needle in s;s=s.replace(needle,insert)
needle="assert pids<=own, ('OTHER_NPU_PROCESSES',sorted(pids-own))\nhealth="
assert needle in s;s=s.replace(needle,"assert pids<=own, ('OTHER_NPU_PROCESSES',sorted(pids-own))\nreference=build_empirical_reference(snapshot,HIST1,HIST2,own)\nhealth=")
s=s.replace("'inference_semantics_unchanged':None,'binding_run_id':RUN_ID", "'inference_semantics_unchanged':None,'hbm_reference':reference,'binding_run_id':RUN_ID")
s=s.replace("pids,_=npu_pids();assert pids<=own,('OTHER_NPU_PROCESSES',sorted(pids-own))\ncmd(['docker','exec',baseline['Id'],'python3','-c',health])", "pids,current=npu_pids();assert pids<=own,('OTHER_NPU_PROCESSES',sorted(pids-own))\nreference=build_empirical_reference(current,HIST1,HIST2,own)\nrecord['hbm_reference']=reference;save(record)\ncmd(['docker','exec',baseline['Id'],'python3','-c',health])")
a=s.index("    pids,after=npu_pids()\n    deadline=time.monotonic()+60");b=s.index("    cmd(['docker','start',diag_id])",a)
s=s[:a]+'''    release=wait_for_empirical_release(lambda:cmd(['docker','exec',STAGE,'npu-smi','info']),reference)
    if not release['satisfies_policy']:
        record.update(state='WAITING_DEVICE_RELEASE',release_reason=release['reason'],other_pids=sorted(release.get('pids',[])))
        save(record)
        raise RuntimeError('Device release does not meet the observed baseline; not starting diagnostic')
    _,after=npu_pids()
    if not verified_release(after,reference)['satisfies_policy']:raise RuntimeError('Device status changed after release observation')
''' +s[b:]
a=s.index('    pids,restore_snapshot=npu_pids()');b=s.index('    save(record)',a)
s=s[:a]+'''    release=wait_for_empirical_release(lambda:cmd(['docker','exec',STAGE,'npu-smi','info']),reference)
    if release['satisfies_policy']:
        cmd(['docker','start',baseline['Id']]);record['state']='BASELINE_RESTORING_AFTER_LAUNCH_ERROR'
    elif release.get('pids'):record['state']='WAITING_OTHER_NPU_PROCESSES_BEFORE_RESTORE'
    else:record['state']='RESTORE_DEFERRED_UNATTRIBUTED_HBM'
''' +s[b:]
compile(s,str(start),'exec');start.write_text(s)

s=supervisor.read_text()
a=s.index("snapshot=cmd(['docker','exec',STAGE,'npu-smi','info']);pids=parse_npu_processes(snapshot)",s.index("RESTORE=r'''"));b=s.index("out['time']=",a)
s=s[:a]+'''baseline=json.loads(cmd(['docker','inspect',BASE]))[0];assert baseline['Id']=='b00f4d2018e3d495d876b6e396ba9579e78b47a97111b99c27d6678e31039092'
if baseline['State']['Running']:out={'state':'BASELINE_ALREADY_RUNNING'}
else:
    release=wait_for_empirical_release(lambda:cmd(['docker','exec',STAGE,'npu-smi','info']),REFERENCE)
    if release['satisfies_policy']:cmd(['docker','start',baseline['Id']]);out={'state':'BASELINE_RESTORING'}
    elif release.get('pids'):out={'state':'WAITING_OTHER_NPU_PROCESSES_BEFORE_RESTORE','other_pids':sorted(release['pids'])}
    else:out={'state':'RESTORE_DEFERRED_UNATTRIBUTED_HBM','reason':release['reason']}
''' +s[b:]
s=s.replace("DIAG_ID=launch['diagnostic_id']", "DIAG_ID=launch['diagnostic_id']\nREFERENCE=launch.get('hbm_reference')")
s=s.replace("+'\\nDIAG_ID='+repr(DIAG_ID)+'\\nAPP='", "+'\\nDIAG_ID='+repr(DIAG_ID)+'\\nREFERENCE='+repr(REFERENCE)+'\\nAPP='")
compile(s,str(supervisor),'exec');supervisor.write_text(s)
print('Empirical 16-chip envelope and two-observation release guard added')
# Persist both copies, then read them back and validate reference/IDs before park.
s=start.read_text();a=s.index('def save(obj):',s.index("REMOTE = r'''"));b=s.index('def npu_pids():',a)
s=s[:a]+'''def save(obj):
    data=json.dumps(obj,indent=2)+'\\n'
    code='\\n'.join(['from pathlib import Path','import os,tempfile', 'p=Path('+repr(WORK+REL)+')', 'p.mkdir(parents=True,exist_ok=True)', 'data='+repr(data), 'def commit(path):', '    fd,tmp=tempfile.mkstemp(prefix=".launch-",dir=p)', '    try:', '        with os.fdopen(fd,"w") as f:', '            f.write(data); f.flush(); os.fsync(f.fileno())', '        os.replace(tmp,path)', '    finally:', '        if os.path.exists(tmp): os.unlink(tmp)', 'commit(p/'+repr('identity-'+RUN_ID+'.json')+')', 'commit(p/"launch.json")', 'fd=os.open(p,os.O_RDONLY|os.O_DIRECTORY)', 'try: os.fsync(fd)', 'finally: os.close(fd)'])
    cmd(['docker','exec',STAGE,'python3','-c',code])
''' +s[b:]
needle="cmd(['docker','exec',baseline['Id'],'python3','-c',health])\nlease=json.loads"
assert needle in s
verify='''cmd(['docker','exec',baseline['Id'],'python3','-c',health])
verify='\\n'.join(['from pathlib import Path','import json', 'p=Path('+repr(WORK+REL)+')', 'ref='+repr(reference), 'expected_run='+repr(RUN_ID), 'baseline='+repr(baseline['Id']), 'diagnostic='+repr(diag_id), 'for name in ("launch.json",'+repr('identity-'+RUN_ID+'.json')+'):', '    x=json.loads((p/name).read_text())', '    assert x.get("binding_run_id")==expected_run and x.get("baseline_id")==baseline and x.get("diagnostic_id")==diagnostic and x.get("hbm_reference")==ref'])
cmd(['docker','exec',STAGE,'python3','-c',verify])
lease=json.loads'''
s=s.replace(needle,verify)
compile(s,str(start),'exec');start.write_text(s)
s=supervisor.read_text().replace('ssh(RESTORE,timeout=160)','ssh(RESTORE,timeout=400)')
compile(s,str(supervisor),'exec');supervisor.write_text(s)
# A deferred restoration is not successful completion.
s=supervisor.read_text();needle="        save(restored['state'],capture_error=None if error is None else repr(error))\n"
assert needle in s;s=s.replace(needle,needle+"        if restored['state'] not in ('BASELINE_RESTORING','BASELINE_ALREADY_RUNNING'):raise RuntimeError('Baseline recovery deferred: '+str(restored))\n")
compile(s,str(supervisor),'exec');supervisor.write_text(s)
