#!/usr/bin/env python3
from pathlib import Path
import json,hashlib,subprocess,shlex,os,fcntl,sys,signal,time
A=Path(__file__).resolve().parent;sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert len(sys.argv)==2 and sha(A/'REVIEW_V2.json')==sys.argv[1]
review=json.load(open(A/'REVIEW_V2.json'))
for name,h in review['files'].items():assert sha(A/name)==h,name
lock=os.open(A/'resume2.lock',os.O_CREAT|os.O_RDWR,0o600);fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not (A/'RESUME_V2_LAUNCH.json').exists()
R='/data/wio/ascend-framework-comparison/assets/unify-20261016';SSH=['ssh','-S','/Users/wio/work/sglang-glm53-pd/ssh-control','-p','60011','-o','BatchMode=yes','-o','ConnectTimeout=20','root@61.241.77.34']
# Stop only the exact local uploader session. Its remote partial remains reusable.
old=json.load(open(A/'LAUNCH.json'))['local_upload_pid'];cmd=subprocess.check_output(['ps','-p',str(old),'-o','command='],text=True);assert str(A/'upload_assets.py') in cmd and os.getpgid(old)==old
os.killpg(old,signal.SIGTERM)
for _ in range(60):
 p=subprocess.run(['pgrep','-g',str(old)],capture_output=True,text=True)
 if p.returncode:break
 time.sleep(.25)
else:raise RuntimeError('Old upload session not fully exited; do not duplicate')
BOOT='''from pathlib import Path
import sys,json,os,hashlib,subprocess
j=json.load(sys.stdin);r=Path(j['root']);assert os.uname().nodename=='A3-5U-Atlas900RCKA3-1';old=json.load(open(r/'LAUNCH.json'));assert not Path('/proc/'+str(old['pid'])).exists();assert json.load(open(r/'STATUS.json'))['phase']=='FAILED_SOURCE_PRESERVE_NODE_STATES'
assert hashlib.sha256((r/'FLASH_SHA256.json').read_bytes()).hexdigest()==j['checkpoint_sha256'];assert not (r/'source_assets_v2.py').exists()
# Confirm exactly the15 earlier workers are live before resuming their source.
p=json.load(open(r/'PLAN.json'));launches=json.load(open(r/'NODE_LAUNCHES.json'))
for e in launches:
 cfg=p['nodes'][str(e['node'])];code='from pathlib import Path;import os;pid='+str(e['pid'])+';assert Path("/proc/"+str(pid)).exists();assert b"node_assets.py" in Path("/proc/"+str(pid)+"/cmdline").read_bytes()'
 out=subprocess.run(['kubectl','exec','-n','default',cfg['pod'],'-c','hostctl','--','nsenter','-t','1','-m','--','python3','-c',code],capture_output=True,text=True,timeout=20);assert out.returncode==0,out.stderr[-1000:]
p['repair_flash_checkpoint_sha256']=j['checkpoint_sha256'];(r/'PLAN_V1_PRESERVED.json').write_bytes((r/'PLAN.json').read_bytes());(r/'PLAN.json').write_text(json.dumps(p,indent=2)+'\\n');(r/'source_assets_v2.py').write_text(j['source']);assert hashlib.sha256((r/'source_assets_v2.py').read_bytes()).hexdigest()==j['source_sha256']
with (r/'coordinator-v2.log').open('wb') as log:q=subprocess.Popen(['python3','-u',str(r/'source_assets_v2.py')],cwd=r,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
rj={'pid':q.pid,'phase':'SOURCE_V2_REEXPORT_AND_REUSE_EXISTING15_WORKERS','root':str(r)};(r/'LAUNCH_V2.json').write_text(json.dumps(rj)+'\\n');print(json.dumps(rj))
'''
payload={'root':R,'checkpoint_sha256':review['flash_checkpoint_sha256'],'source':(A/'source_assets_v2.py').read_text(),'source_sha256':sha(A/'source_assets_v2.py')}
p=subprocess.run(SSH+[shlex.join(['python3','-c',BOOT])],input=json.dumps(payload),capture_output=True,text=True,timeout=90);assert p.returncode==0,p.stderr[-2000:];remote=json.loads(p.stdout)
with (A/'upload-worker-v2.log').open('wb') as log:q=subprocess.Popen([sys.executable,'-u',str(A/'upload_assets_compressed_v2.py')],cwd=A,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
r={'phase':'ASSET_SOURCE_REPAIRED_AND_COMPRESSED_UPLOAD_RESUMED','remote':remote,'new_upload_pid':q.pid,'old_upload_pid':old,'flash_hashes_reused_files':97,'existing_node_workers_reused':15,'model_requests':0,'models_started':False,'epoch':time.time()};(A/'RESUME_V2_LAUNCH.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
