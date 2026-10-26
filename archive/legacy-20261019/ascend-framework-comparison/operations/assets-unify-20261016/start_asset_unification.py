#!/usr/bin/env python3
"""Exactly once: launch finite file staging/Docker-import workers, never inference."""
from pathlib import Path
import json,hashlib,fcntl,os,subprocess,shlex,sys,time
A=Path(__file__).resolve().parent;sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert len(sys.argv)==2 and sha(A/'REVIEW.json')==sys.argv[1]
review=json.load(open(A/'REVIEW.json'))
for name,h in review['files'].items():assert sha(A/name)==h,name
lock=os.open(A/'launch.lock',os.O_CREAT|os.O_RDWR,0o600);fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not (A/'LAUNCH_STARTED.json').exists() and not (A/'LAUNCH.json').exists()
P=json.load(open(A/'PLAN.json'));assert set(map(int,P['nodes']))=={14,15,30,31,32,33,34,35,36,37,41,43,45,46,47}
SSH=['ssh','-S','/Users/wio/work/sglang-glm53-pd/ssh-control','-p','60011','-o','BatchMode=yes','-o','ConnectTimeout=20','root@61.241.77.34']
R='/data/wio/ascend-framework-comparison/assets/unify-20261016'
BOOT='''import sys,json,os,subprocess,hashlib
from pathlib import Path
j=json.load(sys.stdin);r=Path(j['root']);assert os.uname().nodename=='A3-5U-Atlas900RCKA3-1';assert not r.exists(),'Existing pipeline; do not duplicate';r.mkdir(parents=True)
for name,content in j['sources'].items():
 assert name in ['source_assets.py','node_assets.py'];p=r/name;p.write_text(content);assert hashlib.sha256(p.read_bytes()).hexdigest()==j['source_hashes'][name]
(r/'PLAN.json').write_text(json.dumps(j['plan'],indent=2)+'\\n')
with (r/'coordinator.log').open('wb') as log:p=subprocess.Popen(['python3','-u',str(r/'source_assets.py')],cwd=r,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
receipt={'pid':p.pid,'task_root':str(r),'phase':'RESOURCE_PIPELINE_LAUNCHED_NO_MODEL_START','port':28423};(r/'LAUNCH.json').write_text(json.dumps(receipt)+'\\n');print(json.dumps(receipt))
'''
(A/'LAUNCH_STARTED.json').write_text(json.dumps({'pid':os.getpid(),'epoch':time.time(),'scope':'FILES_AND_DOCKER_IMPORT_ONLY'})+'\n')
payload={'root':R,'plan':P,'sources':{n:(A/n).read_text() for n in ['source_assets.py','node_assets.py']},'source_hashes':{n:sha(A/n) for n in ['source_assets.py','node_assets.py']}}
p=subprocess.run(SSH+[shlex.join(['python3','-c',BOOT])],input=json.dumps(payload),capture_output=True,text=True,timeout=90);assert p.returncode==0,p.stderr[-2000:];source=json.loads(p.stdout)
with (A/'upload-worker.log').open('wb') as log:q=subprocess.Popen([sys.executable,'-u',str(A/'upload_assets.py')],cwd=A,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
r={'phase':'ASSET_UNIFICATION_LAUNCHED','source':source,'local_upload_pid':q.pid,'target_nodes':sorted(map(int,P['nodes'])),'excluded_nodes':[38,42],'models_started':False,'model_requests':0,'automations_changed':0,'epoch':time.time()};(A/'LAUNCH.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
