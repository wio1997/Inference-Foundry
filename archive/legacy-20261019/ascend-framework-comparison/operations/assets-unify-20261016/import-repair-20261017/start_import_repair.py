#!/usr/bin/env python3
"""Exactly once deploy a reviewed B103 Docker-import repair, not inference."""
from pathlib import Path
import json,hashlib,subprocess,os,fcntl,sys,shlex,time
Q=Path(__file__).resolve().parent;h=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();assert len(sys.argv)==2 and h(Q/'REVIEW.json')==sys.argv[1];review=json.load(open(Q/'REVIEW.json'))
for n,s in review['files'].items():assert h(Q/n)==s,n
lock=os.open(Q/'launch.lock',os.O_CREAT|os.O_RDWR,0o600);fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not (Q/'LAUNCH_STARTED.json').exists() and not (Q/'LAUNCH.json').exists()
SSH=['ssh','-S','/Users/wio/work/sglang-glm53-pd/ssh-control','-p','60011','-o','BatchMode=yes','-o','ConnectTimeout=15','root@61.241.77.34'];remote='/data/wio/ascend-framework-comparison/assets/unify-20261016/import-repair-20261017'
boot='''from pathlib import Path
import json,sys,hashlib,os,subprocess
j=json.load(sys.stdin);q=Path(j['root']);assert os.uname().nodename=='A3-5U-Atlas900RCKA3-1';assert not q.exists(),'Existing repair: do not duplicate';assert not Path('/proc/53691').exists();q.mkdir()
for n,t in j['files'].items():
 assert n in ['source_repair.py','node_import.py','PINNED.json'];(q/n).write_text(t);assert hashlib.sha256((q/n).read_bytes()).hexdigest()==j['hashes'][n]
with (q/'coordinator.log').open('xb') as log:p=subprocess.Popen(['python3','-u',str(q/'source_repair.py')],cwd=q,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
v={'pid':p.pid,'phase':'B103_SIX_NODE_IMPORT_REPAIR_LAUNCHED','root':str(q),'models_started':False};(q/'LAUNCH.json').write_text(json.dumps(v,indent=2)+'\\n');print(json.dumps(v))
'''
payload={'root':remote,'files':{n:(Q/n).read_text() for n in ['source_repair.py','node_import.py','PINNED.json']},'hashes':{n:h(Q/n) for n in ['source_repair.py','node_import.py','PINNED.json']}}
(Q/'LAUNCH_STARTED.json').write_text(json.dumps({'pid':os.getpid(),'epoch':time.time()})+'\n');p=subprocess.run(SSH+[shlex.join(['python3','-c',boot])],input=json.dumps(payload),capture_output=True,text=True,timeout=45);assert p.returncode==0,p.stderr[-1500:];v=json.loads(p.stdout);(Q/'LAUNCH.json').write_text(json.dumps(v,indent=2)+'\n');print(json.dumps(v))
