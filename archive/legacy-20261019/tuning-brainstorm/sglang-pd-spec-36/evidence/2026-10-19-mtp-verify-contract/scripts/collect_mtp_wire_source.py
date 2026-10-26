#!/usr/bin/env python3
"""Read-only graph source closure; no HTTP or model requests."""
import datetime,hashlib,json,pathlib,shlex,subprocess,sys
R=pathlib.Path(__file__).resolve().parents[1];O=R/'runs/L015-R007-wire-source';O.mkdir(parents=True,exist_ok=False);request=json.loads((R/'specs/L015-wire-request.json').read_text())
INNER=r'''
import pathlib,json,hashlib,subprocess,sys,urllib.request,urllib.error
j=json.loads(sys.argv[1]);root=pathlib.Path('/sgl-workspace/sglang')
head=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip();assert head==j['expected_commit'],'Observed source HEAD drift'
ps=set(j['paths'])
v={'source_commit':head,'files':{},'read_only':True,'generation_requests':0}
for n in sorted(ps):
 assert n.startswith('python/sglang/srt/') and '..' not in n
 p=root/n;b=p.read_bytes();q=subprocess.run(['git','-C',str(root),'show',head+':'+n],capture_output=True,timeout=15)
 v['files'][n]={'sha256':hashlib.sha256(b).hexdigest(),'matches_git_head':q.returncode==0 and q.stdout==b,'content':b.decode()}
print(json.dumps(v))
'''
NODE=r'''
import json,subprocess,sys,socket
j=json.load(sys.stdin);name='sglang-learning37-glm53-20261017'
cid=subprocess.check_output(['docker','inspect','--format','{{.Id}}',name],text=True,timeout=15).strip();assert cid==j['expected_container_id'],'Observed container identity drift'
p=subprocess.run(['docker','exec',name,'python3','-c',j['inner'],json.dumps(j)],capture_output=True,text=True,timeout=90)
assert p.returncode==0,p.stderr
v=json.loads(p.stdout);v['observed_hostname']=socket.gethostname();v['container_id']=cid;print(json.dumps(v))
'''
JUMP=r'''
import json,subprocess,sys
j=json.load(sys.stdin);pod='sglang-tuning-access-37-20261013'
p=subprocess.run(['kubectl','get','pod','-n','default',pod,'-o','json'],capture_output=True,text=True,timeout=20);assert p.returncode==0,p.stderr
v=json.loads(p.stdout);assert v['status']['hostIP']=='172.16.12.37' and v['status']['phase']=='Running'
p=subprocess.run(['kubectl','exec','-i','-n','default',pod,'-c','hostctl','--','nsenter','-t','1','-m','--','python3','-c',j['node']],input=json.dumps(j),capture_output=True,text=True,timeout=110)
sys.stdout.write(p.stdout);sys.stderr.write(p.stderr);sys.exit(p.returncode)
'''
cmd=['ssh','-S','/tmp/glm53-active-20261018.sock','-o','BatchMode=yes','-o','ConnectTimeout=8','-p','60011','root@61.241.77.34',shlex.join(['python3','-c',JUMP])]
request.update({'inner':INNER,'node':NODE});started=datetime.datetime.now(datetime.timezone.utc).isoformat();p=subprocess.run(cmd,input=json.dumps(request),capture_output=True,text=True,timeout=130)
(O/'source-and-config.json').write_text(p.stdout);(O/'stderr.log').write_text(p.stderr)
if p.returncode==0:
 for n,f in json.loads(p.stdout)['files'].items():
  target=R/'sources/installed'/n;target.parent.mkdir(parents=True,exist_ok=True)
  if target.exists():assert hashlib.sha256(target.read_bytes()).hexdigest()==f['sha256'],'Snapshot drift: '+n
  else:target.write_text(f['content'])
v={'started_utc':started,'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'command':cmd,'script_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'request_sha256':hashlib.sha256((R/'specs/L015-wire-request.json').read_bytes()).hexdigest(),'exit_code':p.returncode,'read_only':True,'generation_requests':0,'artifacts':{n:hashlib.sha256((O/n).read_bytes()).hexdigest() for n in ['source-and-config.json','stderr.log']}};(O/'receipt.json').write_text(json.dumps(v,indent=2)+'\n');print(json.dumps({'exit_code':p.returncode}));sys.exit(p.returncode)
