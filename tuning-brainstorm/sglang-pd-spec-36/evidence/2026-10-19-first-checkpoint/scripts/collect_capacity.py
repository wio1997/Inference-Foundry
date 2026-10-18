#!/usr/bin/env python3
"""Read-only NPU/container inventory on existing access pods, bounded concurrency."""
import datetime,hashlib,json,pathlib,shlex,subprocess,sys
ROOT=pathlib.Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/L001-R003-capacity';OUT.mkdir(parents=True,exist_ok=False)
NODE=r'''
import subprocess,json,socket
v={'hostname':socket.gethostname()}
for k,a in [('npu',['npu-smi','info']),('containers',['docker','ps','--format','{{.ID}}\t{{.Names}}\t{{.Image}}'])]:
 p=subprocess.run(a,capture_output=True,text=True,timeout=30);v[k]={'command':a,'exit_code':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
print(json.dumps(v))
'''
JUMP=r'''
import subprocess,json,sys,concurrent.futures
j=json.load(sys.stdin)
p=subprocess.run(['kubectl','get','pods','-n','default','-o','json'],capture_output=True,text=True,timeout=30);assert p.returncode==0,p.stderr
items=json.loads(p.stdout)['items'];targets=[]
for x in items:
 n=x['metadata']['name']
 if x['status'].get('phase')!='Running' or not n.startswith('sglang-'):continue
 if not any(t in n for t in ['tuning-access','container-access','node-access']):continue
 containers=[c['name'] for c in x['spec']['containers']]
 if 'hostctl' not in containers:continue
 targets.append((n,x['status']['hostIP']))
def one(t):
 n,ip=t;a=['kubectl','exec','-i','-n','default',n,'-c','hostctl','--','nsenter','-t','1','-m','--','python3','-c',j['node']]
 try:
  p=subprocess.run(a,capture_output=True,text=True,timeout=65)
  return {'pod':n,'host_ip':ip,'exit_code':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
 except subprocess.TimeoutExpired:return {'pod':n,'host_ip':ip,'exit_code':None,'error':'timeout'}
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:v=list(pool.map(one,targets))
print(json.dumps(v))
'''
cmd=['ssh','-S','/tmp/glm53-active-20261018.sock','-o','BatchMode=yes','-o','ConnectTimeout=8','-p','60011','root@61.241.77.34',shlex.join(['python3','-c',JUMP])]
started=datetime.datetime.now(datetime.timezone.utc).isoformat();p=subprocess.run(cmd,input=json.dumps({'node':NODE}),capture_output=True,text=True,timeout=400)
(OUT/'capacity.json').write_text(p.stdout);(OUT/'stderr.log').write_text(p.stderr)
receipt={'started_utc':started,'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'script_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'command':cmd,'exit_code':p.returncode,'read_only':True,'services_started':False,'artifacts':{n:hashlib.sha256((OUT/n).read_bytes()).hexdigest() for n in ['capacity.json','stderr.log']}}
(OUT/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({'receipt':str(OUT/'receipt.json'),'exit_code':p.returncode}));sys.exit(p.returncode)
