#!/usr/bin/env python3
"""Read-only inventory through an existing authenticated connection; no services started."""
import datetime, hashlib, json, pathlib, shlex, subprocess, sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/L001-R001-environment'
OUT.mkdir(parents=True, exist_ok=True)
assert not (OUT / 'receipt.json').exists(), 'Inspect previous receipt; never replay'
REMOTE = r'''
import json,subprocess,pathlib,hashlib,socket,datetime
def run(a,timeout=30):
 try:
  p=subprocess.run(a,capture_output=True,text=True,timeout=timeout)
  return {'command':a,'exit_code':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
 except subprocess.TimeoutExpired:
  return {'command':a,'exit_code':None,'error':'timeout'}
v={'hostname':socket.gethostname(),'observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'read_only':True}
v['npu']=run(['npu-smi','info'])
v['containers']=run(['docker','ps','-a','--format','{{json .}}'])
v['images']=run(['docker','images','--format','{{json .}}'])
v['model_candidates']=[]
for s in ['/data/wio/models/GLM-5.3-w8a8c8','/data/wio/GLM53/GLM-5.3-w8a8c8']:
 p=pathlib.Path(s);m={'path':s,'exists':p.exists()}
 if p.exists():
  m['files']=[]
  for n in ['config.json','quant_model_description.json','model.safetensors.index.json','tokenizer_config.json','generation_config.json']:
   f=p/n
   if f.is_file():
    b=f.read_bytes();e={'name':n,'size':len(b),'sha256':hashlib.sha256(b).hexdigest()}
    if n=='config.json':e['config']=json.loads(b)
    if n=='model.safetensors.index.json':
     j=json.loads(b);names=sorted(set(j.get('weight_map',{}).values()));e['metadata']=j.get('metadata',{})
     e['shards']=[{'name':n,'size':(p/n).stat().st_size if (p/n).is_file() else None} for n in names]
    m['files'].append(e)
 v['model_candidates'].append(m)
print(json.dumps(v))
'''
JUMP = r'''
import json,subprocess,sys
pod='sglang-tuning-access-37-20261013'
p=subprocess.run(['kubectl','get','pod','-n','default',pod,'-o','json'],capture_output=True,text=True,timeout=30)
assert p.returncode==0,p.stderr
j=json.loads(p.stdout)
assert j['status']['hostIP']=='172.16.12.37' and j['status']['phase']=='Running'
a=['kubectl','exec','-i','-n','default',pod,'-c','hostctl','--','nsenter','-t','1','-m','--','python3','-c',sys.stdin.read()]
p=subprocess.run(a,capture_output=True,text=True,timeout=180)
sys.stdout.write(p.stdout);sys.stderr.write(p.stderr);sys.exit(p.returncode)
'''
cmd=['ssh','-S','/tmp/glm53-active-20261018.sock','-o','BatchMode=yes','-o','ConnectTimeout=8','-p','60011','root@61.241.77.34',shlex.join(['python3','-c',JUMP])]
started=datetime.datetime.now(datetime.timezone.utc).isoformat()
p=subprocess.run(cmd,input=REMOTE,capture_output=True,text=True,timeout=210)
(OUT/'stdout.json').write_text(p.stdout)
(OUT/'stderr.log').write_text(p.stderr)
receipt={'started_utc':started,'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'script_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'command':cmd,'exit_code':p.returncode,'read_only':True,'services_started':False,'artifacts':{n:hashlib.sha256((OUT/n).read_bytes()).hexdigest() for n in ['stdout.json','stderr.log']}}
(OUT/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({'receipt':str(OUT/'receipt.json'),'exit_code':p.returncode}))
sys.exit(p.returncode)
