#!/usr/bin/env python3
"""Read exact own stopped-worker plogs and installed DeepEP strategies on910C."""
import datetime, hashlib, json, pathlib, shlex, subprocess, sys
R = pathlib.Path(__file__).resolve().parents[1]
O = R / 'runs/L039-R049-tiling-details'
O.mkdir(parents=True, exist_ok=False)
HOST = r'''
import hashlib,json,pathlib,re,subprocess,sys
j=json.load(sys.stdin); base=pathlib.Path('/data/wio/sglang-mech36-20261018')
assert json.loads((base/'owner.json').read_text())['task']=='sglang-pd-spec-36'
root=pathlib.Path(j['root']); assert root.resolve().parent==base.resolve()
p=subprocess.run(['docker','inspect','--format','{{json .}}',j['container']],capture_output=True,text=True,timeout=25)
assert p.returncode==0,p.stderr
x=json.loads(p.stdout); assert x['Id']==j['container_id'] and x['Config']['Labels']['task']=='sglang-pd-spec-36' and x['Config']['Labels']['run']=='L039-R044' and not x['State']['Running']
logs=root/'ascend-logs'; fs=sorted((logs/'run/plog').glob('*.log')); assert fs
v={'node':j['node'],'container_id':x['Id'],'representative_run_plog':{},'critical_lines':[],'tiling_library_lines':[],'model_config':None,'read_only':True,'NPU_computation':False}
seen=set(); loads=set()
for f in fs:
 assert f.resolve().is_relative_to(logs.resolve()) and not f.is_symlink()
 before=f.stat(); data=f.read_bytes(); after=f.stat(); assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
 lines=data.decode(errors='replace').splitlines()
 if f==fs[0]:
  assert len(data)<512*1024
  v['representative_run_plog']={'path':str(f),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'text':data.decode(errors='replace')}
 for i,line in enumerate(lines):
  # Drop the runtime prefix to deduplicate the16rank messages by actual body.
  body=re.sub(r'^.*?\[\d+\]\s*','',line)
  body=re.sub(r'\b(?:0x[0-9a-fA-F]+|\d{8,})\b','ID',body)
  if ('[ERROR]' in line or '[WARN]' in line or '[WARNING]' in line) and body not in seen:
   seen.add(body)
   if len(v['critical_lines'])<180:v['critical_lines'].append({'file':str(f.relative_to(logs)),'line':i+1,'text':line[:6000]})
  if any(k in line.lower() for k in ['loadso','tiling.so','opsproto','custom_opp','hwcomputing','optiling','optilingmanager']) and body not in loads:
   loads.add(body)
   if len(v['tiling_library_lines'])<80:v['tiling_library_lines'].append({'file':str(f.relative_to(logs)),'line':i+1,'text':line[:6000]})
model=pathlib.Path(j['model_path']); assert model.resolve().is_relative_to(base.resolve())
f=model/'config.json'; data=f.read_bytes(); assert hashlib.sha256(data).hexdigest()=='ca8f2f47b07919a514c0ca223dc2ea2bc7445afaa5ac76c013a3784e096426ca'
v['model_config']={'path':str(f),'sha256':hashlib.sha256(data).hexdigest(),'config':json.loads(data)}
print(json.dumps(v))
'''
IMAGE = r'''
import hashlib,importlib.util,json,pathlib
spec=importlib.util.find_spec('deep_ep'); pkg=pathlib.Path(spec.origin).parent
v={'package_root':str(pkg),'sources':{},'NPU_device_nodes_absent':not any(pathlib.Path('/dev').glob('davinci*'))}
files=sorted((pkg/'strategies').rglob('*.py'))+[pkg/'utils.py']
assert len(files)<=30
for p in files:
 data=p.read_bytes(); assert len(data)<256*1024
 v['sources'][str(p.relative_to(pkg))]={'path':str(p),'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data),'source':data.decode()}
print(json.dumps(v))
'''
JUMP = r'''
import concurrent.futures,json,shlex,subprocess,sys
j=json.load(sys.stdin); v={'nodes':{},'resource_leases':{},'image_metadata':None,'read_only':True,'NPU_computation':False}
for n,lease in j['leases'].items():
 p=subprocess.run(['kubectl','get','pod','-n','default',lease['name'],'-o','json'],capture_output=True,text=True,timeout=25); assert p.returncode==0,p.stderr
 x=json.loads(p.stdout); assert x['metadata']['uid']==lease['uid'] and x['metadata']['labels']['task']=='sglang-pd-spec-36'
 v['resource_leases'][n]={'name':x['metadata']['name'],'uid':x['metadata']['uid'],'phase':x['status']['phase'],'ready':next((c['status'] for c in x['status'].get('conditions',[]) if c['type']=='Ready'),None)}
def host(item):
 n=item['node']; p=subprocess.run(['kubectl','exec','-i','-n','default',f'sglang-container-access-{n}-20261013','-c','hostctl','--','nsenter','-t','1','-m','-n','--','python3','-c',j['host']],input=json.dumps(item),capture_output=True,text=True,timeout=100)
 return str(n),{'exit_code':p.returncode,'data':json.loads(p.stdout) if p.stdout else None,'stderr':p.stderr}
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:v['nodes']=dict(pool.map(host,j['targets']))
code=r"""
import json,shlex,subprocess,sys
j=json.load(sys.stdin); p=subprocess.run(['docker','inspect','--format','{{json .}}',j['store']['container']],capture_output=True,text=True,timeout=25); assert p.returncode==0,p.stderr
x=json.loads(p.stdout); assert x['Id']==j['store']['container_id'] and x['State']['Running'] and x['Config']['Labels']['task']=='sglang-pd-spec-36' and x['Config']['Labels']['run']=='L039-R033' and not x['HostConfig']['Devices']
setup='set -e; if [ -r /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then . /usr/local/Ascend/ascend-toolkit/set_env.sh; elif [ -r /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash ]; then . /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash; fi; exec python3 -c '+shlex.quote(j['image'])
p=subprocess.run(['docker','exec',j['store']['container'],'/bin/bash','-c',setup],capture_output=True,text=True,timeout=60); sys.stdout.write(p.stdout); sys.stderr.write(p.stderr); sys.exit(p.returncode)
"""
p=subprocess.run(['kubectl','exec','-i','-n','default','sglang-container-access-46-20261013','-c','hostctl','--','nsenter','-t','1','-m','-n','--','python3','-c',code],input=json.dumps({'store':j['store'],'image':j['image']}),capture_output=True,text=True,timeout=100)
v['image_metadata']={'exit_code':p.returncode,'data':json.loads(p.stdout) if p.stdout else None,'stderr':p.stderr}
print(json.dumps(v)); sys.exit(0 if all(x['exit_code']==0 for x in v['nodes'].values()) and p.returncode==0 else 1)
'''
h=json.loads((R/'notes/L039-model-handles.json').read_text())
targets=[{k:x[k] for k in ['node','container','container_id','root','model_path']} for x in h['model_handles']]
payload={'host':HOST,'image':IMAGE,'targets':targets,'leases':{n:{'name':x['name'],'uid':x['uid']} for n,x in h['resource_leases'].items()},'store':h['store']}
cmd=['ssh','-S',str(R/'connections/l038.sock'),'-o','BatchMode=yes','-o','ConnectTimeout=12','-p','60011','root@61.241.77.34',shlex.join(['python3','-c',JUMP])]
start=datetime.datetime.now(datetime.timezone.utc).isoformat(); p=subprocess.run(cmd,input=json.dumps(payload),capture_output=True,text=True,timeout=230)
(O/'stdout.json').write_text(p.stdout); (O/'stderr.log').write_text(p.stderr)
receipt={'started_utc':start,'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'exit_code':p.returncode,'read_only':True,'NPU_computation':False,'script_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'artifacts':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in O.iterdir() if f.is_file()}}
(O/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n'); print(json.dumps(receipt)); sys.exit(p.returncode)
