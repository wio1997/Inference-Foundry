#!/usr/bin/env python3
"""Read only own failed-run CANN logs plus exact-image DeepEP/OPP metadata. No NPU computation."""
import datetime,hashlib,json,pathlib,shlex,subprocess,sys
R=pathlib.Path(__file__).resolve().parents[1];O=R/'runs/L039-R048-cann-tiling';O.mkdir(parents=True,exist_ok=False)
HOST=r'''
import collections,hashlib,json,pathlib,re,subprocess,sys
j=json.load(sys.stdin);base=pathlib.Path('/data/wio/sglang-mech36-20261018');assert json.loads((base/'owner.json').read_text())['task']=='sglang-pd-spec-36';root=pathlib.Path(j['root']);assert root.parent==base and root.resolve().parent==base.resolve();p=subprocess.run(['docker','inspect','--format','{{json .}}',j['container']],capture_output=True,text=True,timeout=25);assert p.returncode==0,p.stderr;x=json.loads(p.stdout);assert x['Id']==j['container_id'] and x['Config']['Labels']['task']=='sglang-pd-spec-36' and x['Config']['Labels']['run']=='L039-R044' and not x['State']['Running'];logs=root/'ascend-logs';v={'node':j['node'],'container_id':x['Id'],'container_state':x['State'],'logs_root':str(logs),'files':[],'unique_error_lines':[],'operator_contexts':[],'source_plan':json.loads((root/'plan.json').read_text()),'mutated_services':False}
seen=set();context_seen=set();total=0
for f in sorted(logs.rglob('*')):
 if not f.is_file() or f.is_symlink():continue
 assert f.resolve().is_relative_to(logs.resolve());before=f.stat();data=f.read_bytes();after=f.stat();assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns),'Own stopped-run log changed';total+=len(data);v['files'].append({'path':str(f.relative_to(logs)),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()});lines=data.decode(errors='replace').splitlines()
 for i,line in enumerate(lines):
  relevant=any(s in line.lower() for s in ['moedispatch','moe_dispatch','moelowlatency','moe_low_latency','camm','tiling','error','ez1008','checkparam','paramcheck','hcomm','commcontext','unsupported','not support'])
  if not relevant:continue
  normal=re.sub(r'\b\d{2}:\d{2}:\d{2}(?:[.:]\d+)*\b','TIME',line);normal=re.sub(r'\[\d+\]','[ID]',normal)
  if normal not in seen and len(v['unique_error_lines'])<240:seen.add(normal);v['unique_error_lines'].append({'file':str(f.relative_to(logs)),'line':i+1,'text':line[:2400]})
  if any(s in line.lower() for s in ['failed','invalid','should','not support','error','mismatch']) and any(s in '\n'.join(lines[max(0,i-8):i+5]).lower() for s in ['moedispatch','moe_dispatch','moelowlatency','moe_low_latency','camm','tiling']):
   text='\n'.join(lines[max(0,i-8):min(len(lines),i+24)]);signature=re.sub(r'\b\d{2}:\d{2}:\d{2}(?:[.:]\d+)*\b','TIME',text)
   if signature not in context_seen and len(v['operator_contexts'])<40:context_seen.add(signature);v['operator_contexts'].append({'file':str(f.relative_to(logs)),'line':i+1,'context':text[:16000]})
v['total_log_bytes_scanned']=total;v['file_count']=len(v['files']);v['logs_root_exists']=logs.is_dir();print(json.dumps(v))
'''
IMAGE=r'''
import ast,hashlib,importlib.metadata,importlib.util,json,os,pathlib,re,subprocess
keys=['ASCEND_HOME_PATH','ASCEND_OPP_PATH','ASCEND_CUSTOM_OPP_PATH','ASCEND_AICPU_PATH','ASCEND_TOOLKIT_HOME','ASCEND_TOOLKIT_HOME_DIR','LD_LIBRARY_PATH','LD_PRELOAD','PYTHONPATH']
v={'selected_environment':{k:os.environ.get(k) for k in keys},'no_NPU_device_nodes':not any(pathlib.Path('/dev').glob('davinci*')),'versions':{},'version_files':{},'DeepEP_sources':{},'operator_metadata':[],'library_inventory':[]}
for name in ['deep-ep','torch','torch-npu','sgl-kernel-npu']:
 try:v['versions'][name]=importlib.metadata.version(name)
 except importlib.metadata.PackageNotFoundError:v['versions'][name]=None
spec=importlib.util.find_spec('deep_ep');assert spec and spec.origin;pkg=pathlib.Path(spec.origin).parent;v['DeepEP_package_root']=str(pkg)
for filename in ['__init__.py','build_config.py','device_info.py','ep_strategy.py','buffer.py']:
 p=pkg/filename
 if p.is_file():
  data=p.read_bytes();v['DeepEP_sources'][filename]={'path':str(p),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'source':data.decode()}
for base in [pathlib.Path('/usr/local/Ascend/ascend-toolkit/latest'),pathlib.Path('/usr/local/Ascend/ascend-toolkit'),pathlib.Path('/usr/local/Ascend/driver')]:
 for name in ['version.cfg','version.info','ascend_toolkit_install.info']:
  p=base/name
  if p.is_file():v['version_files'][str(p)]={'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'text':p.read_text(errors='replace')[:20000]}
 # A version file may sit one architecture directory below latest.
 if base.name=='latest':
  for p in base.glob('*/ascend_toolkit_install.info'):
   if p.is_file():v['version_files'][str(p)]={'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'text':p.read_text(errors='replace')[:20000]}
roots={str(pkg)}
for k in ['ASCEND_OPP_PATH','ASCEND_CUSTOM_OPP_PATH']:
 for item in (os.environ.get(k) or '').split(':'):
  if item:roots.add(item)
roots.add('/usr/local/Ascend/ascend-toolkit/latest/opp');visited=set()
for rootname in sorted(roots):
 root=pathlib.Path(rootname)
 if not root.is_dir():continue
 for current,dirs,files in os.walk(root,followlinks=False):
  dirs[:]=[n for n in dirs if n not in ['__pycache__','.git']]
  for name in files:
   p=pathlib.Path(current)/name
   if p.is_symlink() and not p.exists():continue
   try:real=str(p.resolve());size=p.stat().st_size
   except OSError:continue
   if real in visited:continue
   visited.add(real);lower=name.lower()
   if '.so' in lower and (root==pkg or any(s in lower for s in ['tiling','custom','moe','opapi','hccl','hcomm'])):
    if len(v['library_inventory'])<160:v['library_inventory'].append({'path':str(p),'realpath':real,'bytes':size})
   textual=p.suffix.lower() in ['.json','.ini','.cfg','.info','.yaml','.yml','.py','.sh']
   if not textual or size>4*1024**2 or not any(s in lower for s in ['moe','ops-info','set_env','install.info']):continue
   try:data=p.read_bytes();text=data.decode()
   except (OSError,UnicodeDecodeError):continue
   matches=[];lines=text.splitlines()
   for i,line in enumerate(lines):
    if any(s in line.lower() for s in ['cammoedispatchnormal','cam_moe_dispatch_normal','moelowlatencydispatchv2','moe_low_latency_dispatch_v2']):matches.append({'line':i+1,'context':'\n'.join(lines[max(0,i-4):min(len(lines),i+18)])[:10000]})
   if matches and len(v['operator_metadata'])<80:v['operator_metadata'].append({'path':str(p),'realpath':real,'bytes':size,'sha256':hashlib.sha256(data).hexdigest(),'matches':matches[:8]})
print(json.dumps(v))
'''
JUMP=r'''
import concurrent.futures,json,subprocess,sys
j=json.load(sys.stdin);v={'nodes':{},'resource_leases':{},'image_metadata':None,'read_only':True,'NPU_computation':False}
for n,lease in j['leases'].items():
 p=subprocess.run(['kubectl','get','pod','-n','default',lease['name'],'-o','json'],capture_output=True,text=True,timeout=25)
 if p.returncode==0:
  x=json.loads(p.stdout);v['resource_leases'][n]={'name':x['metadata']['name'],'uid':x['metadata']['uid'],'expected_uid':lease['uid'],'phase':x['status']['phase'],'reason':x['status'].get('reason'),'message':x['status'].get('message'),'ready':next((c['status'] for c in x['status'].get('conditions',[]) if c['type']=='Ready'),None)}
 else:v['resource_leases'][n]={'exit_code':p.returncode,'stderr':p.stderr}
def host(item):
 n=item['node'];p=subprocess.run(['kubectl','exec','-i','-n','default',f'sglang-container-access-{n}-20261013','-c','hostctl','--','nsenter','-t','1','-m','-n','--','python3','-c',j['host']],input=json.dumps(item),capture_output=True,text=True,timeout=100);return str(n),{'exit_code':p.returncode,'data':json.loads(p.stdout) if p.stdout else None,'stderr':p.stderr}
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:v['nodes']=dict(pool.map(host,j['targets']))
code=r"""
import json,shlex,subprocess,sys
j=json.load(sys.stdin);p=subprocess.run(['docker','inspect','--format','{{json .}}',j['store']['container']],capture_output=True,text=True,timeout=25);assert p.returncode==0,p.stderr;x=json.loads(p.stdout);assert x['Id']==j['store']['container_id'] and x['State']['Running'] and x['Config']['Labels']['task']=='sglang-pd-spec-36' and x['Config']['Labels']['run']=='L039-R033';assert not x['HostConfig']['Devices']
setup='set -e; if [ -r /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then . /usr/local/Ascend/ascend-toolkit/set_env.sh; elif [ -r /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash ]; then . /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash; fi; exec python3 -c '+shlex.quote(j['image'])
p=subprocess.run(['docker','exec',j['store']['container'],'/bin/bash','-c',setup],capture_output=True,text=True,timeout=140);sys.stdout.write(p.stdout);sys.stderr.write(p.stderr);sys.exit(p.returncode)
"""
p=subprocess.run(['kubectl','exec','-i','-n','default','sglang-container-access-46-20261013','-c','hostctl','--','nsenter','-t','1','-m','-n','--','python3','-c',code],input=json.dumps({'store':j['store'],'image':j['image']}),capture_output=True,text=True,timeout=190);v['image_metadata']={'exit_code':p.returncode,'data':json.loads(p.stdout) if p.stdout else None,'stderr':p.stderr}
print(json.dumps(v));sys.exit(0 if all(x['exit_code']==0 for x in v['nodes'].values()) and p.returncode==0 else 1)
'''
h=json.loads((R/'notes/L039-model-handles.json').read_text());targets=[{k:x[k] for k in ['node','container','container_id','root']} for x in h['model_handles']];payload={'host':HOST,'image':IMAGE,'targets':targets,'leases':h['resource_leases'],'store':h['store']}
# Normalize snapshot fields to the fixed lease identity needed for read-only observation.
payload['leases']={n:{'name':x['name'],'uid':x['uid']} for n,x in h['resource_leases'].items()}
cmd=['ssh','-S',str(R/'connections/l038.sock'),'-o','BatchMode=yes','-o','ConnectTimeout=12','-p','60011','root@61.241.77.34',shlex.join(['python3','-c',JUMP])];start=datetime.datetime.now(datetime.timezone.utc).isoformat();p=subprocess.run(cmd,input=json.dumps(payload),capture_output=True,text=True,timeout=330)
(O/'stdout.json').write_text(p.stdout);(O/'stderr.log').write_text(p.stderr);receipt={'started_utc':start,'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'exit_code':p.returncode,'read_only':True,'NPU_computation':False,'script_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'artifacts':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in O.iterdir() if f.is_file()}}
(O/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt));sys.exit(p.returncode)
