#!/usr/bin/env python3
"""Preserve own full model logs, stop only proven-failed parent, inspect installed DeepEP API metadata."""
import datetime,hashlib,json,pathlib,shlex,subprocess,sys
R=pathlib.Path(__file__).resolve().parents[1];O=R/'runs/L039-R043-moe-failure';O.mkdir(parents=True,exist_ok=False)
HOST=r'''
import hashlib,json,pathlib,subprocess,sys
j=json.load(sys.stdin);base=pathlib.Path('/data/wio/sglang-mech36-20261018');assert json.loads((base/'owner.json').read_text())['task']=='sglang-pd-spec-36';name=j['container'];p=subprocess.run(['docker','inspect','--format','{{json .}}',name],capture_output=True,text=True,timeout=25);assert p.returncode==0,p.stderr;x=json.loads(p.stdout);assert x['Id']==j['container_id'] and x['Config']['Labels']['task']=='sglang-pd-spec-36' and x['Config']['Labels']['run']=='L039-R039';root=base/f"L039-R043-diagnostics-{j['node']}";root.mkdir(mode=0o700,exist_ok=False);raw=root/'model-docker.log'
with raw.open('xb') as f:p=subprocess.run(['docker','logs','--timestamps',name],stdout=f,stderr=subprocess.STDOUT,timeout=45)
data=raw.read_bytes();text=data.decode(errors='replace');lines=text.splitlines();shape=('scale batch size[16]' in text and 'groupList length[256]' in text);v={'node':j['node'],'container':name,'container_id':x['Id'],'prior_state':x['State'],'raw_log_path':str(raw),'raw_log_sha256':hashlib.sha256(data).hexdigest(),'raw_log_bytes':len(data),'shape_mismatch_16_vs256_seen':shape,'stage_lines':[s[:1800] for s in lines if any(q in s for q in ['Load weight end','Load weight done','Load weight begin','KV Cache','KV cache','Capture cuda graph failed','scale batch size[16]','groupList length[256]','MoE A2A','moe_a2a_backend=','Traceback'])][:80],'error_contexts':[],'stopped_failed_parent':False}
indexes=[i for i,s in enumerate(lines) if 'Scheduler hit an exception' in s or 'RuntimeError: npu_grouped_matmul' in s or 'AclNN_Parameter_Error' in s]
for i in indexes[:3]:v['error_contexts'].append('\n'.join(lines[max(0,i-8):min(len(lines),i+125)]))
if j['node']==44 and x['State']['Running']:
 assert shape,'Do not stop if the expected fatal failure is not proven in own logs'
 p=subprocess.run(['docker','stop','-t','5',name],capture_output=True,text=True,timeout=20);assert p.returncode==0,p.stderr;v['stopped_failed_parent']=True
p=subprocess.run(['docker','inspect','--format','{{json .State}}',name],capture_output=True,text=True,timeout=25);assert p.returncode==0;v['final_state']=json.loads(p.stdout);assert not v['final_state']['Running']
print(json.dumps(v))
'''
DEPS=r'''
import ast,hashlib,importlib.metadata,importlib.util,json,pathlib
v={'torch_imported':False,'API_semantics':'Static package presence/signatures only, not native functional support'}
spec=importlib.util.find_spec('deep_ep');v['deep_ep_spec']=None if spec is None else {'origin':spec.origin,'package_paths':list(spec.submodule_search_locations or [])};v['distributions']={}
for name in ['deep-ep','deep_ep','torch-npu']:
 try:v['distributions'][name]=importlib.metadata.version(name)
 except importlib.metadata.PackageNotFoundError:v['distributions'][name]=None
v['files']={}
if spec and spec.origin:
 base=pathlib.Path(spec.origin).parent
 for f in sorted(base.glob('*.py')):
  data=f.read_bytes();tree=ast.parse(data);methods=[]
  for node in ast.walk(tree):
   if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and any(k in node.name for k in ['dispatch','combine','fused_deep','buffer_size','capture']):methods.append({'name':node.name,'line':node.lineno,'args':[a.arg for a in node.args.args]})
  v['files'][f.name]={'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data),'selected_methods':methods}
print(json.dumps(v))
'''
JUMP=r'''
import concurrent.futures,json,subprocess,sys
j=json.load(sys.stdin)
def host(item):
 n=item['node'];p=subprocess.run(['kubectl','exec','-i','-n','default',f'sglang-container-access-{n}-20261013','-c','hostctl','--','nsenter','-t','1','-m','-n','--','python3','-c',j['host']],input=json.dumps(item),capture_output=True,text=True,timeout=125);return str(n),{'exit_code':p.returncode,'data':json.loads(p.stdout) if p.stdout else None,'stderr':p.stderr}
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:results=dict(pool.map(host,j['targets']))
code="import json,subprocess,sys;j=json.load(sys.stdin);p=subprocess.run(['docker','inspect','--format','{{json .}}',j['store']['name']],capture_output=True,text=True,timeout=20);assert p.returncode==0;x=json.loads(p.stdout);assert x['Id']==j['store']['container_id'] and x['State']['Running'] and x['Config']['Labels']['run']=='L039-R033';p=subprocess.run(['docker','exec',j['store']['name'],'python3','-c',j['deps']],capture_output=True,text=True,timeout=30);sys.stdout.write(p.stdout);sys.stderr.write(p.stderr);sys.exit(p.returncode)"
p=subprocess.run(['kubectl','exec','-i','-n','default','sglang-container-access-46-20261013','-c','hostctl','--','nsenter','-t','1','-m','-n','--','python3','-c',code],input=json.dumps({'store':j['store'],'deps':j['deps']}),capture_output=True,text=True,timeout=65);deps={'exit_code':p.returncode,'data':json.loads(p.stdout) if p.stdout else None,'stderr':p.stderr}
print(json.dumps({'nodes':results,'DeepEP_package':deps,'CPU_store_preserved':True,'model_files_preserved':True,'inference_workload_executed':False}));sys.exit(0 if all(x['exit_code']==0 for x in results.values()) and deps['exit_code']==0 else 1)
'''
a=json.loads((R/'runs/L039-R039-full-bootstrap/remote-result.json').read_text());s=json.loads((R/'runs/L039-R033-shared-store/remote-result.json').read_text())['store']['data'];targets=[{'node':int(n),'container':x['data']['name'],'container_id':x['data']['container_id']} for n,x in a['nodes'].items()];cmd=['ssh','-S',str(R/'connections/l038.sock'),'-o','BatchMode=yes','-o','ConnectTimeout=12','-p','60011','root@61.241.77.34',shlex.join(['python3','-c',JUMP])];start=datetime.datetime.now(datetime.timezone.utc).isoformat();p=subprocess.run(cmd,input=json.dumps({'host':HOST,'deps':DEPS,'targets':targets,'store':{k:s[k] for k in ['name','container_id']}}),capture_output=True,text=True,timeout=225)
(O/'stdout.json').write_text(p.stdout);(O/'stderr.log').write_text(p.stderr);receipt={'started_utc':start,'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'exit_code':p.returncode,'inference_workload_executed':False,'script_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'artifacts':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in O.iterdir() if f.is_file()}}
(O/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt));sys.exit(p.returncode)
