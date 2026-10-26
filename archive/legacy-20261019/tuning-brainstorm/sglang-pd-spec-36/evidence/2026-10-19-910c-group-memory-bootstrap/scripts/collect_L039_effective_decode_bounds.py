#!/usr/bin/env python3
"""Archive own terminalDecode log and exact capture/dispatch/KV-limit source; keep P running."""
import ast,datetime,hashlib,json,pathlib,shlex,subprocess,sys
R=pathlib.Path(__file__).resolve().parents[1];O=R/'runs/L039-R070-effective-bounds';O.mkdir(parents=True,exist_ok=False)
tree=ast.parse((R/'scripts/collect_L039_tiling_details.py').read_text());JUMP=next(n.value.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='JUMP' for t in n.targets))
HOST=r'''
import hashlib,json,pathlib,subprocess,sys,urllib.request
j=json.load(sys.stdin);base=pathlib.Path('/data/wio/sglang-mech36-20261018');assert json.loads((base/'owner.json').read_text())['task']=='sglang-pd-spec-36';p=subprocess.run(['docker','inspect','--format','{{json .}}',j['container']],capture_output=True,text=True,timeout=25);assert p.returncode==0,p.stderr;x=json.loads(p.stdout);assert x['Id']==j['container_id'] and x['Config']['Labels']['task']=='sglang-pd-spec-36' and x['Config']['Labels']['run']=='L039-R067'
v={'node':j['node'],'container_id':x['Id'],'state':x['State'],'selected_env':{key:next((e.split('=',1)[1] for e in x['Config']['Env'] if e.startswith(key+'=')),None) for key in ['SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK','HCCL_BUFFSIZE','DEEPEP_HCCL_BUFFSIZE']},'stopped_services':False,'read_only':True}
if j['node']==44:
 assert x['State']['Running'];opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
 with opener.open('http://172.16.12.44:39044/ready',timeout=4) as resp:v['pure_ready_status']=resp.status
else:
 assert not x['State']['Running'];p=pathlib.Path(j['raw_log_path']);assert p.resolve().is_relative_to(base.resolve());data=p.read_bytes();assert hashlib.sha256(data).hexdigest()==j['raw_log_sha256'];v['dispatch_env_log_lines']=[line[:3000] for line in data.decode(errors='replace').splitlines() if 'SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK' in line]
print(json.dumps(v))
'''
IMAGE=r'''
import hashlib,json,pathlib,sys
root=pathlib.Path('/sgl-workspace/sglang');key='SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK';v={'matches':[],'sources':{},'Torch_imported':'torch' in sys.modules,'NPU_device_nodes_absent':not any(pathlib.Path('/dev').glob('davinci*'))}
for p in sorted((root/'python/sglang').rglob('*.py')):
 data=p.read_bytes()
 if len(data)>1024*1024:continue
 text=data.decode(errors='replace');lines=text.splitlines();matches=[]
 for i,line in enumerate(lines):
  if key in line:matches.append({'line':i+1,'context':'\n'.join(lines[max(0,i-8):min(len(lines),i+12)])})
 if matches:v['matches'].append({'path':str(p),'sha256':hashlib.sha256(data).hexdigest(),'matches':matches})
for name in ['python/sglang/srt/speculative/eagle_draft_extend_cuda_graph_runner.py','python/sglang/srt/speculative/eagle_draft_cuda_graph_runner.py']:
 p=root/name;data=p.read_bytes();v['sources'][name]={'path':str(p),'sha256':hashlib.sha256(data).hexdigest(),'source':data.decode()}
assert not v['Torch_imported'] and v['NPU_device_nodes_absent'];print(json.dumps(v))
'''
h=json.loads((R/'specs/L039-R070-targets.json').read_text());payload={'host':HOST,'image':IMAGE,'targets':h['targets'],'leases':h['resource_leases'],'store':h['store']}
cmd=['ssh','-S',str(R/'connections/l038.sock'),'-o','BatchMode=yes','-o','ConnectTimeout=12','-p','60011','root@61.241.77.34',shlex.join(['python3','-c',JUMP])];start=datetime.datetime.now(datetime.timezone.utc).isoformat();p=subprocess.run(cmd,input=json.dumps(payload),capture_output=True,text=True,timeout=220)
(O/'stdout.json').write_text(p.stdout);(O/'stderr.log').write_text(p.stderr);receipt={'started_utc':start,'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'exit_code':p.returncode,'read_only_except_own_log_archive':True,'NPU_computation':False,'services_stopped':False,'script_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'artifacts':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in O.iterdir() if f.is_file()}}
(O/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt));sys.exit(p.returncode)
