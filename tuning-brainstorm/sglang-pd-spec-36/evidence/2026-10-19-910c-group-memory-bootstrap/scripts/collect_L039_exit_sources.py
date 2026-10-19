#!/usr/bin/env python3
"""Read installed Torch/NPU shutdown and torchrun source via own CPU-only store."""
import ast,datetime,hashlib,json,pathlib,shlex,subprocess,sys
R=pathlib.Path(__file__).resolve().parents[1];O=R/'runs/L039-R054-exit-sources';O.mkdir(parents=True,exist_ok=False)
tree=ast.parse((R/'scripts/collect_L039_tiling_details.py').read_text())
JUMP=next(n.value.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='JUMP' for t in n.targets))
IMAGE=r'''
import hashlib,importlib.metadata,importlib.util,json,os,pathlib
v={'NPU_device_nodes_absent':not any(pathlib.Path('/dev').glob('davinci*')),'Torch_imported':False,'versions':{},'sources':{},'torch_backend_entry_points':[],'selected_environment':{k:os.environ.get(k) for k in ['TORCH_DEVICE_BACKEND_AUTOLOAD','TORCH_NPU_AUTORUN','ASCEND_OPP_PATH']}}
for pkg in ['torch','torch_npu']:
 spec=importlib.util.find_spec(pkg);root=pathlib.Path(spec.origin).parent
 names=['__init__.py','npu/__init__.py'] if pkg=='torch_npu' else ['distributed/run.py','distributed/elastic/multiprocessing/api.py','distributed/elastic/multiprocessing/subprocess_handler/subprocess_handler.py']
 for name in names:
  p=root/name;data=p.read_bytes();assert len(data)<256*1024
  v['sources'][pkg+'/'+name]={'path':str(p),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'source':data.decode()}
for name in ['torch','torch-npu','deep-ep']:v['versions'][name]=importlib.metadata.version(name)
for ep in importlib.metadata.entry_points(group='torch.backends'):v['torch_backend_entry_points'].append({'name':ep.name,'value':ep.value,'group':ep.group})
v['Torch_imported']='torch' in __import__('sys').modules;assert not v['Torch_imported'] and v['NPU_device_nodes_absent']
print(json.dumps(v))
'''
h=json.loads((R/'notes/L039-model-handles.json').read_text());payload={'host':'','image':IMAGE,'targets':[],'leases':{n:{'name':x['name'],'uid':x['uid']} for n,x in h['resource_leases'].items()},'store':h['store']}
cmd=['ssh','-S',str(R/'connections/l038.sock'),'-o','BatchMode=yes','-o','ConnectTimeout=12','-p','60011','root@61.241.77.34',shlex.join(['python3','-c',JUMP])]
start=datetime.datetime.now(datetime.timezone.utc).isoformat();p=subprocess.run(cmd,input=json.dumps(payload),capture_output=True,text=True,timeout=200)
(O/'stdout.json').write_text(p.stdout);(O/'stderr.log').write_text(p.stderr)
receipt={'started_utc':start,'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'exit_code':p.returncode,'read_only':True,'NPU_computation':False,'script_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'artifacts':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in O.iterdir() if f.is_file()}}
(O/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt));sys.exit(p.returncode)
