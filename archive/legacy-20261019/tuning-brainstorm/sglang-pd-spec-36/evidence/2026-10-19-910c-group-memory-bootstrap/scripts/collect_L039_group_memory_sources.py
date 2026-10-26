#!/usr/bin/env python3
"""Read exact installed stream/allocator public headers and Python lifecycle sources."""
import ast,datetime,hashlib,json,pathlib,shlex,subprocess,sys
R=pathlib.Path(__file__).resolve().parents[1];O=R/'runs/L039-R064-group-memory-sources';O.mkdir(parents=True,exist_ok=False)
tree=ast.parse((R/'scripts/collect_L039_tiling_details.py').read_text());JUMP=next(n.value.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='JUMP' for t in n.targets))
IMAGE=r'''
import hashlib,importlib.util,json,pathlib,re,sys
assert 'torch' not in sys.modules
v={'Torch_imported':False,'NPU_device_nodes_absent':not any(pathlib.Path('/dev').glob('davinci*')),'sources':{},'binary_candidates':[],'module_specs':{}}
def capture(p):
 if not p.is_file():return
 data=p.read_bytes();assert len(data)<1024*1024
 v['sources'][str(p)]={'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'source':data.decode(errors='replace')}
root=pathlib.Path('/sgl-workspace/sglang');pattern=re.compile(r'HCCL_BUFFSIZE|hccl_buffer|hccl_config|ProcessGroupHCCL')
for p in sorted((root/'python/sglang/srt').rglob('*.py')):
 if p.stat().st_size>1024*1024:continue
 if '/distributed/' in str(p) or pattern.search(p.read_text(errors='replace')):
  if len(v['sources'])<50:capture(p)
for name in ['deep_ep','sgl_kernel_npu','deep_ep_cpp']:
 spec=importlib.util.find_spec(name);v['module_specs'][name]={'origin':spec.origin,'locations':list(spec.submodule_search_locations or [])} if spec else None
 if spec and spec.origin and pathlib.Path(spec.origin).suffix=='.py':capture(pathlib.Path(spec.origin))
pkg=pathlib.Path(importlib.util.find_spec('deep_ep').origin).parent
candidates=list(pkg.glob('deep_ep_cpp*.so'))+list(pkg.parent.glob('deep_ep_cpp*.so'))
ks=importlib.util.find_spec('sgl_kernel_npu')
if ks and ks.origin:candidates+=list(pathlib.Path(ks.origin).parent.rglob('deep_ep_cpp*.so'))
for p in candidates:
 data=p.read_bytes();s=p.stat();v['binary_candidates'].append({'path':str(p),'realpath':str(p.resolve()),'inode':s.st_ino,'device':s.st_dev,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'symlink':p.is_symlink()})
torchpkg=pathlib.Path(importlib.util.find_spec('torch_npu').origin).parent
for p in (torchpkg/'include').rglob('*ProcessGroupHCCL*'):
 if p.is_file() and p.suffix in ['.h','.hpp']:capture(p)
v['Torch_imported']='torch' in sys.modules;assert not v['Torch_imported'] and v['NPU_device_nodes_absent'];print(json.dumps(v))
'''
h=json.loads((R/'notes/L039-model-handles.json').read_text());payload={'host':'','image':IMAGE,'targets':[],'leases':{n:{'name':x['name'],'uid':x['uid']} for n,x in h['resource_leases'].items()},'store':h['store']}
cmd=['ssh','-S',str(R/'connections/l038.sock'),'-o','BatchMode=yes','-o','ConnectTimeout=12','-p','60011','root@61.241.77.34',shlex.join(['python3','-c',JUMP])];start=datetime.datetime.now(datetime.timezone.utc).isoformat();p=subprocess.run(cmd,input=json.dumps(payload),capture_output=True,text=True,timeout=210)
(O/'stdout.json').write_text(p.stdout);(O/'stderr.log').write_text(p.stderr);receipt={'started_utc':start,'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'exit_code':p.returncode,'read_only':True,'NPU_computation':False,'script_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'artifacts':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in O.iterdir() if f.is_file()}}
(O/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt));sys.exit(p.returncode)
