#!/usr/bin/env python3
"""Read exact installed stream/allocator public headers and Python lifecycle sources."""
import ast,datetime,hashlib,json,pathlib,shlex,subprocess,sys
R=pathlib.Path(__file__).resolve().parents[1];O=R/'runs/L039-R063-allocator-sources';O.mkdir(parents=True,exist_ok=False)
tree=ast.parse((R/'scripts/collect_L039_tiling_details.py').read_text());JUMP=next(n.value.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='JUMP' for t in n.targets))
IMAGE=r'''
import hashlib,importlib.util,json,os,pathlib,re,sys
assert 'torch' not in sys.modules
pkg=pathlib.Path(importlib.util.find_spec('torch_npu').origin).parent
v={'Torch_imported':False,'NPU_device_nodes_absent':not any(pathlib.Path('/dev').glob('davinci*')),'sources':{},'header_matches':[],'DeepEP_Cpp_files':[]}
def capture(p):
 if not p.is_file():return
 data=p.read_bytes();assert len(data)<512*1024
 v['sources'][str(p)]={'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data),'source':data.decode(errors='replace')}
for n in ['_init/core/runtime_lifecycle.py','npu/memory.py','npu/streams.py']:capture(pkg/n)
pattern=re.compile(r'aclrtAllocator(?:GetByStream|Register|Unregister|CreateDesc|DestroyDesc)|register[^;\n]*[Aa]llocator|[Aa]llocator[^;\n]*[Ss]tream|[Ss]tream[^;\n]*[Aa]llocator')
roots=[pkg/'include',pathlib.Path(os.environ['ASCEND_HOME_PATH'])/'include']
seen=set();scanned=0
for root in roots:
 if not root.is_dir():continue
 for p in sorted(root.rglob('*')):
  if p.suffix not in ['.h','.hpp'] or not p.is_file() or p.stat().st_size>512*1024:continue
  real=str(p.resolve())
  if real in seen:continue
  seen.add(real);data=p.read_bytes();scanned+=len(data);text=data.decode(errors='replace');lines=text.splitlines();matches=[]
  for i,line in enumerate(lines):
   if pattern.search(line):matches.append({'line':i+1,'context':'\n'.join(lines[max(0,i-8):min(len(lines),i+16)])[:12000]})
  if matches and len(v['header_matches'])<40:
   v['header_matches'].append({'path':str(p),'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data),'matches':matches[:18]})
  if 'Allocator' in p.name and 'Caching' in p.name and matches and len(v['sources'])<12:capture(p)
v['header_bytes_scanned']=scanned
deep=pathlib.Path(importlib.util.find_spec('deep_ep').origin).parent
for p in deep.rglob('*'):
 if p.is_file() and p.suffix in ['.cpp','.cc','.h','.hpp']:v['DeepEP_Cpp_files'].append(str(p))
v['Torch_imported']='torch' in sys.modules;assert not v['Torch_imported'] and v['NPU_device_nodes_absent'];print(json.dumps(v))
'''
h=json.loads((R/'notes/L039-model-handles.json').read_text());payload={'host':'','image':IMAGE,'targets':[],'leases':{n:{'name':x['name'],'uid':x['uid']} for n,x in h['resource_leases'].items()},'store':h['store']}
cmd=['ssh','-S',str(R/'connections/l038.sock'),'-o','BatchMode=yes','-o','ConnectTimeout=12','-p','60011','root@61.241.77.34',shlex.join(['python3','-c',JUMP])];start=datetime.datetime.now(datetime.timezone.utc).isoformat();p=subprocess.run(cmd,input=json.dumps(payload),capture_output=True,text=True,timeout=210)
(O/'stdout.json').write_text(p.stdout);(O/'stderr.log').write_text(p.stderr);receipt={'started_utc':start,'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'exit_code':p.returncode,'read_only':True,'NPU_computation':False,'script_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'artifacts':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in O.iterdir() if f.is_file()}}
(O/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt));sys.exit(p.returncode)
