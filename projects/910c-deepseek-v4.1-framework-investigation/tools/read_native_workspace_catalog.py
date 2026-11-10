#!/usr/bin/env python3
"""Read-only installed backend/version/selected policy and old memory table inventory."""
from hashlib import sha256
import json
from pathlib import Path
import subprocess
from read_dfc_gemm_native_catalog import DB

REMOTE=r'''
from pathlib import Path
from hashlib import sha256
import json,subprocess,sys,sqlite3
name='wio-vllm-v41-repro171-20261102';c,=json.loads(subprocess.check_output(['docker','inspect',name]));assert c['State']['Running'] and c['Image']=='sha256:ce42d65b2403992ad2e2b06c61f4f048babb57e455cfe307b018345e36c5e98b'
inner=r"""
from pathlib import Path
from hashlib import sha256
import json,ast
site=Path('/usr/local/python3.12.13/lib/python3.12/site-packages');versions=[]
for name in ['torch','torch_npu']:
 for fn in ['version.py','_version.py']:
  p=site/name/fn
  if not p.is_file():continue
  vals={}
  for node in ast.parse(p.read_text()).body:
   if isinstance(node,(ast.Assign,ast.AnnAssign)):
    target=node.targets[0] if isinstance(node,ast.Assign) else node.target
    if isinstance(target,ast.Name) and target.id in ['__version__','git_version','cann_version','debug']:
     try:vals[target.id]=ast.literal_eval(node.value)
     except (ValueError,TypeError):pass
  versions.append(dict(path=str(p),sha256=sha256(p.read_bytes()).hexdigest(),constants=vals))
headers=[];root=site/'torch_npu/include/torch_npu/csrc'
for sub in ['core/npu','framework']:
 for p in sorted((root/sub).glob('*.h')):
  if any(x in p.name.lower() for x in ['allocator','command','queue','stream','graph','option','memory','async','param']):headers.append(dict(path=str(p),bytes=p.stat().st_size))
source_inventory=[]
for root in [Path('/vllm-workspace/torch_npu'),Path('/vllm-workspace/pytorch/torch_npu'),Path('/opt/torch_npu'),site/'torch_npu']:
 if not root.exists():continue
 for n in ['OpCommand.cpp','NPUCachingAllocator.cpp','NPUStream.cpp','NPUGraph.cpp','OptionsManager.cpp','TaskQueue.cpp','OpCommandBase.cpp']:
  source_inventory += [dict(path=str(p),bytes=p.stat().st_size) for p in root.rglob(n) if p.is_file()]
keys={'TASK_QUEUE_ENABLE','COMBINED_ENABLE','ASCEND_LAUNCH_BLOCKING','PYTORCH_NPU_ALLOC_CONF','PYTORCH_ALLOC_CONF','PYTORCH_NO_NPU_MEMORY_CACHING','MULTI_STREAM_MEMORY_REUSE'}
workers=[]
for d in Path('/proc').iterdir():
 if not d.name.isdigit():continue
 try:
  maps=(d/'maps').read_text();libs=sorted(set(line.split()[-1] for line in maps.splitlines() if '/torch_npu/lib/libtorch_npu.so' in line))
  if not libs:continue
  env={}
  for pair in (d/'environ').read_bytes().split(b'\0'):
   k,_,v=pair.partition(b'=')
   if k.decode(errors='replace') in keys:env[k.decode()]=v.decode(errors='replace')
  workers.append(dict(pid=int(d.name),mapped_selected_libraries=libs,initial_selected_environment=env))
 except (OSError,ProcessLookupError,PermissionError):pass
libs=[]
for path in sorted(set(p for w in workers for p in w['mapped_selected_libraries'])):
 p=Path(path);libs.append(dict(path=path,bytes=p.stat().st_size,sha256=sha256(p.read_bytes()).hexdigest()))
print(json.dumps(dict(versions=versions,header_inventory=headers,available_source_inventory=source_inventory,workers=workers,libraries=libs)))
"""
d=json.loads(subprocess.check_output(['docker','exec',name,'python3','-B','-I','-S','-c',inner]))
p=Path(sys.argv[1]);assert p.is_file() and p.is_relative_to('/data/tiankuan/wio');dbhash=sha256(p.read_bytes()).hexdigest();assert dbhash==sys.argv[2]
with sqlite3.connect(p.as_uri()+'?mode=ro&immutable=1',uri=True) as con:
 tables=[r[0] for r in con.execute('select name from sqlite_master where type="table"')];mem=[]
 for t in tables:
  if any(s in t.upper() for s in ['MEM','ALLOC']):
   q='"'+t.replace('"','""')+'"';cols=[list(r) for r in con.execute('pragma table_info('+q+')')];num,=con.execute('select count(*) from '+q).fetchone();mem.append(dict(table=t,rows=num,schema=cols))
d.update(database=dict(path=str(p),sha256=dbhash,memory_tables=mem,table_names=tables),container_id=c['Id'],image=c['Image'],new_NPU_work=0,model_loads=0,service_requests=0,new_profile_collection=0,remote_writes=[])
print(json.dumps(d))
'''


def main():
 pin=next(x for x in json.loads(Path('audit/execution-search/dfc-archived-launch-fields.json').read_text())['profiles'] if x['rank']==0)
 assert pin['database']==DB
 r=subprocess.run(['/Users/wio/.local/bin/baai-ssh','171','--','python3','-B','-I','-S','-c',REMOTE,DB,pin['database_sha256']],capture_output=True,text=True,timeout=150)
 if r.returncode:raise RuntimeError(r.stderr)
 d=json.loads(r.stdout);d['reader_sha256']=sha256(Path(__file__).read_bytes()).hexdigest();Path('audit/execution-search/native-workspace-catalog.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps(dict(versions=d['versions'],workers=len(d['workers']),available_sources=len(d['available_source_inventory']),headers=len(d['header_inventory']),memory_tables=[dict(table=x['table'],rows=x['rows']) for x in d['database']['memory_tables']]),ensure_ascii=False))

if __name__=='__main__':main()
