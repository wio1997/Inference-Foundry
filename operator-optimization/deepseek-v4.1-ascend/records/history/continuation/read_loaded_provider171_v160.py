"""Read own live worker mappings and fixed SDK declarations; never call CANN."""
import sys,json,base64
from pathlib import Path
from finalize_origin_coverage_v9 import N,O,write
sys.path.insert(0,'/Users/wio/work/171-vllm-migration-20261102')
from control import invoke
code=r'''import os,json,re,subprocess,socket,hashlib,base64,stat
from pathlib import Path
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-171'
cid='3be44e7b5c292de6ef7fc51085bfbbad5552acc97fae39c0ae176fe8cfb387eb'
d=json.loads(subprocess.check_output(['docker','inspect',cid],text=True))[0];assert d['State']['Running']
pids=[int(s.strip()) for s in subprocess.check_output(['docker','top',cid,'-eo','pid'],text=True).splitlines() if s.strip().isdigit()]
rows=[];cache={}
for pid in pids:
 p=Path('/proc')/str(pid);found={}
 for line in (p/'maps').read_text().splitlines():
  parts=line.split(None,5)
  if len(parts)<6:continue
  name=parts[5]
  if not name.startswith('/') or not any(Path(name).name.startswith(n) for n in ['libascendcl.so','libruntime.so']):continue
  key=(name,parts[3],int(parts[4]));found[key]=found.get(key,0)+1
 for (name,dev,inode),segments in found.items():
  f=p/'root'/name.lstrip('/');s=f.stat();major,minor=(int(x,16) for x in dev.split(':'));matches=s.st_ino==inode and os.major(s.st_dev)==major and os.minor(s.st_dev)==minor
  identity=(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)
  if identity not in cache:
   b=f.read_bytes();after=f.stat();assert identity==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns)
   cache[identity]={'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'ELF_magic':b[:4].hex(),'ELF_class':b[4],'ELF_data':b[5],'ELF_machine':int.from_bytes(b[18:20],'little')}
  rows.append(dict(pid=pid,path=name,mapped_device=dev,mapped_inode=inode,path_stat_matches_mapping=matches,segments=segments,**cache[identity]))
api=d['State']['Pid'];header_path='/usr/local/Ascend/cann-9.2.0/aarch64-linux/include/acl/acl_rt.h';f=Path('/proc')/str(api)/'root'/header_path.lstrip('/');b=f.read_bytes();text=b.decode()
names=['aclrtLaunchKernelWithHostArgs','aclrtGetFunctionName','aclrtGetThreadLastTaskId','aclrtStreamGetId']
decls={n:re.findall(r'\baclError\s+'+n+r'\s*\(.*?\);',text,re.S) for n in names}
result={'target':'172.16.10.171','container_id':cid,'container_image':d['Image'],'examined_owned_processes':len(pids),'loaded_provider_mapping_records':rows,'header':{'path_in_container':header_path,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'base64':base64.b64encode(b).decode()},'declarations':decls,'no_process_memory_or_CANN_runtime_calls':True,'actual_interposition_or_bypass_coverage_tested':False,'historical_loaded_libraries_attested':False,'overall_complete':False}
out=Path('/data/tiankuan/wio/migration-171-20261102/operator-evidence-v160');out.mkdir(mode=0o700,exist_ok=True);p=out/'loaded-provider-mappings.json';assert not p.exists();p.write_text(json.dumps(result,indent=2));print(json.dumps(result))
'''
d=invoke('target',code,60)
cache=N/'provider-abi-cache-v160';cache.mkdir(exist_ok=True)
header=base64.b64decode(d['header'].pop('base64'));p=cache/'acl_rt.h';assert not p.exists();p.write_bytes(header);d['header']['local_path']=str(p)
write(O/'loaded-provider-bindings-v160.json',d)
print(json.dumps({'owned_processes':d['examined_owned_processes'],'mapping_records':len(d['loaded_provider_mapping_records']),'unique_libraries':sorted({(r['path'],r['sha256'],r['path_stat_matches_mapping']) for r in d['loaded_provider_mapping_records']}),'header':d['header'],'declarations':d['declarations']},ensure_ascii=False))
