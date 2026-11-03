"""Resolve overlay backing identity for only two mapped provider ELF files."""
import json,sys
from finalize_origin_coverage_v9 import O,read,write
sys.path.insert(0,'/Users/wio/work/171-vllm-migration-20261102')
from control import invoke
maps=read(O/'loaded-provider-bindings-v160.json')['loaded_provider_mapping_records']
expected={}
for row in maps:expected[(row['mapped_device'],row['mapped_inode'],row['path'])]=row
code='EXPECTED = '+repr(list(expected.values()))+'\n'+r'''import json,os,subprocess,hashlib,stat
from pathlib import Path
cid='3be44e7b5c292de6ef7fc51085bfbbad5552acc97fae39c0ae176fe8cfb387eb'
d=json.loads(subprocess.check_output(['docker','inspect',cid],text=True))[0];assert d['State']['Running']
gd=d['GraphDriver']['Data'];roots=[gd.get('UpperDir','')]+gd.get('LowerDir','').split(':');roots=[Path(p) for p in roots if p]
out=[]
for row in EXPECTED:
 major,minor=[int(s,16) for s in row['mapped_device'].split(':')];candidates=[]
 for root in roots:
  p=root/row['path'].lstrip('/')
  if not p.is_file():continue
  s=p.stat()
  if s.st_ino!=row['mapped_inode'] or os.major(s.st_dev)!=major or os.minor(s.st_dev)!=minor:continue
  b=p.read_bytes();after=p.stat();assert (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns)
  candidates.append({'backing_file':str(p),'device':f'{os.major(s.st_dev):x}:{os.minor(s.st_dev):x}','inode':s.st_ino,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'stat_matches_kernel_mapping':True})
 out.append({'mapped_path':row['path'],'mapped_device':row['mapped_device'],'mapped_inode':row['mapped_inode'],'candidates':candidates,'all_candidate_bytes_match_container_view':bool(candidates) and all(c['sha256']==row['sha256'] for c in candidates),'runtime_call_selection_or_private_COW_text_attested':False})
result={'scope':'MAPPED_ELF_BACKING_FILE_IDENTITY_ONLY','container_id':cid,'exact_docker_layer_roots_checked':len(roots),'objects':out,'no_recursive_filesystem_or_other_user_process_scan':True,'no_CANN_NPU_or_process_memory_calls':True,'actual_interposition_or_bypass_tested':False,'overall_complete':False}
dest=Path('/data/tiankuan/wio/migration-171-20261102/operator-evidence-v160/mapped-provider-backing.json');assert not dest.exists();dest.write_text(json.dumps(result,indent=2));print(json.dumps(result))
'''
d=invoke('target',code,45);write(O/'mapped-provider-backing-objects-v160.json',d);print(json.dumps(d))
