#!/usr/bin/env python3
"""Inventory owned-container SMLA metadata API producer files, without backend import."""
from pathlib import Path
from hashlib import sha256
import subprocess,json

REMOTE=r'''
import subprocess,json
name='wio-vllm-v41-repro171-20261102';c,=json.loads(subprocess.check_output(['docker','inspect',name]));assert c['State']['Running'] and c['Image']=='sha256:ce42d65b2403992ad2e2b06c61f4f048babb57e455cfe307b018345e36c5e98b'
inner=r"""
from pathlib import Path
from hashlib import sha256
import json
root=Path('/vllm-workspace/vllm-ascend/csrc/attention/sparse_flash_mla_metadata')
files=[]
for p in sorted(root.rglob('*')):
 if p.is_file() and p.suffix in ['.h','.hpp','.cpp','.cc','.json','.txt']:
  b=p.read_bytes();files.append(dict(path=str(p),bytes=len(b),sha256=sha256(b).hexdigest()))
print(json.dumps(dict(metadata_sources=files)))
"""
d=json.loads(subprocess.check_output(['docker','exec',name,'python3','-B','-I','-S','-c',inner]));d.update(container_id=c['Id'],image=c['Image'],new_NPU_work=0,model_loads=0,service_requests=0,new_profile_collection=0,remote_writes=[]);print(json.dumps(d))
'''

if __name__=='__main__':
 r=subprocess.run(['/Users/wio/.local/bin/baai-ssh','171','--','python3','-B','-I','-S','-c',REMOTE],capture_output=True,text=True,timeout=120)
 if r.returncode:raise RuntimeError(r.stderr)
 d=json.loads(r.stdout);d['reader_sha256']=sha256(Path(__file__).read_bytes()).hexdigest();Path('audit/execution-search/smla-schedule-catalog.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');print(json.dumps(d))
