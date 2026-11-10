#!/usr/bin/env python3
"""Read-only catalog of installed CANN9.2.0 direct-transfer structs and checks."""
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
root=Path('/usr/local/Ascend/cann-9.2.0/aarch64-linux/asc')
names=['kernel_struct_data_copy.h','kernel_operator_data_copy_intf.h','kernel_operator_data_copy_intf_impl.h','kernel_operator_data_copy_impl.h','kernel_operator_check_vec_data_copy.h','kernel_operator_check.h']
rows=[]
for n in names:
 for p in root.rglob(n):
  if not p.is_file() or not any(k in str(p) for k in ['/interface/','/dav_c220/','/dav_c2201/','/check/','/basic_api/kernel_operator_data_copy_intf_impl.h']):continue
  b=p.read_bytes();lines=b.decode().splitlines();hits=[]
  for i,l in enumerate(lines):
   if any(k in l for k in ['struct Nd2NzParams','struct Nd2NzParamsV2','CheckGmMemOverflowNd2Nz','Check.*Nd2Nz','nValue','srcDValue','Nd2NzParams&','Nd2NzParams &']):hits.append(dict(line=i+1,text=l[:240]))
  rows.append(dict(path=str(p),bytes=len(b),sha256=sha256(b).hexdigest(),selected_hits=hits[:90]))
print(json.dumps(dict(files=rows)))
"""
d=json.loads(subprocess.check_output(['docker','exec',name,'python3','-B','-I','-S','-c',inner]));d.update(container_id=c['Id'],image=c['Image'],new_NPU_work=0,new_profile_collection=0,service_requests=0,model_loads=0,remote_writes=[]);print(json.dumps(d))
'''

if __name__=='__main__':
 r=subprocess.run(['/Users/wio/.local/bin/baai-ssh','171','--','python3','-B','-I','-S','-c',REMOTE],capture_output=True,text=True,timeout=120)
 if r.returncode:raise RuntimeError(r.stderr)
 d=json.loads(r.stdout);d['reader_sha256']=sha256(Path(__file__).read_bytes()).hexdigest();Path('audit/execution-search/smla-l1-catalog.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');print(json.dumps(dict(files=len(d['files']),selected=[dict(path=x['path'],selected_hits=x['selected_hits'][:10]) for x in d['files']]),ensure_ascii=False))
