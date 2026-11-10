#!/usr/bin/env python3
"""Recheck exact selected package bytes read-only; no NPU/backend initialization."""
from pathlib import Path
from hashlib import sha256
import json,subprocess

if __name__=='__main__':
 root=Path('audit/execution-search');old=json.loads((root/'smla-source-identity.json').read_text());pins=[old['custom_opapi'],old['custom_binary']]
 remote=r'''
from pathlib import Path
from hashlib import sha256
import json,subprocess,sys
name='wio-vllm-v41-repro171-20261102';c,=json.loads(subprocess.check_output(['docker','inspect',name]));assert c['State']['Running'] and c['Image']=='sha256:ce42d65b2403992ad2e2b06c61f4f048babb57e455cfe307b018345e36c5e98b'
inner=r"""
from pathlib import Path
from hashlib import sha256
import json,sys
rows=[]
for pin in json.loads(sys.argv[1]):
 p=Path(pin['path']);b=p.read_bytes();h=sha256(b).hexdigest();row=dict(path=str(p),sha256=h,bytes=len(b),matches_prior=h==pin['sha256'])
 if p.suffix=='.o':
  q=p.with_suffix('.json')
  if q.is_file():
   row['metadata_path']=str(q);row['metadata_sha256']=sha256(q.read_bytes()).hexdigest();row['metadata']=json.loads(q.read_text())
 rows.append(row)
print(json.dumps(rows))
"""
d=json.loads(subprocess.check_output(['docker','exec',name,'python3','-B','-I','-S','-c',inner,sys.argv[1]]));print(json.dumps(dict(files=d,container_id=c['Id'],image=c['Image'],new_NPU_work=0,new_profile_collection=0,service_requests=0,model_loads=0,remote_writes=[],scope='Current disk package bytes and metadata; no actual percall tiling buffer or device-loaded binary attestation.')))
'''
 r=subprocess.run(['/Users/wio/.local/bin/baai-ssh','171','--','python3','-B','-I','-S','-c',remote,json.dumps(pins)],capture_output=True,text=True,timeout=120)
 if r.returncode:raise RuntimeError(r.stderr)
 d=json.loads(r.stdout);d['reader_sha256']=sha256(Path(__file__).read_bytes()).hexdigest();(root/'smla-package-recheck.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');print(json.dumps([dict(path=x['path'],matches_prior=x['matches_prior'],metadata_keys=list(x.get('metadata',{}))) for x in d['files']]))
