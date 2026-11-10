#!/usr/bin/env python3
"""Read installed Torch autoload guard source; never import Torch or NPU."""
from pathlib import Path
from hashlib import sha256
import json,subprocess
ROOT=Path('audit/execution-search')
INNER=r'''
from pathlib import Path
from hashlib import sha256
import json,sysconfig
root=Path(sysconfig.get_path('purelib'));p=root/'torch/__init__.py';b=p.read_bytes();lines=b.decode().splitlines()
hits=[i for i,l in enumerate(lines) if 'TORCH_DEVICE_BACKEND_AUTOLOAD' in l or '_is_device_backend_autoload_enabled()' in l]
assert hits
out=[]
for i in hits:
 a=max(0,i-5);z=min(len(lines),i+8);out.append(dict(first_line=a+1,last_line=z,lines=lines[a:z]))
print(json.dumps(dict(purelib=str(root),torch_init_path=str(p),torch_init_sha256=sha256(b).hexdigest(),autoload_guard=out,framework_imports=0)))
'''
REMOTE=r'''
import json,subprocess,sys
c,=json.loads(subprocess.check_output(['docker','inspect','wio-vllm-v41-repro171-20261102']))
assert c['Id']=='3be44e7b5c292de6ef7fc51085bfbbad5552acc97fae39c0ae176fe8cfb387eb' and c['Image']=='sha256:ce42d65b2403992ad2e2b06c61f4f048babb57e455cfe307b018345e36c5e98b'
assert c['State']['Running']
r=subprocess.run(['docker','exec',c['Id'],'python3','-B','-I','-S','-c',sys.argv[1]],capture_output=True,text=True,timeout=30)
assert r.returncode==0,r.stderr
d=json.loads(r.stdout);d.update(container_id=c['Id'],image=c['Image'],container_running=True,new_NPU_work=0,new_profile=0,service_actions=0,remote_writes=[]);print(json.dumps(d))
'''
def main():
    r=subprocess.run(['/Users/wio/.local/bin/baai-ssh','171','--','python3','-B','-I','-S','-c',REMOTE,INNER],capture_output=True,text=True,timeout=90)
    if r.returncode:raise RuntimeError(r.stderr)
    d=json.loads(r.stdout);d.update(reader_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),scope='Static installed Torch importguard source only. CPU test not executed and no Torch/framework imported.')
    (ROOT/'common-frame-cpu-import-guard.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');print(json.dumps(d,ensure_ascii=False))
if __name__=='__main__':main()
