#!/usr/bin/env python3
"""Hash QLI current package and inspect only whitelisted process library data."""
from hashlib import sha256
import json
from pathlib import Path
import subprocess

REMOTE=r'''
import json,subprocess
name='wio-vllm-v41-repro171-20261102';c,=json.loads(subprocess.check_output(['docker','inspect',name]))
assert c['State']['Running'] and c['Image']=='sha256:ce42d65b2403992ad2e2b06c61f4f048babb57e455cfe307b018345e36c5e98b'
inner=r"""
from hashlib import sha256
from pathlib import Path
import json,os,subprocess
root=Path('/vllm-workspace/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer')
def pin(p):
 raw=p.read_bytes();return dict(path=str(p),bytes=len(raw),sha256=sha256(raw).hexdigest())
lib=root/'op_api/lib/libcust_opapi.so'
api=pin(lib)
r=subprocess.run(['nm','-D',str(lib)],capture_output=True,text=True)
api['exports']=[s for s in r.stdout.splitlines() if 'aclnnVllmAscendQuantLightningIndexerV2' in s]
api['nm_exit']=r.returncode
folder=root/'op_impl/ai_core/tbe/kernel/ascend910_93/vllm_ascend_quant_lightning_indexer_v2'
package=[]
for p in sorted(folder.glob('*')):
 if not p.is_file():continue
 row=pin(p)
 if p.suffix=='.json':row['metadata']=json.loads(p.read_text())
 package.append(row)
proc=[]
for p in sorted(Path('/proc').iterdir()):
 if not p.name.isdigit() or int(p.name)==os.getpid():continue
 try:
  comm=(p/'comm').read_text().strip()
  if not comm.startswith('VLLM::Worker'):continue
  env={}
  for piece in (p/'environ').read_bytes().split(b'\x00'):
   k,sep,v=piece.partition(b'=')
   if k==b'ASCEND_CUSTOM_OPP_PATH':env[k.decode()]=v.decode()
  paths=set()
  for line in (p/'maps').read_text().splitlines():
   if '/libcust_opapi.so' in line or '/libaicpu' in line and 'custom' in line:
    fields=line.split(maxsplit=5)
    if len(fields)==6:paths.add(fields[5])
  proc.append(dict(pid=int(p.name),comm=comm,initial_ASCEND_CUSTOM_OPP_PATH=env.get('ASCEND_CUSTOM_OPP_PATH'),mapped_custom_libraries=sorted(paths)))
 except (OSError,ValueError):pass
print(json.dumps(dict(custom_opapi=api,kernel_package=package,worker_processes=proc,
 scope='Current disk/package and process mappings/initial selected environment, not per-call device binary/tiling buffer or loaded source-to-binary proof. No args or unselected env emitted.')))
"""
d=json.loads(subprocess.check_output(['docker','exec',name,'python3','-B','-I','-S','-c',inner]))
d.update(container_id=c['Id'],image=c['Image'],new_NPU_work=0,model_loads=0,service_requests=0,new_profile_collection=0,remote_writes=[])
print(json.dumps(d))
'''


def main():
    r=subprocess.run(['/Users/wio/.local/bin/baai-ssh','171','--','python3','-B','-I','-S','-c',REMOTE],capture_output=True,text=True,timeout=100)
    if r.returncode:
        raise RuntimeError(r.stderr)
    d=json.loads(r.stdout);d['reader_sha256']=sha256(Path(__file__).read_bytes()).hexdigest()
    Path('audit/execution-search/indexer-package-identity.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(workers=len(d['worker_processes']),kernel_package=[{k:r[k] for k in ['path','sha256','bytes']} for r in d['kernel_package']],opapi=d['custom_opapi']),ensure_ascii=False))


if __name__=='__main__':main()
