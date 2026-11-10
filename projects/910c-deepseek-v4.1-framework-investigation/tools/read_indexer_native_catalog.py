#!/usr/bin/env python3
"""Inventory owned current QLI source/package paths without framework imports."""
from hashlib import sha256
import json
from pathlib import Path
import subprocess

REMOTE = r'''
import json,subprocess
name='wio-vllm-v41-repro171-20261102'
c,=json.loads(subprocess.check_output(['docker','inspect',name]))
assert c['State']['Running'] and c['Image']=='sha256:ce42d65b2403992ad2e2b06c61f4f048babb57e455cfe307b018345e36c5e98b'
inner=r"""
from pathlib import Path
import json
base=Path('/vllm-workspace/vllm-ascend')
src=[]
for folder in ['csrc/attention/quant_lightning_indexer_v2','csrc/attention/quant_lightning_indexer_v2_metadata']:
 root=base/folder
 for p in sorted(root.rglob('*')):
  if p.is_file() and p.suffix in ('.h','.cpp','.cc','.json'):
   src.append(dict(path=str(p),bytes=p.stat().st_size))
root=base/'vllm_ascend/_cann_ops_custom/vendors/custom_transformer'
packages=[]
for folder in ['op_impl/ai_core/tbe/kernel/ascend910_93','op_impl/ai_core/tbe/config/ascend910_93']:
 for p in sorted((root/folder).glob('*')):
  if 'lightning' in p.name.lower() or 'indexer' in p.name.lower():
   packages.append(dict(path=str(p),kind='directory' if p.is_dir() else 'file',bytes=p.stat().st_size))
print(json.dumps(dict(source_inventory=src,package_inventory=packages)))
"""
d=json.loads(subprocess.check_output(['docker','exec',name,'python3','-B','-I','-S','-c',inner]))
d.update(container_id=c['Id'],image=c['Image'],new_NPU_work=0,model_loads=0,service_requests=0,new_profile_collection=0,remote_writes=[])
print(json.dumps(d))
'''


def main():
    p=subprocess.run(['/Users/wio/.local/bin/baai-ssh','171','--','python3','-B','-I','-S','-c',REMOTE],capture_output=True,text=True,timeout=100)
    if p.returncode:
        raise RuntimeError(p.stderr)
    d=json.loads(p.stdout)
    d['reader_sha256']=sha256(Path(__file__).read_bytes()).hexdigest()
    Path('audit/execution-search/indexer-native-catalog.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(d,ensure_ascii=False))


if __name__=='__main__':
    main()
