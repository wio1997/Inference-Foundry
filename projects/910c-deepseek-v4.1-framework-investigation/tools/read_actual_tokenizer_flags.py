#!/usr/bin/env python3
"""Read only selected tokenizer flag, never emit process args or other env."""
from hashlib import sha256
import json
from pathlib import Path
import subprocess

REMOTE=r'''
import subprocess,json,sys
name='wio-vllm-v41-repro171-20261102';item,=json.loads(subprocess.check_output(['docker','inspect',name]));assert item['State']['Running'] and item['Image']=='sha256:ce42d65b2403992ad2e2b06c61f4f048babb57e455cfe307b018345e36c5e98b'
inner=r"""from pathlib import Path
import os,json
rows=[];vanished=0
for p in Path('/proc').iterdir():
 if not p.name.isdigit() or int(p.name)==os.getpid():continue
 try:
  # No args or non-whitelisted env is returned or written.
  selected={}
  for piece in (p/'environ').read_bytes().split(b'\x00'):
   k,sep,v=piece.partition(b'=')
   if k==b'VLLM_USE_FASTOKENS':selected[k.decode()]=v.decode()
  rows.append(dict(pid=int(p.name),VLLM_USE_FASTOKENS=selected.get('VLLM_USE_FASTOKENS'),initial_exec_environment_only=True))
 except (OSError,ValueError):vanished+=1
print(json.dumps(dict(process_flags=rows,vanished_or_unreadable=vanished,scope='Initial exec environment bytes only, not loaded Python env/cache introspection. No non-whitelisted value or process args emitted.')))
"""
r=subprocess.run(['docker','exec',name,'python3','-B','-I','-S','-c',inner],capture_output=True,text=True)
if r.returncode:print(r.stderr,file=sys.stderr);sys.exit(r.returncode)
x=json.loads(r.stdout);x.update(container_id=item['Id'],image=item['Image'],model_weights_loads=0,new_NPU_work=0,service_requests=0,remote_writes=[]);print(json.dumps(x))
'''


def main():
    r=subprocess.run(['/Users/wio/.local/bin/baai-ssh','171','--','python3','-B','-I','-S','-c',REMOTE],capture_output=True,text=True,timeout=100)
    if r.returncode:raise RuntimeError(r.stderr[-2000:])
    x=json.loads(r.stdout);x['reader_sha256']=sha256(Path(__file__).read_bytes()).hexdigest()
    Path('audit/execution-search/actual-tokenizer-runtime-flags.json').write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(processes=len(x['process_flags']),values=sorted(set(str(r['VLLM_USE_FASTOKENS']) for r in x['process_flags'])),new_NPU_work=0,remote_writes=[])))


if __name__=='__main__':main()
