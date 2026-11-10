#!/usr/bin/env python3
"""Inventory owned package Python text for retirement and cache-group dispatch sites."""
from pathlib import Path
from hashlib import sha256
import json,subprocess
ROOT=Path('audit/execution-search')
REMOTE=r'''
from hashlib import sha256
import json,subprocess,tarfile
name='wio-vllm-v41-repro171-20261102';c,=json.loads(subprocess.check_output(['docker','inspect',name]))
assert c['Id']=='3be44e7b5c292de6ef7fc51085bfbbad5552acc97fae39c0ae176fe8cfb387eb' and c['Image']=='sha256:ce42d65b2403992ad2e2b06c61f4f048babb57e455cfe307b018345e36c5e98b'
terms=['retire_engram_lookups','_engram_prepare_stream','get_kv_cache_groups','_get_kv_cache_groups_uniform_page_size','patch_kv_cache_interface']
inventory=[];matches=[]
p=subprocess.Popen(['docker','cp',name+':/vllm-workspace/vllm-ascend/vllm_ascend','-'],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
with tarfile.open(fileobj=p.stdout,mode='r|*') as tar:
 for m in tar:
  if not m.isfile() or not m.name.endswith('.py'):continue
  assert m.size<10_000_000
  b=tar.extractfile(m).read();path='/vllm-workspace/vllm-ascend/'+m.name;h=sha256(b).hexdigest();inventory.append(dict(runtime_path=path,sha256=h,bytes=len(b)))
  lines=b.decode().splitlines();hits=[dict(line=i+1,terms=[t for t in terms if t in s],text=s) for i,s in enumerate(lines) if any(t in s for t in terms)]
  if hits:matches.append(dict(runtime_path=path,sha256=h,hits=hits))
assert p.wait(timeout=30)==0,p.stderr.read().decode()
print(json.dumps(dict(container_id=c['Id'],image=c['Image'],container_state=c['State']['Status'],scope='All regular .py members of owned vllm_ascend package tar; exact textual term search only, no dynamic-call absence proof or claim every file reviewed',terms=terms,python_files=len(inventory),inventory=inventory,matches=matches,new_NPU_work=0,new_profile=0,service_requests=0,remote_writes=[])))
'''
def main():
    r=subprocess.run(['/Users/wio/.local/bin/baai-ssh','171','--','python3','-B','-I','-S','-c',REMOTE],capture_output=True,text=True,timeout=150)
    if r.returncode:raise RuntimeError(r.stderr)
    d=json.loads(r.stdout);d.update(reader_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),whole_chain_complete=False)
    (ROOT/'input-resource-callsite-inventory.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');print(json.dumps(dict(python_files=d['python_files'],matched_files=len(d['matches']),matches=d['matches'],new_NPU_work=0,remote_writes=[])))
if __name__=='__main__':main()
