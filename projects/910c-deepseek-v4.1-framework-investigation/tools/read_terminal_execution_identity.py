#!/usr/bin/env python3
"""Read-only actual-runtime MTP-bank reference closure; no tensor imports/work."""
import hashlib
import json
from pathlib import Path
import subprocess

OUT = Path('audit/execution-search')
INNER = r'''
import hashlib,json,pathlib,re
roots=[pathlib.Path('/vllm-workspace/vllm'),pathlib.Path('/vllm-workspace/vllm-ascend')]
pattern=re.compile(r'_mtp_hidden_buffer|get_mtp_target_hidden_states|needs_mtp_hidden_states')
files=[];hits=[]
for root in roots:
 for p in sorted(root.rglob('*')):
  if not p.is_file() or '.git' in p.parts or p.suffix not in ('.py','.cpp','.h','.hpp','.cc','.c','.cu'): continue
  raw=p.read_bytes();files.append((str(p),hashlib.sha256(raw).hexdigest()))
  text=raw.decode('utf-8',errors='replace');lines=text.splitlines()
  matching=[dict(line=i+1,text=line) for i,line in enumerate(lines) if pattern.search(line)]
  if matching:hits.append(dict(path=str(p),sha256=hashlib.sha256(raw).hexdigest(),matches=matching))
print(json.dumps(dict(roots=[str(r) for r in roots],pattern=pattern.pattern,scanned_file_count=len(files),scanned_paths_and_sha256=files,hits=hits,qualification='Literal reference closure for these exact identifiers in frozen repository source, not a proof against arbitrary external reflection/hooks or other model/config branches.',new_NPU_work=False)))
'''
REMOTE = r'''
import json,subprocess,sys
name='wio-vllm-v41-repro171-20261102'
i,=json.loads(subprocess.check_output(['docker','inspect',name]))
assert i['State']['Running']
assert i['Image']=='sha256:ce42d65b2403992ad2e2b06c61f4f048babb57e455cfe307b018345e36c5e98b'
r=subprocess.run(['docker','exec',name,'python3','-I','-S','-c',sys.argv[1]],capture_output=True,text=True,timeout=50)
assert r.returncode==0,r.stderr
d=json.loads(r.stdout);d.update(container_id=i['Id'],started_at=i['State']['StartedAt'],image=i['Image'],remote_writes=[],service_requests=0)
print(json.dumps(d))
'''

if __name__ == '__main__':
    r=subprocess.run(['/Users/wio/.local/bin/baai-ssh','171','--','python3','-c',REMOTE,INNER],capture_output=True,text=True,timeout=65)
    assert r.returncode==0,r.stderr
    d=json.loads(r.stdout)
    paths=d.pop('scanned_paths_and_sha256')
    inventory=OUT/'terminal-runtime-source-inventory.json'
    inventory.write_text(json.dumps(paths,indent=2)+'\n')
    d.update(reader_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),inventory_path=str(inventory),inventory_sha256=hashlib.sha256(inventory.read_bytes()).hexdigest())
    (OUT/'terminal-execution-identity.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(scanned_file_count=d['scanned_file_count'],matched_files=len(d['hits']),active_model_matches=[h for h in d['hits'] if h['path'].endswith('/vllm_ascend/models/deepseek_v41/model.py') or h['path'].endswith('/vllm_ascend/worker/model_runner_v1.py')],remote_writes=[],service_requests=0,new_NPU_work=False),ensure_ascii=False))
