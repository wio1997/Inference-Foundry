#!/usr/bin/env python3
"""Find registry/profile guards in current Python sources, without importing them."""
from hashlib import sha256
from pathlib import Path
import json, subprocess

REMOTE=r'''
import json,subprocess,sys
name='wio-vllm-v41-repro171-20261102'
c,=json.loads(subprocess.check_output(['docker','inspect',name]));assert c['State']['Running'] and c['Image']=='sha256:ce42d65b2403992ad2e2b06c61f4f048babb57e455cfe307b018345e36c5e98b'
r=subprocess.run(['docker','exec',name,'python3','-B','-I','-S','-c',sys.argv[1]],capture_output=True,text=True,timeout=45);assert r.returncode==0,r.stderr
print(r.stdout)
'''
INNER=r'''
from pathlib import Path
from hashlib import sha256
import json
root=Path('/vllm-workspace/vllm-ascend/vllm_ascend');out=[];n=0
symbols=['register_ascend_customop','AUTO_ENABLE_CUSTOM_OPS']
for p in sorted(root.rglob('*.py')):
 n+=1;raw=p.read_bytes();lines=raw.decode().splitlines();hits=[dict(line=i,text=l) for i,l in enumerate(lines,1) if any(s in l for s in symbols)]
 if hits:out.append(dict(runtime_path=str(p),sha256=sha256(raw).hexdigest(),hits=hits,content=raw.decode()))
print(json.dumps(dict(files_searched=n,symbols=symbols,sources=out)))
'''

if __name__=='__main__':
    r=subprocess.run(['/Users/wio/.local/bin/baai-ssh','171','--','python3','-B','-I','-S','-c',REMOTE,INNER],capture_output=True,text=True,timeout=80)
    if r.returncode: raise RuntimeError(r.stderr)
    d=json.loads(r.stdout);root=Path('audit/execution-search')
    for row in d['sources']:
        p=root/'cache-epilogue-registry-source'/row['runtime_path'].lstrip('/');p.parent.mkdir(parents=True,exist_ok=True);p.write_text(row.pop('content'));assert sha256(p.read_bytes()).hexdigest()==row['sha256'];row['local_path']=str(p)
    d.update(reader_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),new_NPU_work=0,new_profile=0,remote_writes=[])
    (root/'cache-epilogue-registry-callers.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in d.items() if k!='sources'}|dict(sources=[dict(path=s['runtime_path'],hits=s['hits']) for s in d['sources']])))
