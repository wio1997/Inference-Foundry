#!/usr/bin/env python3
"""Fresh disk identities for canonical frame consumers and overwrite/lifetime paths."""
from pathlib import Path
from hashlib import sha256
import json,subprocess
from read_prefix_retirement_identity import REMOTE

OUT=Path('audit/execution-search')
def main():
    prior=json.loads((OUT/'metadata-readers-execution-identity.json').read_text())['sources']
    names=[r['runtime_path'] for r in json.loads((OUT/'common-metadata-dependencies.json').read_text())['sources']]
    names += ['/vllm-workspace/vllm-ascend/vllm_ascend/attention/utils.py','/vllm-workspace/vllm/vllm/forward_context.py','/vllm-workspace/vllm/vllm/v1/attention/backends/utils.py','/vllm-workspace/vllm/vllm/v1/attention/backend.py','/vllm-workspace/vllm-ascend/vllm_ascend/spec_decode/dspark_proposer.py','/vllm-workspace/vllm-ascend/vllm_ascend/spec_decode/llm_base_proposer.py']
    previous=OUT/'common-frame-lifetime-dependencies.json'
    if previous.exists():prior+=json.loads(previous.read_text())['sources']
    spec=[]
    for name in names:
        row=next((r for r in reversed(prior) if r['runtime_path']==name),None)
        p=Path(row['local_path']) if row else None
        spec.append(dict(runtime_path=name,local_path=str(p) if p else None,local_sha256=sha256(p.read_bytes()).hexdigest() if p and p.exists() else None))
    r=subprocess.run(['/Users/wio/.local/bin/baai-ssh','171','--','python3','-B','-I','-S','-c',REMOTE.replace("'python3','-I','-S'","'python3','-B','-I','-S'"),json.dumps(spec)],capture_output=True,text=True,timeout=100)
    if r.returncode:raise RuntimeError(r.stderr)
    d=json.loads(r.stdout);retrieved=[]
    for row in d['sources']:
        body=row.pop('content',None)
        if body is not None:
            p=OUT/'common-frame-lifetime-runtime-source'/row['runtime_path'].lstrip('/');p.parent.mkdir(parents=True,exist_ok=True);p.write_text(body);assert sha256(p.read_bytes()).hexdigest()==row['runtime_sha256'];row['local_path']=str(p);retrieved.append(row['runtime_path'])
    d.update(reader_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),scope='Current disk identity only; explicit source ranges and actual consumer/overwrite contracts reviewed separately.')
    previous.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');print(json.dumps(dict(sources=len(d['sources']),matching_previous=sum(r.get('matches_local',False) for r in d['sources']),retrieved=retrieved,new_NPU_work=0,remote_writes=[])))
if __name__=='__main__':main()
