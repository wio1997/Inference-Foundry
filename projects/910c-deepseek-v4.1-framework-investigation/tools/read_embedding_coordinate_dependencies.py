#!/usr/bin/env python3
"""Follow actual common metadata and DSpark query expansion dependencies."""
from pathlib import Path
from hashlib import sha256
import json,subprocess
from read_swa_query_producer_entry import REMOTE
ROOT=Path('audit/execution-search')
def main():
    prior={}
    for name in ['whole-forward-dispatch','v1-input-dependencies','embedding-context-execution-identity']:
        for s in json.loads((ROOT/(name+'.json')).read_text())['sources']:prior[s['runtime_path']]=s
    names=['/vllm-workspace/vllm-ascend/vllm_ascend/ops/triton/spec_decode/utils.py','/vllm-workspace/vllm-ascend/vllm_ascend/attention/utils.py','/vllm-workspace/vllm/vllm/v1/attention/backends/utils.py','/vllm-workspace/vllm-ascend/vllm_ascend/models/deepseek_v41/dspark.py']
    spec=[]
    for n in names:
        row=prior.get(n);p=Path(row['local_path']) if row and row.get('local_path') else None
        spec.append(dict(runtime_path=n,local_path=str(p) if p and p.exists() else None,local_sha256=sha256(p.read_bytes()).hexdigest() if p and p.exists() else None))
    r=subprocess.run(['/Users/wio/.local/bin/baai-ssh','171','--','python3','-B','-I','-S','-c',REMOTE,json.dumps(spec)],capture_output=True,text=True,timeout=90)
    if r.returncode:raise RuntimeError(r.stderr)
    d=json.loads(r.stdout)
    for s in d['sources']:
        body=s.pop('content',None)
        if body is not None:
            p=ROOT/'embedding-coordinate-source'/s['runtime_path'].lstrip('/');p.parent.mkdir(parents=True,exist_ok=True);p.write_text(body);assert sha256(p.read_bytes()).hexdigest()==s['runtime_sha256'];s['local_path']=str(p)
    d.update(reader_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),whole_chain_complete=False,qualification='Current source only: metadata attributes/aliasing, kernel dependency and source controls separate from raw trace population and primitive bits.')
    (ROOT/'embedding-coordinate-dependencies.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');print(json.dumps(dict(sources=len(d['sources']),matches=sum(s['matches_local'] for s in d['sources']),new_bodies=sum(s['local_sha256'] is None for s in d['sources']),new_NPU_work=0,remote_writes=[])))
if __name__=='__main__':main()
