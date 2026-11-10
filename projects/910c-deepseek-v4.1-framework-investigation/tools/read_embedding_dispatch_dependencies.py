#!/usr/bin/env python3
"""Follow the exact registered embedding override and processor imports."""
from pathlib import Path
from hashlib import sha256
import json,subprocess
from read_swa_query_producer_entry import REMOTE
ROOT=Path('audit/execution-search')
def main():
    names=['/vllm-workspace/vllm-ascend/vllm_ascend/ops/vocab_parallel_embedding.py','/vllm-workspace/vllm/vllm/models/deepseek_v41/common/mm_preprocess.py']
    spec=[dict(runtime_path=n,local_path=None,local_sha256=None) for n in names]
    r=subprocess.run(['/Users/wio/.local/bin/baai-ssh','171','--','python3','-B','-I','-S','-c',REMOTE,json.dumps(spec)],capture_output=True,text=True,timeout=90)
    if r.returncode:raise RuntimeError(r.stderr)
    d=json.loads(r.stdout)
    for s in d['sources']:
        body=s.pop('content');p=ROOT/'embedding-dispatch-source'/s['runtime_path'].lstrip('/');p.parent.mkdir(parents=True,exist_ok=True);p.write_text(body);assert sha256(p.read_bytes()).hexdigest()==s['runtime_sha256'];s['local_path']=str(p)
    d.update(reader_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),whole_chain_complete=False,qualification='Additional imported class/processor source only; no live-class or executed-branch evidence.')
    (ROOT/'embedding-dispatch-dependencies.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');print(json.dumps(dict(sources=2,new_bodies=2,new_NPU_work=0,remote_writes=[])))
if __name__=='__main__':main()
