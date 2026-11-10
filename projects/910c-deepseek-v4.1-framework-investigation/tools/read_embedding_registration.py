#!/usr/bin/env python3
"""Follow OOT/pluggable registration and quantization package dispatch."""
from pathlib import Path
from hashlib import sha256
import json,subprocess
from read_swa_query_producer_entry import REMOTE
ROOT=Path('audit/execution-search')
def main():
    names=['/vllm-workspace/vllm/vllm/model_executor/custom_op.py','/vllm-workspace/vllm-ascend/vllm_ascend/quantization/__init__.py','/vllm-workspace/vllm-ascend/vllm_ascend/platform.py']
    spec=[dict(runtime_path=n,local_path=None,local_sha256=None) for n in names]
    r=subprocess.run(['/Users/wio/.local/bin/baai-ssh','171','--','python3','-B','-I','-S','-c',REMOTE,json.dumps(spec)],capture_output=True,text=True,timeout=90)
    if r.returncode:raise RuntimeError(r.stderr)
    d=json.loads(r.stdout)
    for s in d['sources']:
        body=s.pop('content');p=ROOT/'embedding-registration-source'/s['runtime_path'].lstrip('/');p.parent.mkdir(parents=True,exist_ok=True);p.write_text(body);assert sha256(p.read_bytes()).hexdigest()==s['runtime_sha256'];s['local_path']=str(p)
    d.update(reader_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),whole_chain_complete=False,qualification='Imported OOT registry/quant package/platform source identity only; no live registry inspection or framework import.')
    (ROOT/'embedding-registration-dependencies.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');print(json.dumps(dict(sources=3,new_bodies=3,new_NPU_work=0,remote_writes=[])))
if __name__=='__main__':main()
