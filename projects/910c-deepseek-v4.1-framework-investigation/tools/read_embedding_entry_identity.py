#!/usr/bin/env python3
"""Read-only current pins for the next input/embedding/context entry."""
import hashlib
import json
from pathlib import Path
import subprocess
from read_metadata_reader_identity import BASE, FROZEN, REMOTE

OUT = Path('audit/execution-search')
NAMES = [
    'vllm-ascend/vllm_ascend/models/deepseek_v41/vl_model.py',
    'vllm-ascend/vllm_ascend/models/deepseek_v41/model.py',
    'vllm-ascend/vllm_ascend/models/deepseek_v41/dspark.py',
    'vllm-ascend/vllm_ascend/spec_decode/dspark_proposer.py',
    'vllm-ascend/vllm_ascend/spec_decode/dflash_proposer.py',
    'vllm-ascend/vllm_ascend/spec_decode/eagle_proposer.py',
    'vllm-ascend/vllm_ascend/spec_decode/llm_base_proposer.py',
    'vllm-ascend/vllm_ascend/ops/triton/spec_decode/utils.py',
    'vllm-ascend/vllm_ascend/attention/dsa_v1.py',
    'vllm-ascend/vllm_ascend/models/common/ops/sequence_parallel.py',
    'vllm-ascend/vllm_ascend/worker/model_runner_v1.py',
    'vllm/vllm/v1/worker/gpu_model_runner.py',
    'vllm/vllm/v1/spec_decode/eagle.py',
    'vllm/vllm/model_executor/models/utils.py',
    'vllm/vllm/model_executor/layers/vocab_parallel_embedding.py',
]

def main():
    indexed = {r['runtime_path']: Path(r['local_path']) for r in
               json.loads((BASE/'full-review/flow-index.json').read_bytes())['files']}
    for name in ['staging-execution-identity.json', 'metadata-readers-execution-identity.json']:
        indexed.update({r['runtime_path']:Path(r['local_path']) for r in json.loads((OUT/name).read_bytes())['sources']})
    specs = []
    for name in NAMES:
        runtime = '/vllm-workspace/'+name
        p = next((p for p in [indexed.get(runtime),FROZEN/name] if p and p.is_file()),None)
        specs.append(dict(runtime_path=runtime, local_path=str(p) if p else None,
                          local_sha256=hashlib.sha256(p.read_bytes()).hexdigest() if p else None))
    r = subprocess.run(['/Users/wio/.local/bin/baai-ssh','171','--','python3','-c',REMOTE,json.dumps(specs)],
                       capture_output=True,text=True,timeout=55)
    assert r.returncode==0,r.stderr
    d = json.loads(r.stdout)
    for row in d['sources']:
        content = row.pop('content')
        if content is not None:
            p = OUT/'embedding-entry-runtime-source'/Path(row['runtime_path']).relative_to('/vllm-workspace')
            p.parent.mkdir(parents=True,exist_ok=True);p.write_text(content)
            row['previous_local_path']=row['local_path'];row['local_path']=str(p)
            assert hashlib.sha256(p.read_bytes()).hexdigest()==row['runtime_sha256']
    d['reader_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (OUT/'embedding-context-execution-identity.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(sources=len(d['sources']),matched=sum(r['matches_local'] for r in d['sources']),
                          new_NPU_work=False,remote_writes=[])))

if __name__=='__main__':main()
