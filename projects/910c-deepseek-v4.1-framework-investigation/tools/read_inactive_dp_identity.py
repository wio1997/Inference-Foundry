#!/usr/bin/env python3
"""Current read-only source identity, including missing worker callees."""
import hashlib
import json
from pathlib import Path
import subprocess
from read_metadata_reader_identity import BASE, FROZEN, REMOTE

OUT = Path('audit/execution-search')
NAMES = [
    'vllm-ascend/vllm_ascend/worker/worker.py',
    'vllm-ascend/vllm_ascend/worker/model_runner_v1.py',
    'vllm-ascend/vllm_ascend/ascend_config.py',
    'vllm-ascend/vllm_ascend/ascend_forward_context.py',
    'vllm-ascend/vllm_ascend/utils.py',
    'vllm-ascend/vllm_ascend/spec_decode/dspark_proposer.py',
    'vllm-ascend/vllm_ascend/spec_decode/llm_base_proposer.py',
    'vllm-ascend/vllm_ascend/models/deepseek_v41/model.py',
    'vllm-ascend/vllm_ascend/models/deepseek_v41/dspark.py',
    'vllm-ascend/vllm_ascend/attention/dsa_v41.py',
    'vllm-ascend/vllm_ascend/attention/context_parallel/dsa_v41_cp.py',
    'vllm-ascend/vllm_ascend/ops/fused_moe/prepare_finalize.py',
    'vllm-ascend/vllm_ascend/ops/fused_moe/fused_moe.py',
    'vllm-ascend/vllm_ascend/ops/fused_moe/shared_experts.py',
    'vllm-ascend/vllm_ascend/model_executor/warmup/kernel_warmup.py',
    'vllm-ascend/vllm_ascend/profiler/torch_npu_profiler.py',
    'vllm/vllm/v1/engine/core.py',
    'vllm/vllm/v1/worker/gpu_worker.py',
    'vllm/vllm/v1/worker/gpu_model_runner.py',
    'vllm/vllm/v1/worker/workspace.py',
]


def main():
    indexed = {r['runtime_path']:Path(r['local_path']) for r in
               json.loads((BASE/'full-review/flow-index.json').read_bytes())['files']}
    for name in ['embedding-context-execution-identity.json','staging-execution-identity.json',
                 'metadata-host-execution-identity.json']:
        for r in json.loads((OUT/name).read_bytes()).get('sources',[]):
            indexed[r['runtime_path']] = Path(r['local_path'])
    specs = []
    for name in NAMES:
        runtime = '/vllm-workspace/'+name
        p = next((p for p in [indexed.get(runtime),FROZEN/name] if p and p.is_file()), None)
        specs.append(dict(runtime_path=runtime,local_path=str(p) if p else None,
                          local_sha256=hashlib.sha256(p.read_bytes()).hexdigest() if p else None))
    r = subprocess.run(['/Users/wio/.local/bin/baai-ssh','171','--','python3','-c',REMOTE,json.dumps(specs)],
                       capture_output=True,text=True,timeout=55)
    assert r.returncode==0,r.stderr
    d = json.loads(r.stdout)
    for row in d['sources']:
        content = row.pop('content')
        if content is not None:
            p = OUT/'inactive-dp-runtime-source'/Path(row['runtime_path']).relative_to('/vllm-workspace')
            p.parent.mkdir(parents=True,exist_ok=True);p.write_text(content)
            row['previous_local_path'] = row['local_path'];row['local_path'] = str(p)
            assert hashlib.sha256(p.read_bytes()).hexdigest()==row['runtime_sha256']
    d['reader_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    d['scope'] = 'After successful171 hostname check; current original container/source tree only, not loaded Python bytecode or a new device/profile result.'
    (OUT/'inactive-dp-execution-identity.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(sources=len(d['sources']),matched=sum(r['matches_local'] for r in d['sources']),
                          fetched=[r['runtime_path'] for r in d['sources'] if not r['matches_local']],
                          new_NPU_work=0,remote_writes=[])))


if __name__=='__main__':
    main()
