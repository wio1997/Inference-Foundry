#!/usr/bin/env python3
"""Pin the last target layer's actual physical-cache chain; read-only remote."""
import hashlib
import json
from pathlib import Path
import subprocess
from read_metadata_reader_identity import BASE, FROZEN, REMOTE

OUT = Path('audit/execution-search')
NAMES = [
    'vllm-ascend/vllm_ascend/models/deepseek_v41/model.py',
    'vllm-ascend/vllm_ascend/models/deepseek_v41/cache_config.py',
    'vllm-ascend/vllm_ascend/core/kv_cache_interface.py',
    'vllm-ascend/vllm_ascend/attention/dsa_v41.py',
    'vllm-ascend/vllm_ascend/attention/context_parallel/dsa_v41_cp.py',
    'vllm-ascend/vllm_ascend/attention/context_parallel/dsa_cp.py',
    'vllm-ascend/vllm_ascend/worker/model_runner_v1.py',
    'vllm-ascend/vllm_ascend/worker/block_table.py',
    'vllm-ascend/vllm_ascend/ops/triton/compute_slot_mapping.py',
    'vllm-ascend/vllm_ascend/device/hardware_profile.py',
    'vllm/vllm/v1/core/single_type_kv_cache_manager.py',
    'vllm/vllm/v1/core/kv_cache_coordinator.py',
    'vllm/vllm/v1/core/kv_cache_manager.py',
    'vllm/vllm/v1/core/block_pool.py',
    'vllm/vllm/v1/core/kv_cache_utils.py',
    'vllm/vllm/v1/kv_cache_interface.py',
]

def main():
    indexed = {r['runtime_path']: Path(r['local_path']) for r in
               json.loads((BASE/'full-review/flow-index.json').read_bytes())['files']}
    for name in ['staging-execution-identity.json', 'metadata-readers-execution-identity.json',
                 'final-cache-execution-identity.json']:
        p = OUT/name
        if p.is_file():
            indexed.update({r['runtime_path']: Path(r['local_path']) for r in
                            json.loads(p.read_bytes())['sources']})
    specs = []
    for name in NAMES:
        runtime = '/vllm-workspace/'+name
        p = next((p for p in [indexed.get(runtime), FROZEN/name] if p and p.is_file()), None)
        specs.append(dict(runtime_path=runtime, local_path=str(p) if p else None,
                          local_sha256=hashlib.sha256(p.read_bytes()).hexdigest() if p else None))
    r = subprocess.run(['/Users/wio/.local/bin/baai-ssh', '171', '--', 'python3', '-c',
                        REMOTE, json.dumps(specs)], capture_output=True, text=True, timeout=55)
    assert r.returncode == 0, r.stderr
    d = json.loads(r.stdout)
    retrieved = []
    for row in d['sources']:
        content = row.pop('content')
        if content is not None:
            p = OUT/'final-cache-runtime-source'/Path(row['runtime_path']).relative_to('/vllm-workspace')
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content)
            row['previous_local_path'] = row['local_path']
            row['local_path'] = str(p)
            assert hashlib.sha256(p.read_bytes()).hexdigest() == row['runtime_sha256']
            retrieved.append(row['runtime_path'])
    d['reader_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (OUT/'final-cache-execution-identity.json').write_text(json.dumps(d, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(dict(source_count=len(d['sources']), matching_previous=sum(r['matches_local'] for r in d['sources']),
                          retrieved=retrieved, new_NPU_work=False, remote_writes=[])))

if __name__ == '__main__':
    main()
