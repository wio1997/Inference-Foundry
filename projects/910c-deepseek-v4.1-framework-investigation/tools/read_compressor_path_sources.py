#!/usr/bin/env python3
"""Read real C1/C2 producer, persistent state and index/cache consumers."""
from pathlib import Path
from hashlib import sha256
import json,subprocess
from read_swa_query_producer_entry import REMOTE
R=Path('audit/execution-search')
NAMES=[
 'vllm-ascend/vllm_ascend/models/deepseek_v41/compressor.py',
 'vllm-ascend/vllm_ascend/models/deepseek_v41/indexer.py',
 'vllm-ascend/vllm_ascend/models/deepseek_v41/model.py',
 'vllm-ascend/vllm_ascend/models/deepseek_v41/cache_config.py',
 'vllm-ascend/vllm_ascend/attention/dsa_v41.py',
 'vllm-ascend/vllm_ascend/attention/context_parallel/dsa_v41_cp.py',
 'vllm-ascend/vllm_ascend/ops/triton/compressor/compressor_triton.py',
 'vllm-ascend/vllm_ascend/ops/triton/c2_ring_metadata.py',
 'vllm-ascend/vllm_ascend/device/device_op.py',
 'vllm-ascend/vllm_ascend/device/hardware_profile.py',
 'vllm-ascend/vllm_ascend/core/kv_cache_interface.py',
 'vllm-ascend/vllm_ascend/worker/device_metadata.py',
 'vllm-ascend/vllm_ascend/ops/rope_dsv4.py',
 'vllm-ascend/vllm_ascend/ops/layernorm.py',
 'vllm-ascend/vllm_ascend/ops/triton/quantize_indexer_query.py',
 'vllm-ascend/vllm_ascend/ops/triton/prepare_indexer_indices.py',
]
def main():
    old={}
    for file in ['whole-forward-dependencies.json','whole-forward-binding-dependencies.json','index-key-publication-dependencies.json','aux-context-dependencies.json','worker-protocol-dependencies.json']:
        p=R/file
        if p.exists():old.update({x['runtime_path']:x for x in json.loads(p.read_text())['sources']})
    frozen=Path(old['/vllm-workspace/vllm-ascend/vllm_ascend/worker/model_runner_v1.py']['local_path']).parents[3];spec=[]
    for name in NAMES:
        runtime='/vllm-workspace/'+name;p=Path(old[runtime]['local_path']) if runtime in old else frozen/name
        if not p.is_file():
            found=list(R.glob('*/vllm-workspace/'+name))+list(R.glob('*/'+name))
            if found:p=found[0]
        spec.append(dict(runtime_path=runtime,local_path=str(p) if p.is_file() else None,local_sha256=sha256(p.read_bytes()).hexdigest() if p.is_file() else None))
    result=subprocess.run(['/Users/wio/.local/bin/baai-ssh','171','--','python3','-B','-I','-S','-c',REMOTE,json.dumps(spec)],capture_output=True,text=True,timeout=150)
    if result.returncode:raise RuntimeError(result.stderr)
    d=json.loads(result.stdout)
    for s in d['sources']:
        content=s.pop('content',None)
        if content is not None:
            p=R/'compressor-path-source'/s['runtime_path'].lstrip('/');p.parent.mkdir(parents=True,exist_ok=True);p.write_text(content);assert sha256(p.read_bytes()).hexdigest()==s['runtime_sha256'];s['local_path']=str(p)
    d.update(reader_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),whole_chain_complete=False,reviewed_ranges=[],qualification='Current disk source identities, not complete reviews or loaded-native/population/bit admission. No process or imports in container; docker cp stdout tar only.')
    (R/'compressor-path-dependencies.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');print(json.dumps(dict(source_pins=len(d['sources']),matches=sum(x['matches_local'] for x in d['sources']),new_bodies=sum(x['local_sha256'] is None for x in d['sources']),remote_writes=[])))
if __name__=='__main__':main()
