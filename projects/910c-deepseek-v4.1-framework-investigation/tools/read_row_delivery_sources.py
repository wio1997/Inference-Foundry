#!/usr/bin/env python3
"""Read only concrete input-tail and MoE delivery dependencies on owned 171."""
from pathlib import Path
from hashlib import sha256
import json, subprocess
from read_swa_query_producer_entry import REMOTE

ROOT = Path('audit/execution-search')

def main():
    prior = {}
    for name in ['worker-protocol-dependencies.json', 'worker-protocol-aux-dependencies.json', 'model-gate-dependencies.json', 'model-gate-consumer-dependencies.json', 'input-resource-source-review.json', 'embedding-static-source-review.json']:
        prior.update({s['runtime_path']: s for s in json.loads((ROOT / name).read_text())['sources']})
    frozen = Path(prior['/vllm-workspace/vllm-ascend/vllm_ascend/worker/model_runner_v1.py']['local_path']).parents[3]
    names = [
        'vllm-ascend/vllm_ascend/worker/model_runner_v1.py',
        'vllm/vllm/v1/worker/gpu_model_runner.py',
        'vllm/vllm/v1/utils.py',
        'vllm-ascend/vllm_ascend/spec_decode/utils.py',
        'vllm-ascend/vllm_ascend/spec_decode/llm_base_proposer.py',
        'vllm-ascend/vllm_ascend/spec_decode/dspark_proposer.py',
        'vllm-ascend/vllm_ascend/spec_decode/dflash_proposer.py',
        'vllm-ascend/vllm_ascend/ops/triton/spec_decode/utils.py',
        'vllm-ascend/vllm_ascend/models/deepseek_v41/model.py',
        'vllm-ascend/vllm_ascend/models/deepseek_v41/dspark.py',
        'vllm-ascend/vllm_ascend/models/deepseek_v41/engram/hash_state.py',
        'vllm-ascend/vllm_ascend/ops/fused_moe/fused_moe.py',
        'vllm-ascend/vllm_ascend/ops/fused_moe/moe_comm_method.py',
        'vllm-ascend/vllm_ascend/ops/fused_moe/prepare_finalize.py',
        'vllm-ascend/vllm_ascend/ops/fused_moe/routed_experts.py',
        'vllm-ascend/vllm_ascend/ops/fused_moe/shared_experts.py',
        'vllm-ascend/vllm_ascend/ops/fused_moe/dataclass/prepare_finalize.py',
        'vllm-ascend/vllm_ascend/ascend_forward_context.py',
        'vllm-ascend/vllm_ascend/quantization/methods/w8a8/w8a8_dynamic.py',
        'vllm-ascend/vllm_ascend/ops/fused_moe/dataclass/fused_experts.py',
        'vllm-ascend/vllm_ascend/ops/fused_moe/token_dispatcher.py',
        'vllm-ascend/vllm_ascend/ascend_config.py',
        'vllm-ascend/vllm_ascend/device/hardware_profile.py',
        'vllm-ascend/csrc/mc2/dispatch_ffn_combine/op_kernel/dispatch_ffn_combine.h',
        'vllm-ascend/csrc/mc2/dispatch_ffn_combine/op_kernel/dispatch_ffn_combine_kernel.hpp',
        'vllm-ascend/csrc/mc2/dispatch_ffn_combine/op_kernel/unpermute/moe_token_unpermute.h',
        'vllm-ascend/csrc/mc2/dispatch_ffn_combine/op_host/dispatch_ffn_combine_tiling.cpp',
        'vllm-ascend/csrc/mc2/dispatch_ffn_combine/dispatch_ffn_combine_torch_adpt.h',
        'vllm-ascend/csrc/mc2/dispatch_ffn_combine/op_host/op_api/aclnn_dispatch_ffn_combine.cpp',
        'vllm-ascend/csrc/torch_binding.cpp',
    ]
    specs = []
    for name in names:
        runtime = '/vllm-workspace/' + name
        path = Path(prior[runtime]['local_path']) if runtime in prior else frozen / name
        specs.append(dict(runtime_path=runtime, local_path=str(path) if path.is_file() else None,
                          local_sha256=sha256(path.read_bytes()).hexdigest() if path.is_file() else None))
    p = subprocess.run(['/Users/wio/.local/bin/baai-ssh', '171', '--', 'python3', '-B', '-I', '-S', '-c', REMOTE, json.dumps(specs)], capture_output=True, text=True, timeout=150)
    if p.returncode:
        raise RuntimeError(p.stderr)
    result = json.loads(p.stdout)
    for row in result['sources']:
        content = row.pop('content', None)
        if content is not None:
            path = ROOT / 'row-delivery-source' / row['runtime_path'].lstrip('/')
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
            assert sha256(path.read_bytes()).hexdigest() == row['runtime_sha256']
            row['local_path'] = str(path)
    result.update(reader_sha256=sha256(Path(__file__).read_bytes()).hexdigest(), whole_chain_complete=False,
                  qualification='Disk identities only. Reviewed ranges are separate; no Torch import, container process, native bits, population or health claim.')
    (ROOT / 'row-delivery-dependencies.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(sources=len(result['sources']), matches=sum(r['matches_local'] for r in result['sources']), new_bodies=sum(r['local_sha256'] is None for r in result['sources']), remote_writes=[])))

if __name__ == '__main__':
    main()
