#!/usr/bin/env python3
"""Pin DFC route-map producers and consumers through readonly docker-cp."""
from hashlib import sha256
from pathlib import Path
import json
import subprocess
from read_swa_query_producer_entry import REMOTE, FROZEN

OUT = Path('audit/execution-search')

def main():
    root = 'csrc/mc2/dispatch_ffn_combine/op_kernel/'
    names = ['dispatch_ffn_combine_kernel.hpp']
    names += ['moe_init_routing_quant_v2/' + n for n in [
        'moe_init_routing_quant_v2.cpp', 'moe_v2_fullload_dynamic_quant.h',
        'moe_v2_sort_one_core.h', 'moe_v2_sort_multi_core.h',
        'moe_v2_sort_base.h', 'moe_v2_mrgsort.h', 'moe_v2_mrgsort_out.h',
        'moe_v2_src_to_dst_op.h', 'moe_v2_expert_token_out.h', 'moe_v2_common.h',
    ]]
    specs = []
    for name in names:
        p = FROZEN / 'vllm-ascend' / root / name
        specs.append(dict(runtime_path='/vllm-workspace/vllm-ascend/' + root + name,
                          local_path=str(p), local_sha256=sha256(p.read_bytes()).hexdigest()))
    result = subprocess.run(['/Users/wio/.local/bin/baai-ssh', '171', '--',
                             'python3', '-B', '-I', '-S', '-c', REMOTE, json.dumps(specs)],
                            capture_output=True, text=True, timeout=115)
    if result.returncode:
        raise RuntimeError(result.stderr)
    data = json.loads(result.stdout)
    for row in data['sources']:
        body = row.pop('content', None)
        if body is not None:
            p = OUT / 'dfc-row-map-source' / row['runtime_path'].lstrip('/')
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(body)
            assert sha256(p.read_bytes()).hexdigest() == row['runtime_sha256']
            row['local_path'] = str(p)
    data.update(reader_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
                scope='Current disk identities only. Listed body review, native key, sort stability and device bit gates separate.')
    (OUT / 'dfc-row-map-dependencies.json').write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(sources=len(data['sources']), matches=sum(r.get('matches_local', False) for r in data['sources']),
                         container_state=data['container_state'], new_NPU_work=0)))

if __name__ == '__main__':
    main()
