#!/usr/bin/env python3
"""Read bounded worker and connector interfaces needed for PD validation."""
import hashlib
import json
from pathlib import Path
import re
import subprocess


def main():
    root = Path('/vllm-workspace/vllm-ascend')
    files = {}
    for rel in ['vllm_ascend/worker/worker.py', 'vllm_ascend/platform.py',
                'vllm_ascend/worker/model_runner_v1.py',
                'vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake/connector.py',
                'vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake/pull_scheduler.py']:
        path = root/rel
        data = path.read_bytes()
        lines = data.decode().splitlines()
        pattern = r'^class |def (?:load_model|initialize_from_config|execute_model|get_num_new_matched_tokens|request_finished)|kv_transfer_params|worker_cls|model_runner\.kv|self\.kv_caches|logger\.'
        selected = set()
        for i, line in enumerate(lines):
            if re.search(pattern, line):
                selected.update(range(max(0, i-2), min(len(lines), i+42)))
        files[rel] = {'sha256': hashlib.sha256(data).hexdigest(), 'total_lines': len(lines),
                      'lines': [{'line': i+1, 'text': lines[i]} for i in sorted(selected)]}
    lib = Path('/usr/local/python3.12.13/lib/python3.12/site-packages/mooncake/ascend_transport.so')
    strings = subprocess.check_output(['strings', str(lib)], text=True).splitlines()
    controls = sorted(set(s for s in strings if re.fullmatch(r'(?:ASCEND|HCCL|MOONCAKE)_[A-Z_0-9]+', s)))
    print(json.dumps({'files': files, 'te_binary_controls': controls}, indent=2))


if __name__ == '__main__':
    main()
