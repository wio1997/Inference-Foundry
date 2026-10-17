#!/usr/bin/env python3
"""Read the exact installed PD interfaces; no imports that initialize an NPU."""
import hashlib
import json
from pathlib import Path

ASCEND = Path('/vllm-workspace/vllm-ascend')
VLLM = Path('/vllm-workspace/vllm')


def main():
    paths = [
        ASCEND/'vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py',
        ASCEND/'vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake/connector.py',
        ASCEND/'vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake/base_worker.py',
        ASCEND/'vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake/pull_worker.py',
        ASCEND/'vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake/base_scheduler.py',
        ASCEND/'vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake/pull_scheduler.py',
        ASCEND/'vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake/utils.py',
        ASCEND/'vllm_ascend/distributed/kv_transfer/utils/mooncake_transfer_engine.py',
        ASCEND/'vllm_ascend/distributed/kv_transfer/__init__.py',
        VLLM/'vllm/distributed/kv_transfer/kv_connector/factory.py',
    ]
    root = ASCEND/'vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake'
    paths += sorted(p for p in root.glob('*.py') if p not in paths)
    result = {'directory': [str(p) for p in sorted(root.glob('*'))], 'files': {}}
    for p in paths:
        if not p.is_file():
            result['files'][str(p)] = {'exists': False}
            continue
        b = p.read_bytes()
        result['files'][str(p)] = {'sha256': hashlib.sha256(b).hexdigest(), 'bytes': len(b),
                                  'source': b.decode() if len(b) < 160000 else 'over size limit'}
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
