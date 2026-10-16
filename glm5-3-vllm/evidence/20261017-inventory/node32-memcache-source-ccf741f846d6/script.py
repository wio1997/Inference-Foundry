#!/usr/bin/env python3
"""Inventory installed MemCache/MemFabric interfaces without NPU initialization or installation."""
import hashlib
import importlib.metadata as md
import json
from pathlib import Path


def main():
    root = Path('/vllm-workspace/vllm-ascend')
    store = root/'vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store'
    paths = sorted(store.rglob('*.py'))
    paths += sorted((root/'vllm_ascend/distributed/kv_transfer/utils').glob('*mem*py'))
    paths += [root/'docs/source/user_guide/feature_guide/kv_cache_pool_ascend_store.md',
              root/'docs/source/tutorials/models/GLM5.3.md']
    packages = []
    for dist in md.distributions():
        name = dist.metadata.get('Name', '')
        if any(s in name.lower() for s in ('memcache', 'memfabric', 'datasystem', 'hixl', 'mooncake')):
            packages.append({'name': name, 'version': dist.version,
                             'location': str(dist.locate_file(''))})
    result = {'packages': packages, 'directory': [str(p) for p in sorted(store.glob('*'))], 'files': {}}
    for p in paths:
        if not p.is_file():
            result['files'][str(p)] = {'exists': False}
            continue
        b = p.read_bytes()
        result['files'][str(p)] = {'bytes': len(b), 'sha256': hashlib.sha256(b).hexdigest(),
                                  'source': b.decode() if len(b) < 350000 else 'size limit'}
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
