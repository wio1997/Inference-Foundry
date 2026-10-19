#!/usr/bin/env python3
"""Read installed MemCache SDK Python/config artifacts; do not load native libraries."""
import hashlib
import importlib.metadata as md
import json
from pathlib import Path


def main():
    result = {'packages': {}}
    for package in ['memcache_hybrid', 'memfabric_hybrid']:
        dist = md.distribution(package)
        root = Path(dist.locate_file(package))
        record = {'version': dist.version, 'root': str(root), 'files': [], 'sources': {}}
        for p in sorted(root.rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts:
                record['files'].append({'path': str(p.relative_to(root)), 'bytes': p.stat().st_size})
                if p.suffix in ('.py', '.conf', '.json', '.ini', '.pyi') and p.stat().st_size < 200000:
                    b = p.read_bytes()
                    record['sources'][str(p.relative_to(root))] = {'sha256': hashlib.sha256(b).hexdigest(),
                                                                  'source': b.decode(errors='replace')}
        result['packages'][package] = record
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
