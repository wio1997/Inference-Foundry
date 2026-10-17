#!/usr/bin/env python3
"""Read installed MemFabric data-op enums used to interpret runtime logs."""
import hashlib
import json
from pathlib import Path


def main():
    root = Path('/usr/local/python3.12.13/lib/python3.12/site-packages/memfabric_hybrid')
    result = {}
    for relative in ['include/hybm/hybm_def.h', 'include/smem/host/smem_bm_def.h']:
        data = (root/relative).read_bytes()
        result[relative] = {'sha256': hashlib.sha256(data).hexdigest(), 'source': data.decode()}
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
