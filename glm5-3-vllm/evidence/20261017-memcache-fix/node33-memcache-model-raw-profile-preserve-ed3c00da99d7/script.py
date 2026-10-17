#!/usr/bin/env python3
"""Retain the original manifest's profile files before parser simplification."""
import hashlib
import json
import os
from pathlib import Path
import socket


def main():
    node = {'B4-33U-Atlas900RCKA3-32': 32, 'B5-5U-Atlas900RCKA3-33': 33}[socket.gethostname()]
    revision = 7 if node == 32 else 6
    root = Path(f'/data/wio/glm5-3-vllm-execution/20261017-memcache/pd-node{node}-fixed-v{revision}')
    rows = json.loads((root/'pd-functional-evidence-v1/profile-manifest.json').read_text())
    out = root/'profile-raw-preserved'
    out.mkdir(exist_ok=False)
    missing = []
    for row in rows:
        rel = Path(row['path'])
        assert not rel.is_absolute() and '..' not in rel.parts and rel.parts[0] == 'profile'
        source = root/rel
        if not source.exists():
            missing.append(row)
            continue
        assert not source.is_symlink()
        dest = out/rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        os.link(source, dest)
    changed = []
    for row in rows:
        path = out/row['path']
        if not path.exists():
            continue
        h = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1048576), b''):
                h.update(block)
        if path.stat().st_size != row['bytes'] or h.hexdigest() != row['sha256']:
            changed.append(row['path'])
    report = {'node': node, 'original_manifest_count': len(rows), 'preserved_count': len(rows)-len(missing),
              'missing': missing, 'changed': changed, 'method': 'Hard-link stopped raw profile files before parser cleanup, then verify original size and SHA256.'}
    (out/'preservation.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
    assert not missing and not changed


if __name__ == '__main__':
    main()
