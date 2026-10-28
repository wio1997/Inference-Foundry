#!/usr/bin/env python3
"""Check all files in a source tree against an explicit JSON SHA256 manifest."""
import hashlib
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
manifest = json.loads(pathlib.Path(sys.argv[2]).read_text())
if any(p.is_symlink() for p in root.rglob('*')):
    raise ValueError('source tree contains symlinks')
actual = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
if actual != set(manifest):
    raise ValueError(f'file set mismatch: expected {len(manifest)}, got {len(actual)}')
for name, expected in manifest.items():
    h = hashlib.sha256()
    with (root/name).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    if h.hexdigest() != expected:
        raise ValueError(f'hash mismatch: {name}')
print(f'PASS: {len(manifest)} source files')
