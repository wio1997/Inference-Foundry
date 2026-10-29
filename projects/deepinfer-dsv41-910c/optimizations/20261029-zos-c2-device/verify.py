#!/usr/bin/env python3
"""Verify all C2 device-commit patch inputs or outputs against the manifest."""
import argparse
import hashlib
import json
from pathlib import Path

parser=argparse.ArgumentParser()
parser.add_argument('source',type=Path)
parser.add_argument('stage',choices=('base','candidate'))
args=parser.parse_args()
root=args.source.resolve()
manifest=json.loads(Path(__file__).with_name('manifest.json').read_text())
key=args.stage+'_sha256'
for name,record in manifest['files'].items():
    path=(root/name).resolve()
    if not path.is_relative_to(root):raise ValueError('manifest path escapes source')
    expected=record[key]
    if expected is None:
        if path.exists():raise ValueError('unexpected pre-existing file: '+name)
    else:
        actual=hashlib.sha256(path.read_bytes()).hexdigest()
        if actual!=expected:raise ValueError('SHA256 mismatch: '+name)
print('PASS:',args.stage,'files:',len(manifest['files']))
