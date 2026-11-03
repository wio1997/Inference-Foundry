#!/usr/bin/env python3
"""Bounded, read-only source inventory. Intended for docker exec, no imports of target packages."""
import base64
import hashlib
import json
import os
from pathlib import Path

roots = (
    '/vllm-workspace/vllm-ascend/csrc/moe/hc_pre',
    '/vllm-workspace/vllm-ascend/csrc/moe/hc_post',
)
extra = ('/vllm-workspace/vllm-ascend/tests/e2e/nightly/single_node/ops/singlecard_ops/test_npu_hc_pre.py',)
allowed = {'.h', '.hpp', '.cpp', '.cc', '.c', '.py', '.json', '.md', '.yaml', '.yml'}
records=[]
missing=[]
total=0
visited_dirs=0
for root in roots:
    p=Path(root)
    if p.is_symlink(): raise RuntimeError('symlink root refused: '+root)
    if not p.exists():
        missing.append(root)
        continue
    if not p.is_dir(): raise RuntimeError('non-directory root: '+root)
    for current, dirs, names in os.walk(p, followlinks=False):
        visited_dirs+=1
        if visited_dirs>128: raise RuntimeError("directory count exceeds bound")
        if len(dirs)+len(names)>1024: raise RuntimeError("directory entry count exceeds bound")
        dirs.sort(); names.sort()
        for name in dirs:
            if (Path(current)/name).is_symlink(): raise RuntimeError('symlink dir refused')
        for name in names:
            f=Path(current)/name
            if f.is_symlink(): raise RuntimeError('symlink file refused')
            if f.suffix in allowed:
                records.append(f)
                if len(records)>128: raise RuntimeError("source file count exceeds bound")
for raw in extra:
    f=Path(raw)
    if f.is_symlink(): raise RuntimeError('symlink file refused: '+raw)
    if f.is_file(): records.append(f)
    else: missing.append(raw)
if len(records)>128: raise RuntimeError('source file count exceeds bound')
result=[]
for f in sorted(records):
    if not f.is_file(): raise RuntimeError('non-regular file: '+str(f))
    size=f.stat().st_size
    if size>1024*1024: raise RuntimeError('individual source exceeds bound: '+str(f))
    total+=size
    if total>8*1024*1024: raise RuntimeError('aggregate source exceeds bound')
    raw=f.read_bytes()
    if len(raw)!=size: raise RuntimeError('source changed while reading')
    result.append({'path':str(f),'size':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),
                   'base64':base64.b64encode(raw).decode('ascii')})
print(json.dumps({'files':result,'missing':missing},sort_keys=True))
