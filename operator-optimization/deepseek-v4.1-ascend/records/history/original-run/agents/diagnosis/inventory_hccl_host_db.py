#!/usr/bin/env python3
"""Inventory only eight known host folders, without opening databases or device nodes."""
import hashlib, json, os, stat, sys
from pathlib import Path

root = Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-20261028/diagnostic/runtime/diagnostic-traces')
items = json.loads(sys.argv[1])
assert len(items) == 8 and sorted(x['rank'] for x in items) == list(range(8))
result = []
for item in items:
    start = root / item['relative_path']
    if start.is_symlink() or hashlib.sha256(start.read_bytes()).hexdigest() != item['sha256']:
        raise RuntimeError('start_info identity mismatch: rank ' + str(item['rank']))
    host = start.parent
    stack = [(host, 0)]
    count = 0
    matches, failures, skipped_dirs = [], [], []
    while stack:
        d, depth = stack.pop()
        try:
            with os.scandir(d) as iterator:
                entries = list(iterator)
        except OSError as e:
            failures.append({'relative_dir':str(d.relative_to(host)), 'error':str(e)})
            continue
        for ent in entries:
            count += 1
            if count > 5000:
                raise RuntimeError('directory entry bound exceeded: rank ' + str(item['rank']))
            try:
                st = ent.stat(follow_symlinks=False)
                relative = str(Path(ent.path).relative_to(host))
                if stat.S_ISLNK(st.st_mode):
                    skipped_dirs.append({'path':relative,'reason':'symlink'})
                elif stat.S_ISDIR(st.st_mode):
                    if depth < 4: stack.append((Path(ent.path), depth + 1))
                    else: skipped_dirs.append({'path':relative,'reason':'depth_limit'})
                elif stat.S_ISREG(st.st_mode) and ent.name.lower() == 'hccl.db':
                    matches.append({'path':relative,'size_bytes':st.st_size,'mtime_ns':st.st_mtime_ns})
            except OSError as e:
                failures.append({'relative_dir':str(d.relative_to(host)), 'error':str(e)})
    result.append({'rank':item['rank'],'host_relative':str(host.relative_to(root)),
                   'start_info_sha256':item['sha256'],'entries_seen':count,'hccl_db':matches,
                   'failures':failures,'not_traversed':skipped_dirs})
print(json.dumps({'scope':'Only 8 indexed host directories, no symlinks, depth <= 4; filename hccl.db, case insensitive',
                  'ranks':result,'no_device_work':True},ensure_ascii=False,indent=2))
