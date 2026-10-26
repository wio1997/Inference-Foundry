#!/usr/bin/env python3
"""Copy public checkpoint to local disk, verifying every file with SHA256."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import threading
import time

root = Path(os.environ['WIO_ROOT'])
source = Path(os.environ['PUBLIC_MODEL_SOURCE'])
target = root / source.name
staging = root / ('.' + source.name + '.partial')
project = root / 'projects/910c-deepseek-v4.1-flash'
run = project / 'runs/copy-checkpoint'
run.mkdir(parents=True, exist_ok=True)
lock = (run / 'copy.lock').open('w')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
if target.exists():
    raise SystemExit('Final target already exists; inspect it before copying')
expected_config = '43d3b2bc55626893d59275b32e04068c0fc916f5c382eb91de73bdbade9c88f7'
assert hashlib.sha256((source / 'config.json').read_bytes()).hexdigest() == expected_config
files = []
for f in sorted(source.rglob('*')):
    if f.is_symlink():
        raise RuntimeError('Source contains a symlink: ' + str(f))
    if f.is_file():
        stat = f.stat()
        files.append({'path': str(f.relative_to(source)), 'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns})
(run / 'source-manifest.json').write_text(json.dumps(files, indent=2) + '\n')
total = sum(x['bytes'] for x in files)
if shutil.disk_usage(root).free < total + 100 * 1024**3:
    raise SystemExit('Insufficient free disk space')
staging.mkdir(exist_ok=True)
for p in source.rglob('*'):
    if p.is_dir():
        (staging / p.relative_to(source)).mkdir(parents=True, exist_ok=True)
mutex = threading.Lock()
done = []
copied = 0
started = time.time()
state = {'state': 'COPYING', 'pid': os.getpid(), 'source': str(source), 'target': str(target), 'staging': str(staging), 'total_bytes': total, 'total_files': len(files), 'copied_bytes': 0, 'verified_bytes': 0, 'verified_files': 0, 'started_at': datetime.now().astimezone().isoformat(), 'workers': 8}
def save_state():
    state['updated_at'] = datetime.now().astimezone().isoformat()
    state['elapsed_seconds'] = time.time() - started
    tmp = run / 'status.json.tmp'
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(run / 'status.json')
save_state()
def copy_one(item):
    global copied
    src = source / item['path']
    dst = staging / item['path']
    dst.parent.mkdir(parents=True, exist_ok=True)
    partial = dst.with_name(dst.name + '.copying')
    digest = hashlib.sha256()
    with src.open('rb') as reader, partial.open('wb') as writer:
        while True:
            chunk = reader.read(8 * 1024**2)
            if not chunk:
                break
            writer.write(chunk)
            digest.update(chunk)
            with mutex:
                copied += len(chunk)
                state['copied_bytes'] = copied
        writer.flush()
        os.fsync(writer.fileno())
    check = hashlib.sha256()
    with partial.open('rb') as reader:
        for chunk in iter(lambda: reader.read(8 * 1024**2), b''):
            check.update(chunk)
    if digest.digest() != check.digest():
        raise RuntimeError('Destination SHA256 mismatch: ' + item['path'])
    after = src.stat()
    if after.st_size != item['bytes'] or after.st_mtime_ns != item['mtime_ns']:
        raise RuntimeError('Source changed during copy: ' + item['path'])
    shutil.copystat(src, partial)
    partial.replace(dst)
    return {**item, 'sha256': digest.hexdigest(), 'destination_sha256_verified': True}
try:
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(copy_one, item) for item in sorted(files, key=lambda x: x['bytes'], reverse=True)]
        pending = set(futures)
        while pending:
            finished = [future for future in pending if future.done()]
            for future in finished:
                receipt = future.result()
                done.append(receipt)
                state['verified_files'] += 1
                state['verified_bytes'] += receipt['bytes']
                print(json.dumps({'file': receipt['path'], 'verified_files': state['verified_files'], 'total_files': len(files)}, ensure_ascii=False), flush=True)
                pending.remove(future)
            with mutex:
                save_state()
            if pending:
                time.sleep(2)
    (run / 'verified-manifest.json').write_text(json.dumps(sorted(done, key=lambda x: x['path']), indent=2) + '\n')
    current = {str(p.relative_to(source)): (p.stat().st_size, p.stat().st_mtime_ns) for p in source.rglob('*') if p.is_file()}
    assert current == {x['path']: (x['bytes'], x['mtime_ns']) for x in files}, 'Source manifest changed'
    staging.rename(target)
    state['state'] = 'VERIFIED_COMPLETE'
    state['completed_at'] = datetime.now().astimezone().isoformat()
    save_state()
    print(json.dumps(state, ensure_ascii=False), flush=True)
except BaseException as error:
    state['state'] = 'FAILED'
    state['error'] = repr(error)
    save_state()
    raise
