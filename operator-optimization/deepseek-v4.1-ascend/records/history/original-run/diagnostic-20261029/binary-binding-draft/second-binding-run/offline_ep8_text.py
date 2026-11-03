#!/usr/bin/env python3
"""Inside task-owned stage: copy EP8 raw and produce a single-rank CPU-only DB."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import time

root = Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029')
raw = root / 'binding-diagnostic-selective2/runtime/diagnostic-traces/dp1_pp0_tp0_dcp0_ep8_rank0_746_20261029050352206_ascend_pt'
job = root / 'binary-binding-draft/second-binding-run/offline-ep8-text-20261029'
copy = job / raw.name
assert raw.is_dir() and not raw.is_symlink()
assert not job.exists(), 'job directory exists; inspect previous result, do not overwrite'
assert shutil.disk_usage(root).free >= 3 * 1024**3, 'insufficient free disk'
job.mkdir()


def write_state(state, **kw):
    p = job / 'status.json'
    tmp = job / 'status.tmp'
    tmp.write_text(json.dumps({'state': state, 'time_unix': time.time(), **kw}, indent=2) + '\n')
    os.replace(tmp, p)


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


try:
    write_state('COPYING')
    for p in raw.rglob('*'):
        if p.is_symlink() or not (p.is_dir() or p.is_file()):
            raise RuntimeError('unsupported raw input entry: ' + str(p.relative_to(raw)))
    shutil.copytree(raw, copy, symlinks=False, copy_function=shutil.copy2)
    manifest = []
    for p in sorted(raw.rglob('*')):
        if not p.is_file():
            continue
        q = copy / p.relative_to(raw)
        a = p.stat()
        src_hash, copy_hash = sha(p), sha(q)
        b = p.stat()
        if (a.st_ino, a.st_size, a.st_mtime_ns) != (b.st_ino, b.st_size, b.st_mtime_ns) or src_hash != copy_hash:
            raise RuntimeError('raw changed or copy mismatch: ' + str(p.relative_to(raw)))
        manifest.append({'relative_path': str(p.relative_to(raw)), 'bytes': a.st_size, 'sha256': src_hash})
    (job / 'input-copy-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    write_state('PARSING_TEXT', input_files=len(manifest), input_bytes=sum(x['bytes'] for x in manifest))
    temp = job / 'tmp'
    temp.mkdir()
    os.environ['TMPDIR'] = str(temp)
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    import torch
    torch.set_num_threads(1)
    from torch_npu.profiler.profiler import analyse
    analyse(str(copy), max_process_number=1, export_type='text')
    dbs = sorted(p for p in copy.rglob('*.db') if p.is_file() and not p.is_symlink())
    summary = []
    for p in dbs:
        c = sqlite3.connect(p.resolve().as_uri() + '?mode=ro', uri=True)
        try:
            c.execute('PRAGMA query_only=ON')
            tables = [x[0] for x in c.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
            wanted = {}
            for name in ('PYTORCH_API', 'CONNECTION_IDS', 'CANN_API', 'TASK', 'STRING_IDS'):
                if name in tables:
                    wanted[name] = {'count': c.execute('SELECT COUNT(*) FROM "' + name + '"').fetchone()[0],
                                    'columns': [r[1] for r in c.execute('PRAGMA table_info("' + name + '")')]}
            summary.append({'path': str(p.relative_to(job)), 'bytes': p.stat().st_size,
                            'sha256': sha(p), 'tables': tables, 'wanted': wanted})
        finally:
            c.close()
    write_state('ANALYSE_RETURNED', dbs=summary, note='Return alone does not prove complete export; inspect required tables and logs.')
    print(json.dumps({'state': 'ANALYSE_RETURNED', 'databases': len(summary), 'dbs': summary}), flush=True)
except BaseException as exc:
    write_state('FAILED', error_type=type(exc).__name__, error=str(exc)[:1000])
    raise
