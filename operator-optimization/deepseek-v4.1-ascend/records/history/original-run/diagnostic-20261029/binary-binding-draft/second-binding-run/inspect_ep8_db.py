#!/usr/bin/env python3
"""Bounded read-only examination of EP8 offline DB and exported log coverage."""
from collections import Counter
import json
from pathlib import Path
import shutil
import sqlite3

root = Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029')
job = root / 'binary-binding-draft/second-binding-run/offline-ep8-20261029'
status = json.loads((job / 'status.json').read_text())
assert status['state'] == 'ANALYSE_RETURNED'
paths = [x for x in status['dbs'] if x['path'].endswith('/ascend_pytorch_profiler_8.db')]
assert len(paths) == 1
db = job / paths[0]['path']
assert not Path(str(db) + '-wal').exists(), 'DB has WAL; immutable read would discard pending data'
assert not Path(str(db) + '-journal').exists(), 'DB has rollback journal; stop and investigate'
c = sqlite3.connect(db.resolve().as_uri() + '?mode=ro&immutable=1', uri=True)
c.execute('PRAGMA query_only=ON')
c.row_factory = sqlite3.Row
integrity = [x[0] for x in c.execute('PRAGMA integrity_check')]
values = {}
for pattern in ('%wio_diag:%', '%enqueue%', '%dequeue%', '%aclrtLaunchKernel%', '%DispatchFFNCombine%', '%dispatch_ffn_combine%'):
    rows = c.execute('''SELECT a.type, COUNT(*) n FROM PYTORCH_API a JOIN STRING_IDS s ON s.id=a.name
                        WHERE s.value LIKE ? GROUP BY a.type ORDER BY a.type''', (pattern,)).fetchall()
    values[pattern] = [dict(row) for row in rows]
names = c.execute('''SELECT value FROM STRING_IDS WHERE value LIKE '%enqueue%' OR value LIKE '%dequeue%'
                      OR value LIKE 'wio_diag:%' LIMIT 12''').fetchall()
times = {}
for table in ('PYTORCH_API', 'CANN_API', 'TASK'):
    times[table] = dict(c.execute('SELECT MIN(CAST(startNs AS INTEGER)) AS min_ns, MAX(CAST(endNs AS INTEGER)) AS max_ns FROM ' + table).fetchone())
ep8_launches = []
correlation = json.loads((job.parent / 'host-scope-correlation.json').read_text())
for event in correlation['matches']:
    if event['c_pid'] == 746 and event['type'] == 'launch':
        ep8_launches.append(event)
ids = sorted({(e['c_pid'] << 32) | e['c_tid'] for e in ep8_launches})
threads = []
for tid in ids:
    r = c.execute('SELECT COUNT(*) n, MIN(CAST(startNs AS INTEGER)) min_ns, MAX(CAST(endNs AS INTEGER)) max_ns FROM CANN_API WHERE globalTid=?', (tid,)).fetchone()
    threads.append({'encoded_tid': tid, **dict(r)})
log_errors = []
for p in job.rglob('*'):
    if not p.is_file() or p.is_symlink() or p.stat().st_size > 2 * 1024 * 1024:
        continue
    if p.suffix.lower() not in ('.log', '.txt'):
        continue
    try:
        for i, line in enumerate(p.read_text(errors='replace').splitlines(), 1):
            if 'disk I/O error' in line or 'sqlite error' in line.lower():
                log_errors.append({'file': str(p.relative_to(job)), 'line': i, 'message': line[:500]})
                if len(log_errors) >= 30:
                    break
    except OSError:
        pass
    if len(log_errors) >= 30:
        break
c.close()
usage = shutil.disk_usage(job)
result = {'integrity_check': integrity, 'sqlite_mode': 'read-only immutable snapshot',
          'queue_and_marker_counts': values, 'sample_names': [row[0] for row in names if not str(row[0]).startswith('wio_diag:')],
          'exported_time_ranges': times, 'c_launches_ep8': len(ep8_launches),
          'c_launch_time_range': [min(e['c_begin_ns'] for e in ep8_launches), max(e['c_end_ns'] for e in ep8_launches)] if ep8_launches else None,
          'candidate_encoded_threads': threads, 'disk': {'total': usage.total, 'free': usage.free},
          'tmp_exists': (job / 'tmp').is_dir(), 'log_error_hits': log_errors,
          'limits': ['Integrity check does not establish complete export, provenance, or timebase comparability.',
                     'Profiler logged SQLite disk I/O error; missing rows cannot be interpreted as absence.',
                     'PID<<32|TID is a candidate encoding pending source/version confirmation.']}
(job / 'db-coverage-inspection.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result))
