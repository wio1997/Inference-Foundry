"""Exported resource taxonomy of existing profiles, separate from utilization."""
import collections
import concurrent.futures
import hashlib
import json
import os
import sqlite3
import time
from pathlib import Path

A = Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/runtime/analysis')
P = A / 'joint168'
O = A / 'joint170'
state = {'state': 'RUNNING', 'pid': os.getpid(), 'start_ticks': Path('/proc/'+str(os.getpid())+'/stat').read_text().rsplit(')', 1)[1].split()[19], 'started_unix': time.time(), 'completed': 0, 'CPU_only': True, 'max_workers': 2}


def status():
    (O / 'RESOURCE-STATUS.json').write_text(json.dumps(state, indent=2)+'\n')


def one(item):
    phase, db = item
    c = sqlite3.connect('file:'+str(db)+'?mode=ro&immutable=1', uri=True)
    c.row_factory = sqlite3.Row
    s = dict(c.execute('select id,value from STRING_IDS'))
    info = {r['globalTaskId']: (s[r['name']], s[r['taskType']]) for r in c.execute('select globalTaskId,name,taskType from COMPUTE_TASK_INFO')}
    by = collections.defaultdict(lambda: {'executions': 0, 'cumulative_task_ns': 0})
    total = unmatched = 0
    native_ids = set()
    for t in c.execute('select rowid,globalTaskId,startNs,endNs,taskType from TASK'):
        total += 1
        x = info.get(t['globalTaskId'])
        if x is None:
            unmatched += 1
            continue
        name, exported = x
        key = (name, exported, s[t['taskType']])
        r = by[key]
        r['executions'] += 1
        r['cumulative_task_ns'] += int(t['endNs'])-int(t['startNs'])
        native_ids.add(t['rowid'])
    c.close()
    return {'phase': phase, 'profile': db.parent.parent.name, 'database': str(db), 'TASK_rows': total, 'TASK_rows_with_COMPUTE_info': len(native_ids), 'TASK_rows_without_COMPUTE_info': unmatched, 'operator_resources': [{'op': key[0], 'exported_compute_resource': key[1], 'native_task_type': key[2], **value} for key, value in sorted(by.items())]}


status()
items = [(phase, db) for phase in ['startup', 'runtime'] for db in sorted((P/phase).glob('*_ascend_pt/ASCEND_PROFILER_OUTPUT/ascend_pytorch_profiler*.db'))]
assert len(items) == 32
rows = []
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
    for row in pool.map(one, items):
        rows.append(row)
        state['completed'] = len(rows)
        status()
aggregate = collections.defaultdict(lambda: {'executions': 0, 'cumulative_task_ns': 0, 'profiles': 0})
for row in rows:
    for r in row['operator_resources']:
        key = (row['phase'], r['op'], r['exported_compute_resource'], r['native_task_type'])
        a = aggregate[key]
        a['executions'] += r['executions']
        a['cumulative_task_ns'] += r['cumulative_task_ns']
        a['profiles'] += 1
result = {'status': 'EXPORTED_OPERATOR_RESOURCE_TAXONOMY_RETURNED', 'profiles': rows, 'aggregate': [{'phase': k[0], 'op': k[1], 'exported_compute_resource': k[2], 'native_task_type': k[3], **v} for k, v in sorted(aggregate.items())], 'limits': ['This inventory classifies official exported task/resource labels;it does not prove instruction-mode throughput, physical utilization or exclusive occupancy.', 'AICPU,AI Core,Vector,mixed kernels and communications are not one common MAC-capacity pool.', 'TASK durations may include waiting; sums across profiles/ranks/overlapping resources are not request latency.', 'TASK rows without COMPUTE metadata are inventoried,not declared zero-work or assigned arbitrary arithmetic.'], 'no_new_device_work': True, 'whole_modules_closed': 0, 'overall_complete': False}
f = O / 'EXPORTED-RESOURCE-TAXONOMY.json'
f.write_text(json.dumps(result, indent=2)+'\n')
state.update(state='RETURNED', elapsed_seconds=time.time()-state['started_unix'], result=str(f), sha256=hashlib.sha256(f.read_bytes()).hexdigest())
status()
print(json.dumps({k: v for k, v in state.items()}))
