#!/usr/bin/env python3
"""Read one frozen EP8 CSV and profiler DB; never initialize NPU or print tensors."""
import csv
import hashlib
import json
import pathlib
import sqlite3
from decimal import Decimal

root = pathlib.Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-20261028/diagnostic/runtime/diagnostic-traces/dp1_pp0_tp0_dcp0_ep8_rank0_727_20261028020238580_ascend_pt/ASCEND_PROFILER_OUTPUT')
csv_path, db_path = root / 'kernel_details.csv', root / 'ascend_pytorch_profiler_8.db'

def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

csv_sha, db_sha = sha(csv_path), sha(db_path)
assert csv_sha == '488d3e3736677d7d041edbe8307d3c06b79553362e54aef07aba37fd74229cc7'
assert db_sha == '67d14f7f386658f8c8c5d3c9059dff9b0f1decb200eba621a3432fcf1e23de58'
start_ns, end_ns = 1793152960209278701, 1793152960209304681
fields = ['Name', 'Type', 'Model ID', 'Task ID', 'Stream ID', 'Start Time(us)', 'Duration(us)',
          'Input Shapes', 'Input Data Types', 'Input Formats', 'Output Shapes', 'Output Data Types',
          'Output Formats', 'aicore_time(us)', 'aic_total_cycles', 'aic_mac_ratio', 'aic_mte2_ratio',
          'aic_scalar_ratio', 'cube_utilization(%)']
with csv_path.open(newline='') as f:
    reader = csv.DictReader(f)
    assert set(fields).issubset(reader.fieldnames)
    matches = [row for row in reader if row['Model ID'] == '49' and row['Stream ID'] == '89'
               and row['Task ID'] == '1300' and Decimal(row['Start Time(us)']) * 1000 == start_ns
               and Decimal(row['Duration(us)']) * 1000 == end_ns - start_ns]
assert len(matches) == 1
row = {key: matches[0][key].strip() for key in fields}
assert row['Input Shapes'] == '"3,5120;144,320,16,32;4608"'
assert row['Output Shapes'] == '"3,4608"' and row['Output Data Types'] == 'INT32'
db = sqlite3.connect(db_path.as_uri() + '?mode=ro&immutable=1', uri=True)
db.row_factory = sqlite3.Row
db.execute('PRAGMA query_only=ON')
tasks = [dict(x) for x in db.execute('SELECT globalTaskId,streamId,taskId,modelId,startNs,endNs,connectionId FROM TASK WHERE streamId=? AND modelId=? AND taskId=? AND CAST(startNs AS INTEGER)=? AND CAST(endNs AS INTEGER)=? LIMIT 3', (89, 49, 1300, start_ns, end_ns))]
assert len(tasks) == 1
task = tasks[0]
cid = task['connectionId']
groups = [x[0] for x in db.execute('SELECT id FROM CONNECTION_IDS WHERE connectionId=? LIMIT 21', (cid,))]
out = {'scope': 'EP8 layer24 run2, one exact recorded task; not all 1920 tasks',
       'csv_sha256': csv_sha, 'db_sha256': db_sha, 'csv': row, 'task': task,
       'global_task_id_reuse_count': db.execute('SELECT count(*) FROM TASK WHERE globalTaskId=?', (task['globalTaskId'],)).fetchone()[0],
       'cann_api_count_with_connection_id': db.execute('SELECT count(*) FROM CANN_API WHERE connectionId=?', (cid,)).fetchone()[0],
       'connection_group_count_capped_at_21': len(groups),
       'pytorch_api_count_by_connection_group': [db.execute('SELECT count(*) FROM PYTORCH_API WHERE connectionId=?', (group,)).fetchone()[0] for group in groups],
       'limits': ['Reported pipeline ratios are not total device utilization, idle time, memory bandwidth saturation or causal attribution.',
                  'Connection-id matches and absence are restricted to these queried tables; do not prove or exclude host binding via other mechanisms.',
                  'Task identity does not establish tensor identity, runtime quantization method, NZ mapping, valid row count or actual arithmetic.',
                  'Profile measurement is not the unprofiled service baseline; no NPU workload was started.']}
db.close()
print(json.dumps(out, ensure_ascii=False, indent=2))
