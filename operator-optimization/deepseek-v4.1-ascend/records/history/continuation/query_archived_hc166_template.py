from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import sqlite3, socket, hashlib, json

assert socket.gethostname() == 'bm-jn-zs-zone1-910C-64G-10-105'
DB = Path('/data/tiankuan/wio/migration-166-20261030/source169-project/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029/runtime/diagnostic-traces/dp0_pp0_tp3_dcp0_ep3_rank3_2128_20261029023525650_ascend_pt/ASCEND_PROFILER_OUTPUT/ascend_pytorch_profiler_3.db')
SHA = '084b8718888e33271435ec79d7573c95ca1615f3e7484c807ea24590fad234c4'
before = DB.stat()
h = hashlib.sha256()
with DB.open('rb') as f:
    for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
        h.update(block)
assert h.hexdigest() == SHA
db = sqlite3.connect(DB.as_uri() + '?mode=ro&immutable=1', uri=True)
db.row_factory = sqlite3.Row
db.execute('PRAGMA query_only=ON')
SELECT = '''SELECT t.startNs,t.endNs,t.deviceId,t.streamId,t.taskId,t.globalTaskId,t.globalPid,t.connectionId,t.modelId,
    ty.value AS task_type_name,n.value AS kernel_name,op.value AS op_type,
    ish.value AS input_shapes,osh.value AS output_shapes,idt.value AS input_dtypes,odt.value AS output_dtypes
    FROM TASK t LEFT JOIN STRING_IDS ty ON ty.id=t.taskType
    LEFT JOIN COMPUTE_TASK_INFO c ON c.globalTaskId=t.globalTaskId
    LEFT JOIN STRING_IDS n ON n.id=c.name LEFT JOIN STRING_IDS op ON op.id=c.opType
    LEFT JOIN STRING_IDS ish ON ish.id=c.inputShapes LEFT JOIN STRING_IDS osh ON osh.id=c.outputShapes
    LEFT JOIN STRING_IDS idt ON idt.id=c.inputDataTypes LEFT JOIN STRING_IDS odt ON odt.id=c.outputDataTypes'''


items = []
for spec in CONFIG:
    lo, hi = spec['interval_ns']
    anchor_rows = []
    for a in spec['anchors']:
        vals = [dict(r) for r in db.execute(SELECT + ' WHERE t.deviceId=3 AND t.streamId=? AND t.taskId=? AND t.startNs=? AND t.endNs=?', (int(a['stream']), int(a['task_id']), a['start_ns'], a['end_ns']))]
        named = [x for x in vals if x['kernel_name']]
        assert len(named) == 1, (spec['label'], 'anchor_not_unique', len(named))
        anchor_rows.extend(named)
    records = [dict(r) for r in db.execute(SELECT + ' WHERE t.deviceId=3 AND t.startNs<? AND t.endNs>? ORDER BY t.startNs,t.streamId LIMIT 5001', (hi, lo))]
    assert len(records) <= 5000, ('record_budget_exceeded', spec['label'])
    comm = [dict(x) for x in db.execute('SELECT c.*,n.value AS operation_name,g.value AS group_name FROM COMMUNICATION_OP c LEFT JOIN STRING_IDS n ON n.id=c.opName LEFT JOIN STRING_IDS g ON g.id=c.groupName WHERE c.deviceId=3 AND c.startNs<? AND c.endNs>? ORDER BY c.startNs', (hi, lo))]
    assert len(comm) < 100
    items.append(dict(**spec, anchor_rows=anchor_rows, task_rows=records, communication_ops=comm,
                      record_count=len(records), compute_records=sum(bool(r['kernel_name']) for r in records),
                      op_type_counts=dict(Counter(r['op_type'] or r['task_type_name'] or 'UNRESOLVED' for r in records))))
# One bounded batch per table rather than repeating unindexed scans per task.
all_named = [r for x in items for r in x['anchor_rows'] + x['task_rows'] if r['kernel_name']]
connections = sorted({r['connectionId'] for r in all_named})
assert 0 < len(connections) < 900
api_rows = [dict(r) for r in db.execute('''SELECT ids.connectionId AS device_connection,p.startNs,p.endNs,p.globalTid,p.connectionId,p.callchainId,n.value AS name
    FROM CONNECTION_IDS ids JOIN PYTORCH_API p ON p.connectionId=ids.id
    LEFT JOIN STRING_IDS n ON n.id=p.name WHERE ids.connectionId IN (''' + ','.join('?'*len(connections)) + ') ORDER BY p.startNs', connections)]
assert len(api_rows) < 1000
chain_ids = sorted({r['callchainId'] for r in api_rows if r['callchainId'] is not None})
frames = {}
if chain_ids:
    for r in db.execute('SELECT c.id,c.stackDepth,s.value AS frame FROM PYTORCH_CALLCHAINS c JOIN STRING_IDS s ON s.id=c.stack WHERE c.id IN (' + ','.join('?'*len(chain_ids)) + ') ORDER BY c.id,c.stackDepth', chain_ids):
        frames.setdefault(r['id'], []).append(dict(stackDepth=r['stackDepth'], frame=r['frame']))
apis_by_connection = {}
for a in api_rows:
    a['frames'] = frames.get(a['callchainId'], [])
    apis_by_connection.setdefault(a['device_connection'], []).append(a)
for r in all_named:
    r['api_candidates'] = apis_by_connection.get(r['connectionId'], [])
    assert len(r['api_candidates']) < 30
db.close()
after = DB.stat()
assert before.st_size == after.st_size and before.st_mtime_ns == after.st_mtime_ns
result = dict(created_utc=datetime.now(timezone.utc).isoformat(), database=str(DB), database_sha256=SHA,
              database_stat_unchanged=True, items=items, no_device_or_inference_work=True,
              limits=['Only selected EP3 archived intervals; not all rank/branch/request coverage.',
                      'Exact TASK/API/callchain binding identifies recorded submission location, not tensor identity, alias generation or loaded source SHA.',
                      'Leading/tail time intervals are bounded adjacency, not guaranteed complete model-init/tail attribution.',
                      'TASK/COMM records may overlap or represent waits/summary activity; not exclusive device busy/idle or critical-path time.',
                      'No historical tensor values, actual slot/mix contents or event object identity recovered.'])
out = Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/profile-readonly-20261031/ep3-hc-boundary-task-stack-evidence-v2.json')
assert not out.exists(), 'refuse_overwrite'
out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')


def short(row):
    outrow = {k: row[k] for k in ['startNs', 'endNs', 'streamId', 'taskId', 'globalTaskId', 'op_type', 'kernel_name', 'input_shapes', 'output_shapes']}
    outrow['api_candidates'] = []
    for a in row.get('api_candidates', []):
        relevant = [f for f in a['frames'] if any(s in f['frame'] for s in ['model.py(', 'dspark', 'engram', 'hc_', 'layernorm', 'vocab_parallel', 'logits_processor'])]
        outrow['api_candidates'].append(dict(name=a['name'], callchainId=a['callchainId'], frames=relevant[:12], relevant_frame_count=len(relevant)))
    return outrow


public = []
for x in items:
    # All records are retained server-side. Display task-ID/span, complete type counts and up to 16 compute rows per interval.
    compute = [r for r in x['task_rows'] if r['kernel_name']]
    public.append(dict(label=x['label'], interval_ns=x['interval_ns'], record_count=x['record_count'],
                       compute_records=x['compute_records'], op_type_counts=x['op_type_counts'],
                       anchor_rows=[short(r) for r in x['anchor_rows']],
                       compute_display=[short(r) for r in compute[:16]], compute_display_truncated=len(compute)>16,
                       communication_count=len(x['communication_ops']),
                       communication_ops=[{k:r[k] for k in ['opId','connectionId','startNs','endNs','rankSize','count','dataType','operation_name']} for r in x['communication_ops']]))
print(json.dumps(dict(created_utc=result['created_utc'], database_sha256=SHA, database_stat_unchanged=True,
                      full_result=str(out), items=public, no_device_or_inference_work=True, limits=result['limits']), ensure_ascii=False))
