from pathlib import Path
from datetime import datetime,timezone
import socket,sqlite3,json,hashlib
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
DB=Path('/data/tiankuan/wio/migration-166-20261030/source169-project/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029/runtime/diagnostic-traces/dp0_pp0_tp3_dcp0_ep3_rank3_2128_20261029023525650_ascend_pt/ASCEND_PROFILER_OUTPUT/ascend_pytorch_profiler_3.db')
SHA='084b8718888e33271435ec79d7573c95ca1615f3e7484c807ea24590fad234c4';before=DB.stat()
h=hashlib.sha256()
with DB.open('rb') as f:
 for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
assert h.hexdigest()==SHA
db=sqlite3.connect(DB.as_uri()+'?mode=ro&immutable=1',uri=True);db.row_factory=sqlite3.Row;db.execute('PRAGMA query_only=ON')
def rows(sql,args=()):return [dict(r) for r in db.execute(sql,args)]
duplicates=rows('''SELECT t.globalTaskId,COUNT(*) AS join_rows,COUNT(DISTINCT t.rowid) AS task_records,COUNT(DISTINCT c.rowid) AS compute_records
 FROM TASK t JOIN COMPUTE_TASK_INFO c ON c.globalTaskId=t.globalTaskId WHERE t.deviceId=3 GROUP BY t.globalTaskId HAVING COUNT(*)>1 ORDER BY join_rows DESC,t.globalTaskId LIMIT 8''')
for r in duplicates:
 r['task_examples']=rows('SELECT t.rowid AS task_record_rowid,t.*,s.value AS task_type_name FROM TASK t LEFT JOIN STRING_IDS s ON s.id=t.taskType WHERE t.deviceId=3 AND t.globalTaskId=? ORDER BY t.startNs LIMIT 8',(r['globalTaskId'],))
 r['compute_examples']=rows('SELECT c.rowid AS compute_record_rowid,c.*,n.value AS kernel_name,o.value AS op_type FROM COMPUTE_TASK_INFO c LEFT JOIN STRING_IDS n ON n.id=c.name LEFT JOIN STRING_IDS o ON o.id=c.opType WHERE c.globalTaskId=? LIMIT 8',(r['globalTaskId'],))
summary=rows('''SELECT COUNT(*) AS join_rows,COUNT(DISTINCT t.rowid) AS task_records,COUNT(DISTINCT c.rowid) AS compute_records,COUNT(DISTINCT t.globalTaskId) AS global_ids
 FROM TASK t JOIN COMPUTE_TASK_INFO c ON c.globalTaskId=t.globalTaskId WHERE t.deviceId=3''')[0]
schemas={table:rows('PRAGMA table_info('+table+')') for table in ['TASK','COMPUTE_TASK_INFO','PYTORCH_API','CONNECTION_IDS']}
db.close();after=DB.stat();assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
result=dict(created_utc=datetime.now(timezone.utc).isoformat(),database_sha256=SHA,database_stat_unchanged=True,counts=summary,duplicate_examples=duplicates,schemas=schemas,no_target_source_read=True,no_device_or_inference_work=True,overall_complete=False)
out=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/profile-readonly-20261031/ep3-task-record-identity-v9.json');assert not out.exists();payload=json.dumps(result,ensure_ascii=False,indent=2)+'\n';out.write_text(payload)
print(json.dumps(dict(result=result,full_result=str(out),result_sha256=hashlib.sha256(payload.encode()).hexdigest()),ensure_ascii=False))
