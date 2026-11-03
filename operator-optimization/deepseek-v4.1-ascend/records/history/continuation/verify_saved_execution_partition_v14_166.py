from pathlib import Path
from collections import Counter
from datetime import datetime,timezone
import hashlib,json,socket
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
P=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/hostprof-20261030T212812Z-360e0c72/analysis/native-execution-identity-v14')
b=(P/'index.json').read_bytes();assert hashlib.sha256(b).hexdigest()=='a3aa47cdf9170785432ecbc18117a489f3fba37d89959f110f4fcbd8890c7dc4';x=json.loads(b)
OUT=P/'execution-partition-validation-v14.json';assert not OUT.exists();rows=[]
for ref in x['databases']:
 b=Path(ref['path']).read_bytes();assert hashlib.sha256(b).hexdigest()==ref['sha256'];d=json.loads(b)
 defs={c['globalTaskId']for c in d['COMPUTE_definitions']};compute={t['task_record_rowid']for t in d['TASK_records']if t['globalTaskId']in defs}
 native={t['task_record_rowid']for t in d['TASK_records']if t.get('native_linked_API_rowids')};graph={i for g in d['graph_API_control_bindings']if g['control_pair_valid']for i in g['compute_task_rowids']}
 assert native<=compute and graph<=compute
 rows.append(dict(database_index=d['database_index'],compute=len(compute),native_only=len(native-graph),graph_only=len(graph-native),both=len(native&graph),unassigned=len(compute-native-graph)))
summary=dict(created_utc=datetime.now(timezone.utc).isoformat(),databases=len(rows),totals={k:sum(r[k]for r in rows)for k in ['compute','native_only','graph_only','both','unassigned']},no_source_DB_or_device_work=True,overall_complete=False)
assert summary['totals']['compute']==sum(summary['totals'][k]for k in ['native_only','graph_only','both','unassigned'])
payload=(json.dumps(dict(summary=summary,databases=rows),separators=(',',':'))+'\n').encode();OUT.write_bytes(payload)
print(json.dumps(dict(summary=summary,full_result=str(OUT),result_sha256=hashlib.sha256(payload).hexdigest())))
