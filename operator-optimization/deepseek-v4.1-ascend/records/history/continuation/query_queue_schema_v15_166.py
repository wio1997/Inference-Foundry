from pathlib import Path
from collections import Counter
from datetime import datetime,timezone
import json,hashlib,sqlite3,socket
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
P=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/hostprof-20261030T212812Z-360e0c72/analysis/native-execution-identity-v14')
def pin(path,sha):
 b=path.read_bytes();assert hashlib.sha256(b).hexdigest()==sha;return json.loads(b)
idx=pin(P/'index.json','a3aa47cdf9170785432ecbc18117a489f3fba37d89959f110f4fcbd8890c7dc4')
q=pin(P/'queue-bridge-v15.json','654f27199041bb92e29188189b041be87965ec6e8127902cd73c75a3c98a4eeb')
OUT=P/'queue-schema-and-ID-relations-v15.json';assert not OUT.exists();files=[]
for di in sorted({r['database_index']for r in q['records']}):
 ref=idx['databases'][di];d=pin(Path(ref['path']),ref['sha256']);db=sqlite3.connect(Path(d['database_path']).as_uri()+'?mode=ro&immutable=1',uri=True);db.row_factory=sqlite3.Row;db.execute('PRAGMA query_only=ON')
 tables=[r[0]for r in db.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
 named=[t for t in tables if any(s in t.upper()for s in ['QUEUE','FLOW','API_TYPE','PYTORCH','CONNECTION'])]
 schemas={t:[dict(r)for r in db.execute('PRAGMA table_info('+t+')')]for t in named}
 counts=[dict(r)for r in db.execute("SELECT p.type,s.value AS name,COUNT(*)AS records FROM PYTORCH_API p JOIN STRING_IDS s ON s.id=p.name WHERE lower(s.value)LIKE '%enqueue%' OR lower(s.value)LIKE '%dequeue%' GROUP BY p.type,p.name")]
 enums={t:[dict(r)for r in db.execute('SELECT * FROM '+t)]for t in tables if t.startswith('ENUM_')and ('API'in t or 'QUEUE'in t)}
 records=[]
 for row in [r for r in q['records']if r['database_index']==di]:
  deq=row['backend_Dequeue_scopes'][0];key=deq['connectionId'];links=[dict(r)for r in db.execute('SELECT rowid AS connection_rowid,* FROM CONNECTION_IDS WHERE id=? OR connectionId=?',(key,key))]
  values={r['connectionId']for r in links if r['id']==key};otherlinks=[]
  if values:
   vals=sorted(values);otherlinks=[dict(r)for r in db.execute('SELECT rowid AS connection_rowid,* FROM CONNECTION_IDS WHERE connectionId IN ('+','.join('?'for _ in vals)+')',vals)]
  pids={r['id']for r in otherlinks};apis=[]
  if pids:
   vals=sorted(pids);apis=[dict(r)for r in db.execute('SELECT p.rowid AS api_rowid,p.*,s.value AS name_text FROM PYTORCH_API p LEFT JOIN STRING_IDS s ON s.id=p.name WHERE p.connectionId IN ('+','.join('?'for _ in vals)+')',vals)]
  records.append(dict(task_rowid=row['TASK']['task_record_rowid'],op=row['definition']['opType_text'],backend_Dequeue=deq,CONNECTION_IDS_matching_either_column=links,CONNECTION_IDS_shared_values=otherlinks,PYTORCH_same_connection_list_candidates=apis))
 files.append(dict(database_index=di,all_table_names=tables,queue_flow_schemas=schemas,queue_API_type_name_counts=counts,enums=enums,records=records));db.close()
summary=dict(created_utc=datetime.now(timezone.utc).isoformat(),databases=len(files),tasks=sum(len(f['records'])for f in files),queue_table_names=sorted({t for f in files for t in f['queue_flow_schemas']}),tasks_with_CONNECTION_IDS_rows=sum(bool(r['CONNECTION_IDS_matching_either_column'])for f in files for r in f['records']),tasks_with_shared_value_API_candidates=sum(bool(r['PYTORCH_same_connection_list_candidates'])for f in files for r in f['records']),no_target_source_or_device_work=True,overall_complete=False,limits=['Column membership alternatives are diagnostic until actual API type/export schema semantics are reviewed. No mixed numeric namespaces or nearesttime promotion.'])
payload=(json.dumps(dict(summary=summary,databases=files),ensure_ascii=False,separators=(',',':'))+'\n').encode();OUT.write_bytes(payload)
public=dict(summary=summary,first_database=files[0],record_examples=[r for f in files for r in f['records']][:3],full_result=str(OUT),result_sha256=hashlib.sha256(payload).hexdigest())
if len(json.dumps(public).encode())>30000:public.pop('first_database');public['display_truncated']=True
print(json.dumps(public,ensure_ascii=False))
