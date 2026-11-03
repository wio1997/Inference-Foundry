from pathlib import Path
from collections import Counter
from datetime import datetime,timezone
import json,hashlib,sqlite3,socket,time
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
P=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/hostprof-20261030T212812Z-360e0c72')
OUT=P/'analysis/native-execution-identity-v14/saved-CPU-process-surface-coverage-v17.json';assert not OUT.exists()
def pin(path,sha):
 b=path.read_bytes();assert hashlib.sha256(b).hexdigest()==sha;return json.loads(b)
def ro(path):
 db=sqlite3.connect(path.as_uri()+'?mode=ro&immutable=1',uri=True);db.row_factory=sqlite3.Row;db.execute('PRAGMA query_only=ON');return db
indexpath=P/'analysis/offline-db-v12/database-index.json';ib=indexpath.read_bytes();idx=json.loads(ib)
helpers=[];workers=[];t0=time.monotonic()
for ref in idx['databases']:
 path=Path(ref['path'].replace('/work/vllm-ascend-reproduce-172-on-166-20261030','/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030',1))
 if path.name!='analysis.db':continue
 b=path.read_bytes();assert hashlib.sha256(b).hexdigest()==ref['sha256'];db=ro(path)
 tables=[r[0]for r in db.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
 rows=[]
 for name in tables:
  quoted='"'+name.replace('"','""')+'"';rows.append(dict(table=name,records=db.execute('SELECT COUNT(*) FROM '+quoted).fetchone()[0],schema=[dict(r)for r in db.execute('PRAGMA table_info('+quoted+')')]))
 helpers.append(dict(path=str(path),sha256=ref['sha256'],bytes=len(b),tables=rows));db.close()
caps=pin(P/'analysis/profile-capabilities-index-v12-v2.json','79ba556ef3855b9066392e438371a0a9d3b38ea7511269ce7ff2a6814529a095')
dbrefs=caps.get('databases',caps.get('database_files',[]))
assert isinstance(dbrefs,list)and len(dbrefs)==32,(list(caps),type(dbrefs),len(dbrefs))
for ref in dbrefs:
 path=Path(ref['path']);db=ro(path);db.set_progress_handler(lambda:int(time.monotonic()-t0>140),100000)
 tids=[int(r[0])for r in db.execute('SELECT DISTINCT globalTid FROM PYTORCH_API')if r[0]is not None]
 names=[dict(r)for r in db.execute('SELECT s.value AS name,COUNT(*) AS records FROM PYTORCH_API p JOIN STRING_IDS s ON s.id=p.name GROUP BY p.name')]
 workers.append(dict(path=str(path),profile_directory=path.parent.parent.name,PYTORCH_API_globalPid_counts=dict(Counter(t>>32 for t in tids)),distinct_globalTids=len(tids),API_names_matching_H01_H03=[r for r in names if any(v in r['name'].lower()for v in ['enginecore','hash_block','prefix_hash','wait_for_response','get_response'])]));db.close()
frontend=pin(P/'analysis/native-execution-identity-v14/frontend-CPU-trace-inventory-v14.json','4da035667b2a4dde966b884590ab8d8d0668bf408258ee172be043d715efab5f74cdcf518649792') if False else pin(P/'analysis/native-execution-identity-v14/frontend-CPU-trace-inventory-v14.json','4da035667b2a4dde966b884590ab8d8d0668bf408258ee172be04310329a09d3')
front=[]
for f in frontend['files']:
 records=f['complete_events'];front.append(dict(path=f['path'],raw_sha256=f['raw_sha256'],event_PID_counts=dict(Counter(str(r.get('pid'))for r in records)),thread_count=len({(r.get('pid'),r.get('tid'))for r in records}),categories=dict(Counter(r.get('cat')for r in records)),H01_H03_name_matches=[r for r in f['function_name_aggregates']if any(v in r['name'].lower()for v in ['enginecore','hash_block_tokens','get_block_hash','_wait_for_response'])]))
helper_tables=Counter(t['table']for d in helpers for t in d['tables'])
summary=dict(created_utc=datetime.now(timezone.utc).isoformat(),helper_databases=len(helpers),worker_API_databases=len(workers),frontend_files=len(front),helper_table_presence_counts=dict(helper_tables),helper_PYTORCH_API_tables=helper_tables.get('PYTORCH_API',0),worker_event_PIDs=sorted({int(pid)for r in workers for pid in r['PYTORCH_API_globalPid_counts']}),frontend_event_PIDs=sorted({pid for r in front for pid in r['event_PID_counts']}),no_model_source_import_or_device_work=True,overall_complete=False,limits=['HelperDB schema and paths determine their actual surface; database count is not independent-process coverage.','A worker-side API/callchain profiler does not provide EngineCore onCPU time merely because callers share a repository.','Name absence is not proof a function never ran. CPU wall trace is not onCPU attribution or producer-ready timing.'])
payload=(json.dumps(dict(summary=summary,database_index_source=dict(path=str(indexpath),sha256=hashlib.sha256(ib).hexdigest()),helper_databases=helpers,worker_API_processes=workers,frontend_processes=front),ensure_ascii=False,separators=(',',':'))+'\n').encode();OUT.write_bytes(payload)
print(json.dumps(dict(summary=summary,helper_first=helpers[:1],worker_first=workers[:1],frontend=front,full_result=str(OUT),result_sha256=hashlib.sha256(payload).hexdigest()),ensure_ascii=False))
