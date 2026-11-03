from pathlib import Path
from collections import Counter
from datetime import datetime,timezone
import hashlib,json,sqlite3,socket
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
P=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/hostprof-20261030T212812Z-360e0c72/analysis/native-execution-identity-v14')
b=(P/'index.json').read_bytes();assert hashlib.sha256(b).hexdigest()=='a3aa47cdf9170785432ecbc18117a489f3fba37d89959f110f4fcbd8890c7dc4';x=json.loads(b)
OUT=P/'unassigned-compute-native-context-v14.json';assert not OUT.exists();rows=[]
for ref in x['databases']:
 b=Path(ref['path']).read_bytes();assert hashlib.sha256(b).hexdigest()==ref['sha256'];d=json.loads(b)
 defs={c['globalTaskId']:c for c in d['COMPUTE_definitions']};graph={i for g in d['graph_API_control_bindings']if g['control_pair_valid']for i in g['compute_task_rowids']}
 missing=[t for t in d['TASK_records']if t['globalTaskId']in defs and not t.get('native_linked_API_rowids')and t['task_record_rowid']not in graph]
 if not missing:continue
 db=sqlite3.connect(Path(d['database_path']).as_uri()+'?mode=ro&immutable=1',uri=True);db.row_factory=sqlite3.Row;db.execute('PRAGMA query_only=ON')
 for t in missing:
  nativeids={t['connectionId']}|{r['id']for r in d['CONNECTION_IDS']if r['connectionId']==t['connectionId']}
  actual=[i for i in nativeids if i is not None];apis=[]
  if actual:
   apis=[dict(r)for r in db.execute('SELECT a.rowid AS cann_api_rowid,a.*,s.value AS name_text FROM CANN_API a LEFT JOIN STRING_IDS s ON s.id=a.name WHERE a.connectionId IN ('+','.join('?'for i in actual)+')',actual)]
  scopes=[]
  for a in apis:
   ps=[dict(r)for r in db.execute('SELECT p.rowid AS api_rowid,CAST(p.startNs AS INTEGER)AS startNs,CAST(p.endNs AS INTEGER)AS endNs,p.globalTid,p.connectionId,p.callchainId,s.value AS name FROM PYTORCH_API p LEFT JOIN STRING_IDS s ON s.id=p.name WHERE p.globalTid=? AND CAST(p.startNs AS INTEGER)<=? AND CAST(p.endNs AS INTEGER)>=?',(a['globalTid'],a['startNs'],a['endNs']))]
   scopes.append(dict(cann_api_rowid=a['cann_api_rowid'],same_thread_enclosing_PYTORCH_API_records=ps))
  rows.append(dict(database_index=d['database_index'],task=t,definition=defs[t['globalTaskId']],native_connection_candidates=sorted(actual),native_CANN_APIs=apis,host_scope_candidates=scopes))
 db.close()
summary=dict(created_utc=datetime.now(timezone.utc).isoformat(),unassigned_compute_records=len(rows),op_type_counts=dict(Counter(r['definition']['opType_text']for r in rows)),native_CANN_API_candidate_count_hist=dict(Counter(len(r['native_CANN_APIs'])for r in rows)),host_scope_candidate_count_hist=dict(Counter(sum(len(s['same_thread_enclosing_PYTORCH_API_records'])for s in r['host_scope_candidates'])for r in rows)),no_target_source_or_device_work=True,overall_complete=False,limits=['Candidate native/API scopes are retained as stored; do not promote same-thread containment to direct native PYTORCH connection. No nearesttime or missing-source fabrication.'])
payload=(json.dumps(dict(summary=summary,records=rows),ensure_ascii=False,separators=(',',':'))+'\n').encode();OUT.write_bytes(payload)
print(json.dumps(dict(summary=summary,records=rows,full_result=str(OUT),result_sha256=hashlib.sha256(payload).hexdigest()),ensure_ascii=False))
