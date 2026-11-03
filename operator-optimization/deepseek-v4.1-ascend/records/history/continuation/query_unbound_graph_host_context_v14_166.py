from pathlib import Path
from collections import Counter
from datetime import datetime,timezone
import json,hashlib,sqlite3,socket,time
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
P=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/hostprof-20261030T212812Z-360e0c72')
ROOT=P/'analysis/native-execution-identity-v14';b=(ROOT/'index.json').read_bytes();assert hashlib.sha256(b).hexdigest()=='a3aa47cdf9170785432ecbc18117a489f3fba37d89959f110f4fcbd8890c7dc4';index=json.loads(b)
OUT=ROOT/'unbound-graph-enclosing-host-context-v14.json';assert not OUT.exists()
t0=time.monotonic();rows=[];canonical={}
for ref in index['databases']:
 b=Path(ref['path']).read_bytes();assert hashlib.sha256(b).hexdigest()==ref['sha256'];d=json.loads(b)
 gs=[(i,g)for i,g in enumerate(d['graph_API_control_bindings'])if not g['same_thread_markers']]
 if not gs:continue
 db=sqlite3.connect(Path(d['database_path']).as_uri()+'?mode=ro&immutable=1',uri=True);db.row_factory=sqlite3.Row;db.execute('PRAGMA query_only=ON');db.set_progress_handler(lambda:int(time.monotonic()-t0>140),100000)
 for gi,g in gs:
  a=g['CANN_API']
  scopes=[dict(r)for r in db.execute('''SELECT p.rowid AS api_rowid,CAST(p.startNs AS INTEGER)AS startNs,CAST(p.endNs AS INTEGER)AS endNs,p.globalTid,p.connectionId,p.callchainId,s.value AS name FROM PYTORCH_API p JOIN STRING_IDS s ON s.id=p.name WHERE p.globalTid=? AND CAST(p.startNs AS INTEGER)<=? AND CAST(p.endNs AS INTEGER)>=? ORDER BY CAST(p.startNs AS INTEGER),CAST(p.endNs AS INTEGER)DESC''',(a['globalTid'],a['startNs'],a['endNs']))]
  assert len(scopes)<500
  ids={s['callchainId']for s in scopes if s['callchainId']is not None};fs={}
  if ids:
   literal=','.join(str(int(i))for i in sorted(ids))
   for r in db.execute('SELECT c.id,c.stackDepth,s.value AS frame FROM PYTORCH_CALLCHAINS c JOIN STRING_IDS s ON s.id=c.stack WHERE c.id IN ('+literal+') ORDER BY c.id,c.stackDepth'):fs.setdefault(r['id'],[]).append(dict(stackDepth=r['stackDepth'],frame=r['frame']))
  for s in scopes:
   frames=fs.get(s['callchainId'],[]);key=hashlib.sha256(json.dumps(frames,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()if frames else None
   if key:canonical[key]=frames
   s['chain_key']=key;s['project_frames']=[r['frame']for r in frames if '/vllm_ascend/'in r['frame']or '/vllm/vllm/'in r['frame']]
  rows.append(dict(database_index=d['database_index'],graph_ordinal=gi,CANN_API=a,same_thread_enclosing_scopes=scopes,recorded_dummy_ancestry=any(': _dummy_run'in f or ': dummy_run'in f for s in scopes for f in s['project_frames']),recorded_profile_ancestry=any(': profile_run'in f for s in scopes for f in s['project_frames'])))
 db.close()
summary=dict(created_utc=datetime.now(timezone.utc).isoformat(),graphs=len(rows),enclosing_scope_count_hist=dict(Counter(len(r['same_thread_enclosing_scopes'])for r in rows)),recorded_dummy_ancestry_graphs=sum(r['recorded_dummy_ancestry']for r in rows),recorded_profile_ancestry_graphs=sum(r['recorded_profile_ancestry']for r in rows),enclosing_scope_name_counts=dict(Counter(s['name']for r in rows for s in r['same_thread_enclosing_scopes'])),canonical_fullchains=len(canonical),no_target_source_read=True,no_device_or_service_work=True,overall_complete=False,limits=['Same-thread host scope contains native CANN submission; this is submission context, not tensor identity or client schedule. Existing native CANN/control-pair graph relation retained separately.'])
payload=(json.dumps(dict(summary=summary,graph_contexts=rows,canonical_fullchains=canonical),ensure_ascii=False,separators=(',',':'))+'\n').encode();OUT.write_bytes(payload)
print(json.dumps(dict(summary=summary,first_graph=rows[:1],full_result=str(OUT),result_sha256=hashlib.sha256(payload).hexdigest()),ensure_ascii=False))
