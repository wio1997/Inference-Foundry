from pathlib import Path
from collections import defaultdict
from datetime import datetime,timezone
import hashlib,json,sqlite3,socket,time
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
P=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/hostprof-20261030T212812Z-360e0c72/analysis')
def pin(path,sha):
 b=path.read_bytes();assert hashlib.sha256(b).hexdigest()==sha;return json.loads(b)
C=P/'native-communications-v15';E=P/'native-execution-identity-v14'
idx=pin(C/'index.json','20ea4452458001c372c2d2c9b5cd9f4f13d951cfa7c77bb7e5e4bf3faa259016');ref=idx['databases'][8];d=pin(Path(ref['path']),ref['sha256']);op=next(o for o in d['COMM_OP_records']if o['COMM_OP_rowid']==232)
assert not op['native_PYTORCH_API_candidates']and not op['graph_control_container_candidates']
OUT=C/'unassigned-communication-native-queue-context-v16.json';assert not OUT.exists()
db=sqlite3.connect(Path(d['database_path']).as_uri()+'?mode=ro&immutable=1',uri=True);db.row_factory=sqlite3.Row;db.execute('PRAGMA query_only=ON');t0=time.monotonic();db.set_progress_handler(lambda:int(time.monotonic()-t0>100),100000)
sql='SELECT p.rowid AS api_rowid,CAST(p.startNs AS INTEGER)AS startNs,CAST(p.endNs AS INTEGER)AS endNs,p.globalTid,p.connectionId,p.callchainId,s.value AS name FROM PYTORCH_API p LEFT JOIN STRING_IDS s ON s.id=p.name '
def parents(a):return [dict(r)for r in db.execute(sql+'WHERE p.globalTid=? AND CAST(p.startNs AS INTEGER)<=? AND CAST(p.endNs AS INTEGER)>=?',(a['globalTid'],a['startNs'],a['endNs']))]
launches=[dict(r)for r in db.execute('SELECT a.rowid AS cann_api_rowid,a.*,s.value AS name_text FROM CANN_API a LEFT JOIN STRING_IDS s ON s.id=a.name WHERE a.connectionId=?',(op['connectionId'],))]
rows=[];need=set()
for a in launches:
 ps=parents(a);qs=[]
 for q in [p for p in ps if p['name'].startswith('Dequeue@')]:
  flows=sorted({r['connectionId']for r in db.execute('SELECT connectionId FROM CONNECTION_IDS WHERE id=?',(q['connectionId'],))});aliases=[];same=[]
  if flows:
   aliases=[dict(r)for r in db.execute('SELECT id,connectionId FROM CONNECTION_IDS WHERE connectionId IN ('+','.join('?'for _ in flows)+')',flows)]
  ids=sorted({r['id']for r in aliases})
  if ids:same=[dict(r)for r in db.execute(sql+'WHERE p.connectionId IN ('+','.join('?'for _ in ids)+')',ids)]
  es=[]
  for e in [p for p in same if p['name'].startswith('Enqueue@')]:
   ss=parents(e);need.update(p['callchainId']for p in ss if p['callchainId']is not None)
   es.append(dict(Enqueue=e,same_process=e['globalTid']>>32==q['globalTid']>>32,name_matches=e['name'].removeprefix('Enqueue@')==q['name'].removeprefix('Dequeue@'),enqueue_precedes_dequeue=e['endNs']<=q['startNs'],enclosing_source_scopes=ss))
  qs.append(dict(Dequeue=q,decoded_flows=flows,CONNECTION_IDS_alias_rows=aliases,queue_API_records=same,frontend_candidates=es))
 need.update(p['callchainId']for p in ps if p['callchainId']is not None)
 rows.append(dict(CANN_API=a,same_thread_enclosing_scopes=ps,queue_candidates=qs))
frames=defaultdict(list);canonical={}
if need:
 vals=','.join(str(int(i))for i in need)
 for r in db.execute('SELECT c.id,c.stackDepth,s.value AS frame FROM PYTORCH_CALLCHAINS c JOIN STRING_IDS s ON s.id=c.stack WHERE c.id IN ('+vals+') ORDER BY c.id,c.stackDepth'):frames[r['id']].append(dict(stackDepth=r['stackDepth'],frame=r['frame']))
def bind(scopes):
 for p in scopes:
  fs=frames.get(p['callchainId'],[]);key=hashlib.sha256(json.dumps(fs,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()if fs else None
  if key:canonical[key]=fs
  p['chain_key']=key;p['project_frames']=[r['frame']for r in fs if '/vllm_ascend/'in r['frame']or '/vllm/vllm/'in r['frame']]
for r in rows:
 bind(r['same_thread_enclosing_scopes'])
 for q in r['queue_candidates']:
  for e in q['frontend_candidates']:
   bind(e['enclosing_source_scopes']);ss=[p for p in e['enclosing_source_scopes']if p['project_frames']]
   width=min((p['endNs']-p['startNs']for p in ss),default=None);e['innermost_project_source_candidates']=[p for p in ss if p['endNs']-p['startNs']==width]
db.close()
summary=dict(created_utc=datetime.now(timezone.utc).isoformat(),database_index=8,COMM_OP_rowid=232,exact_native_connection=op['connectionId'],CANN_API_records=len(launches),backend_dequeue_candidates=sum(len(r['queue_candidates'])for r in rows),frontend_candidates=sum(len(q['frontend_candidates'])for r in rows for q in r['queue_candidates']),canonical_chains=len(canonical),no_model_source_read_or_device_work=True,overall_complete=False,limits=['Exact COMM_OP.connectionId to CANN_API native namespace only; no comparison to list IDs or unrelated numeric namespaces.','Queue foreign keys decoded through CONNECTION_IDS; all multiplicities, PID, name and CPU ordering retained. Source association needs semantic review; no nearest-time fallback.'])
payload=(json.dumps(dict(summary=summary,COMM_OP=op,native_launch_context=rows,canonical_fullchains=canonical),ensure_ascii=False,separators=(',',':'))+'\n').encode();OUT.write_bytes(payload)
print(json.dumps(dict(summary=summary,native_launch_context=rows,full_result=str(OUT),result_sha256=hashlib.sha256(payload).hexdigest()),ensure_ascii=False))
