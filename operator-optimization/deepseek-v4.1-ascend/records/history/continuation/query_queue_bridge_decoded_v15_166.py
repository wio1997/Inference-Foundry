from pathlib import Path
from collections import Counter,defaultdict
from datetime import datetime,timezone
import hashlib,json,sqlite3,socket
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
P=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/hostprof-20261030T212812Z-360e0c72/analysis/native-execution-identity-v14')
def pin(path,sha):
 b=path.read_bytes();assert hashlib.sha256(b).hexdigest()==sha;return json.loads(b)
x=pin(P/'index.json','a3aa47cdf9170785432ecbc18117a489f3fba37d89959f110f4fcbd8890c7dc4')
orphans=pin(P/'unassigned-compute-native-context-v14.json','1d8517a1a1e5c9f7a272db2482ac0550d86cb40b15d9e5c0f5d3a86e3a80e3c8')
OUT=P/'queue-bridge-decoded-connections-v15.json';assert not OUT.exists();rows=[];canonical={}
for di in sorted({r['database_index']for r in orphans['records']}):
 ref=x['databases'][di];d=pin(Path(ref['path']),ref['sha256'])
 db=sqlite3.connect(Path(d['database_path']).as_uri()+'?mode=ro&immutable=1',uri=True);db.row_factory=sqlite3.Row;db.execute('PRAGMA query_only=ON')
 for r in [r for r in orphans['records']if r['database_index']==di]:
  t=r['task'];launch=[a for a in r['native_CANN_APIs']if a['connectionId']==t['connectionId']];assert len(launch)==1;a=launch[0]
  scopes=next(s['same_thread_enclosing_PYTORCH_API_records']for s in r['host_scope_candidates']if s['cann_api_rowid']==a['cann_api_rowid'])
  deq=[s for s in scopes if s['name'].startswith('Dequeue@')]
  out=dict(database_index=di,TASK=t,definition=r['definition'],exact_native_launch=a,backend_Dequeue_scopes=deq,queue_key_matches=[],enclosing_frontend_scopes=[])
  for q in deq:
   flow_values={v['connectionId']for v in db.execute('SELECT connectionId FROM CONNECTION_IDS WHERE id=?',(q['connectionId'],))}
   aliases=[]
   if flow_values:
    vals=sorted(flow_values);aliases=[dict(v)for v in db.execute('SELECT id,connectionId FROM CONNECTION_IDS WHERE connectionId IN ('+','.join('?'for _ in vals)+')',vals)]
   foreign_keys=sorted({v['id']for v in aliases});same=[]
   if foreign_keys:
    same=[dict(v)for v in db.execute('SELECT p.rowid AS api_rowid,CAST(p.startNs AS INTEGER)AS startNs,CAST(p.endNs AS INTEGER)AS endNs,p.globalTid,p.connectionId,p.callchainId,s.value AS name FROM PYTORCH_API p LEFT JOIN STRING_IDS s ON s.id=p.name WHERE p.connectionId IN ('+','.join('?'for _ in foreign_keys)+') ORDER BY CAST(p.startNs AS INTEGER)',foreign_keys)]
   out['queue_key_matches'].append(dict(backend_api_rowid=q['api_rowid'],queue_connection_list_id=q['connectionId'],decoded_flow_values=sorted(flow_values),CONNECTION_IDS_alias_rows=aliases,PYTORCH_API_records=same))
   for enq in [v for v in same if v['name'].startswith('Enqueue@')]:
    ps=[dict(v)for v in db.execute('SELECT p.rowid AS api_rowid,CAST(p.startNs AS INTEGER)AS startNs,CAST(p.endNs AS INTEGER)AS endNs,p.globalTid,p.connectionId,p.callchainId,s.value AS name FROM PYTORCH_API p LEFT JOIN STRING_IDS s ON s.id=p.name WHERE p.globalTid=? AND CAST(p.startNs AS INTEGER)<=? AND CAST(p.endNs AS INTEGER)>=? ORDER BY CAST(p.startNs AS INTEGER),CAST(p.endNs AS INTEGER)DESC',(enq['globalTid'],enq['startNs'],enq['endNs']))]
    out['enclosing_frontend_scopes'].append(dict(Enqueue=enq,scopes=ps))
  ids={s['callchainId']for group in out['enclosing_frontend_scopes']for s in group['scopes']if s['callchainId']is not None};frames=defaultdict(list)
  if ids:
   literal=','.join(str(int(cid))for cid in ids)
   for f in db.execute('SELECT c.id,c.stackDepth,s.value AS frame FROM PYTORCH_CALLCHAINS c JOIN STRING_IDS s ON s.id=c.stack WHERE c.id IN ('+literal+') ORDER BY c.id,c.stackDepth'):frames[f['id']].append(dict(stackDepth=f['stackDepth'],frame=f['frame']))
  for group in out['enclosing_frontend_scopes']:
   for s in group['scopes']:
    fs=frames.get(s['callchainId'],[]);key=hashlib.sha256(json.dumps(fs,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()if fs else None
    if key:canonical[key]=fs
    s['chain_key']=key;s['project_frames']=[f['frame']for f in fs if '/vllm_ascend/'in f['frame']or '/vllm/vllm/'in f['frame']]
  rows.append(out)
 db.close()
summary=dict(created_utc=datetime.now(timezone.utc).isoformat(),compute_records=len(rows),unique_backend_Dequeue=sum(len(r['backend_Dequeue_scopes'])==1 for r in rows),queue_Enqueue_count_hist=dict(Counter(sum(v['name'].startswith('Enqueue@')for g in r['queue_key_matches']for v in g['PYTORCH_API_records'])for r in rows)),frontend_scope_count_hist=dict(Counter(sum(len(g['scopes'])for g in r['enclosing_frontend_scopes'])for r in rows)),records_with_project_source_candidates=sum(any(s['project_frames']for g in r['enclosing_frontend_scopes']for s in g['scopes'])for r in rows),canonical_fullchains=len(canonical),no_target_source_or_device_work=True,overall_complete=False,limits=['TASK→CANN uses exact native connectionId; CANN→Dequeue and Enqueue→parent use same-thread containment. PYTORCH connectionId foreign keys are decoded through CONNECTION_IDS.id; Enqueue/Dequeue share an actual flow value, not necessarily a list ID. Actual DB row relations retained. All multiplicities retained; no cross-table numeric-ID guesses or nearest-time fallback.'])
payload=(json.dumps(dict(summary=summary,records=rows,canonical_fullchains=canonical),ensure_ascii=False,separators=(',',':'))+'\n').encode();OUT.write_bytes(payload)
public=dict(summary=summary,record_summaries=[dict(database_index=r['database_index'],task_rowid=r['TASK']['task_record_rowid'],op=r['definition']['opType_text'],queue_matches=r['queue_key_matches'],frontend_scopes=r['enclosing_frontend_scopes'])for r in rows],full_result=str(OUT),result_sha256=hashlib.sha256(payload).hexdigest())
if len(json.dumps(public).encode())>30000:public.pop('record_summaries');public['display_truncated']=True
print(json.dumps(public,ensure_ascii=False))
