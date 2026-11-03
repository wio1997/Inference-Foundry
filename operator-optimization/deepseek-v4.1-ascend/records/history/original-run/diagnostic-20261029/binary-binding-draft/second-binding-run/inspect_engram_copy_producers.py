"""Bounded same-thread profiler view ops preceding four copy enqueue anchors."""
from pathlib import Path
import json, sqlite3
root=Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029/binary-binding-draft/second-binding-run/offline-ep8-20261029')
status=json.loads((root/'status.json').read_text());item=next(x for x in status['dbs'] if x['path'].endswith('/ascend_pytorch_profiler_8.db'));db=root/item['path']
assert not Path(str(db)+'-wal').exists() and not Path(str(db)+'-journal').exists()
c=sqlite3.connect(db.resolve().as_uri()+'?mode=ro&immutable=1',uri=True);c.row_factory=sqlite3.Row;c.execute('PRAGMA query_only=ON')
# Four anchors were extracted from exact callchain-containing copy scopes.
anchors=[('746-20-a',1793250233639810590),('746-20-b',1793250233642447830),('746-65-a',1793250236481524370),('746-65-b',1793250236484217420)]
tid=3204045603562
names=('aten::slice','aten::select','aten::index_select','aten::cat','aten::contiguous','aten::clone','aten::copy_')
q='''SELECT a.startNs,a.endNs,a.callchainId,s.value name,sh.value inputShapes,dt.value inputDtypes FROM PYTORCH_API a JOIN STRING_IDS s ON s.id=a.name LEFT JOIN STRING_IDS sh ON sh.id=a.inputShapes LEFT JOIN STRING_IDS dt ON dt.id=a.inputDtypes WHERE a.type=50001 AND a.globalTid=? AND CAST(a.startNs AS INTEGER)>=? AND CAST(a.startNs AS INTEGER)<=? AND s.value IN (?,?,?,?,?,?,?) ORDER BY CAST(a.startNs AS INTEGER)'''
out=[]
for label,t in anchors:
 rows=[dict(x) for x in c.execute(q,(tid,t-5_000_000,t,*names)).fetchall()]
 if len(rows)>250:raise RuntimeError('too many candidate rows')
 for row in rows:
  if row['callchainId'] is not None:
   chain=c.execute('SELECT k.stackDepth,s.value FROM PYTORCH_CALLCHAINS k LEFT JOIN STRING_IDS s ON s.id=k.stack WHERE k.id=? ORDER BY k.stackDepth',(row['callchainId'],)).fetchall()
   row['callchain']=[dict(x) for x in chain]
 relevant=[x for x in rows if any(('deepseek_v41/model.py(1217)' in (a.get('value') or '') or 'engram/embedding.py(342)' in (a.get('value') or '') or 'engram/hash_state.py' in (a.get('value') or '') or 'engram/parallel.py' in (a.get('value') or '')) for a in x.get('callchain',[]))]
 out.append({'anchor':label,'start_ns':t,'window_ns':5000000,'candidate_count':len(rows),'relevant_count':len(relevant),'rows':relevant})
print(json.dumps({'db_sha256_expected':item['sha256'],'scope':'same TID, bounded 5ms before each exact anchored copy start, filtered op and matching stack; containment/order does not prove tensor identity','anchors':out},indent=2))
