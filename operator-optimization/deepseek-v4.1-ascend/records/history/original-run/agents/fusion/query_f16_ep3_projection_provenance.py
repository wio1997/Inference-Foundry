#!/usr/bin/env python3
"""Query 2 fixed EP3 L36 projection tasks in saved profiler DB, CPU only."""
import hashlib,json,pathlib,sqlite3
p=pathlib.Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029/runtime/diagnostic-traces/dp0_pp0_tp3_dcp0_ep3_rank3_2128_20261029023525650_ascend_pt/ASCEND_PROFILER_OUTPUT/ascend_pytorch_profiler_3.db')
h=hashlib.sha256()
with p.open('rb') as f:
 for b in iter(lambda:f.read(1048576),b''):h.update(b)
assert h.hexdigest()=='084b8718888e33271435ec79d7573c95ca1615f3e7484c807ea24590fad234c4'
c=sqlite3.connect(p.as_uri()+'?mode=ro',uri=True);c.row_factory=sqlite3.Row;c.execute('PRAGMA query_only=ON')
target=[('first',41440,1793241329812070995,1793241329147301570),('second',41441,1793241329812341915,1793241329149780480)];out=[]
for label,task_id,device_ns,host_ns in target:
 rows=[dict(r) for r in c.execute('SELECT * FROM TASK WHERE streamId=47 AND taskId=? AND startNs=?',(task_id,device_ns)).fetchall()];assert len(rows)==1
 t=rows[0];cid=t['connectionId'];gid=t['globalTaskId'];conn=[r[0] for r in c.execute('SELECT id FROM CONNECTION_IDS WHERE connectionId=?',(cid,)).fetchall()];apis=[]
 for ident in conn:
  matches=c.execute('SELECT p.startNs,p.endNs,p.globalTid,p.connectionId,p.callchainId,s.value FROM PYTORCH_API p LEFT JOIN STRING_IDS s ON s.id=p.name WHERE p.connectionId=?',(ident,)).fetchall()
  for r in matches:
   item=dict(r);frames=c.execute('SELECT p.stackDepth,s.value FROM PYTORCH_CALLCHAINS p JOIN STRING_IDS s ON s.id=p.stack WHERE p.id=? ORDER BY p.stackDepth',(r['callchainId'],)).fetchall();item['frames']=[{'depth':z[0],'frame':str(z[1]).rsplit('/',1)[-1]} for z in frames if any(v in str(z[1]) for v in ('dsa_v41','dsa_cp.py','model.py','linear.py','torch/_ops.py'))];apis.append(item)
 same_start=c.execute('SELECT p.startNs,p.endNs,p.globalTid,p.connectionId,p.callchainId,s.value FROM PYTORCH_API p LEFT JOIN STRING_IDS s ON s.id=p.name WHERE p.globalTid=? AND p.startNs=?',(9139690408016,str(host_ns))).fetchall()
 out.append({'label':label,'task':t,'connection_host_ids':conn,'apis':apis,'host_exact_start': [dict(r) for r in same_start],'task_count_by_gid':c.execute('SELECT count(*) FROM TASK WHERE globalTaskId=?',(gid,)).fetchone()[0]})
c.close();print(json.dumps({'db_sha256':h.hexdigest(),'scope':'two fixed EP3 layer36 target tasks; selected connection, start and limited callchain frames','rows':out,'limits':['Recorded host callchains not same as proven loaded source; no tensor/storage identity.','Connection crosswalk does not prove synchronization, tensor dependence or whole model coverage.','Only selected records, not independently measured performance.']},ensure_ascii=False,indent=2))
