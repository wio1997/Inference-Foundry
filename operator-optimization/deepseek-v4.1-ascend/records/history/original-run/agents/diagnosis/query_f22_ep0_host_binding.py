#!/usr/bin/env python3
"""Bounded read-only host/flow/device checks for six saved EP0 MoE events."""
import hashlib,json,pathlib,sqlite3,time
R=pathlib.Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41')
ps=list((R/'diagnostic-20261029/runtime/diagnostic-traces').glob('*_ep0_rank*/ASCEND_PROFILER_OUTPUT/ascend_pytorch_profiler_0.db'));assert len(ps)==1;p=ps[0]
pre=p.stat();h=hashlib.sha256()
with p.open('rb') as f:
 for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
c=sqlite3.connect(p.as_uri()+'?mode=ro',uri=True);c.row_factory=sqlite3.Row;c.execute('PRAGMA query_only=ON');start=time.monotonic();c.set_progress_handler(lambda:int(time.monotonic()-start>20),100000)
tid=3161095930592;scope=list(c.execute('SELECT p.startNs,p.endNs,p.globalTid,s.value name FROM PYTORCH_API p JOIN STRING_IDS s ON s.id=p.name WHERE p.globalTid=? AND s.value=?',(tid,'wio_diag:DeepseekV41DecoderLayer.forward:736-51')));assert len(scope)==1;scope=scope[0]
refs=[(496477,1198825,120151,4722,1793241333374667046,'DispatchFFNCombine'),(496478,1198831,120161,4723,1793241333401140997,'aclnnDynamicQuantV2_DynamicQuant_DynamicQuant'),(496479,1198838,120166,4724,1793241333401165397,'aclnnQuantMatmulWeightNz_QuantBatchMatmulV3_QuantBatchMatmulV3'),(496480,1198846,120171,4725,1793241333401242975,'DequantSwigluQuant'),(496481,1198853,120180,4726,1793241333401272875,'aclnnQuantMatmulWeightNz_QuantBatchMatmulV3_QuantBatchMatmulV3'),(496482,1198860,120188,4727,1793241333401331474,'aclnnForeachAddListV2_ForeachAddListV2_ForeachAddList')]
rows=[]
for host,chain,flow,gid,ns,expected in refs:
 a=list(c.execute('SELECT p.startNs,p.endNs,p.globalTid,p.connectionId,p.callchainId,s.value name FROM PYTORCH_API p LEFT JOIN STRING_IDS s ON s.id=p.name WHERE p.connectionId=?',(host,)));assert len(a)==1;a=a[0];assert a['globalTid']==tid and a['callchainId']==chain and int(scope['startNs'])<=int(a['startNs'])<=int(a['endNs'])<=int(scope['endNs'])
 links=list(c.execute('SELECT connectionId FROM CONNECTION_IDS WHERE id=?',(host,)));assert len(links)==1 and links[0][0]==flow
 all_tasks=list(c.execute('SELECT t.startNs,t.endNs,t.globalTaskId,t.streamId,t.taskId,s.value name FROM TASK t LEFT JOIN COMPUTE_TASK_INFO i ON i.globalTaskId=t.globalTaskId LEFT JOIN STRING_IDS s ON s.id=i.name WHERE t.connectionId=?',(flow,)))
 named=[x for x in all_tasks if x['globalTaskId']==gid and x['startNs']==ns and x['name']==expected];assert len(named)==1 and named[0]['streamId']==45
 assert c.execute('SELECT count(*) FROM TASK WHERE globalTaskId=?',(gid,)).fetchone()[0]==1 and c.execute('SELECT count(*) FROM COMPUTE_TASK_INFO WHERE globalTaskId=?',(gid,)).fetchone()[0]==1
 frames=[]
 for f in c.execute('SELECT c.stackDepth,s.value frame FROM PYTORCH_CALLCHAINS c JOIN STRING_IDS s ON s.id=c.stack WHERE c.id=? ORDER BY c.stackDepth',(chain,)):
  v=f['frame'] or ''
  if any(x in v for x in ('shared_experts.py','fused_moe.py','moe_runner.py','model.py','moe_comm_method.py','w8a8_dynamic.py','routed_experts.py')):frames.append({'depth':f['stackDepth'],'frame':v.rsplit('/',1)[-1]})
 rows.append({'host':dict(a),'flow':flow,'named_task':dict(named[0]),'other_task_count':len(all_tasks)-1,'frames':frames})
c.close();post=p.stat();assert (pre.st_size,pre.st_mtime_ns,pre.st_ino)==(post.st_size,post.st_mtime_ns,post.st_ino)
print(json.dumps({'db_sha256':h.hexdigest(),'db_size':pre.st_size,'scope':dict(scope),'rows':rows,'limits':['Saved host stacks and flow associations do not bind loaded source file hash, historical binary, tensor storage or values.','Two flows each include an additional unnamed TASK; it is not independently counted as a new operator.','Stage source is a reference snapshot, not proof of historical execution identity.']},indent=2))
