#!/usr/bin/env python3
"""Read-only fixed EP3 notification IDs and host flow associations."""
import sqlite3,pathlib,json,hashlib
p=pathlib.Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029/runtime/diagnostic-traces/dp0_pp0_tp3_dcp0_ep3_rank3_2128_20261029023525650_ascend_pt/ASCEND_PROFILER_OUTPUT/ascend_pytorch_profiler_3.db')
c=sqlite3.connect(p.as_uri()+'?mode=ro',uri=True);c.row_factory=sqlite3.Row;c.execute('PRAGMA automatic_index=OFF')
q='''SELECT ct.notifyId,ct.opId,ct.globalTaskId,s.value name,t.startNs,t.endNs,t.streamId,t.taskId,ct.srcRank,ct.dstRank,ct.transportType,ct.linkType FROM COMMUNICATION_TASK_INFO ct JOIN TASK t ON t.globalTaskId=ct.globalTaskId LEFT JOIN STRING_IDS s ON s.id=ct.name WHERE ct.opId=249 AND ct.notifyId IN (7290,7292,7294,7357) ORDER BY t.startNs'''
rows=[dict(r) for r in c.execute(q)]
q='''SELECT p.startNs,p.endNs,p.globalTid,p.connectionId,p.callchainId,s.value name FROM PYTORCH_API p LEFT JOIN STRING_IDS s ON p.name=s.id WHERE p.connectionId IN (431408,431416) ORDER BY CAST(p.startNs AS INTEGER)'''
host=[dict(r) for r in c.execute(q)];assert len(host)==2 and host[0]['globalTid']==host[1]['globalTid']
stack=[]
q='SELECT c.stackDepth,s.value frame FROM PYTORCH_CALLCHAINS c LEFT JOIN STRING_IDS s ON c.stack=s.id WHERE c.id=? ORDER BY c.stackDepth'
for r in c.execute(q,(host[1]['callchainId'],)):
 if any(x in (r['frame'] or '') for x in ['distributed','sequence_parallel','deepseek_v41/model.py']):stack.append(dict(r))
links=[dict(r) for r in c.execute('SELECT id,connectionId FROM CONNECTION_IDS WHERE id IN (431408,431416) AND connectionId IN (102676,102723) ORDER BY id')]
first={}
for nid in [7290,7292,7294,7357]:
 x=[r for r in rows if r['notifyId']==nid];wait=[r for r in x if r['name']=='Notify_Wait'];record=[r for r in x if r['name']=='Notify_Record'];first[str(nid)]={'waits':len(wait),'records':len(record),'first_wait':wait[0] if wait else None,'first_record':record[0] if record else None,'first_wait_end_minus_first_record_end_ns':wait[0]['endNs']-record[0]['endNs'] if wait and record else None}
print(json.dumps({'source_hash':'see previous exact TASK/DB record','op_id':249,'host':host,'host_start_difference_ns':int(host[1]['startNs'])-int(host[0]['startNs']),'selected_collective_host_stack':stack,'connection_links':links,'notify_sequences':first,'rows':rows,'limits':'one profiled EP3 rank, saved database, no new NPU. Same ID repeated per op; no validated 9.2 pairing contract, generation or cross-track timing calibration. Stack identified a historical frame, frozen source binary identity not bound.'},indent=2))
