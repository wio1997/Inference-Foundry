#!/usr/bin/env python3
"""Read three saved profiler gap windows; no device activity."""
import argparse,json,sqlite3
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--task-root',required=True);a=ap.parse_args();root=Path(a.task_root)/'diagnostic-20261029/runtime/diagnostic-traces'
cases=[('EP2-L15','dp0_pp0_tp2_dcp0_ep2_rank2_1113_20261029023525648_ascend_pt',2,43942,1793241334299714963,43946,1793241334305544006),('EP3-L15','dp0_pp0_tp3_dcp0_ep3_rank3_2128_20261029023525650_ascend_pt',3,43942,1793241334309640915,43946,1793241334316357695),('EP2-L28','dp0_pp0_tp2_dcp0_ep2_rank2_1113_20261029023525648_ascend_pt',2,44673,1793241334589297557,44677,1793241334589411535)]
out=[]
for name,rank,dev,pad_id,pad_start,gate_id,gate_start in cases:
 ds=list((root/rank/'ASCEND_PROFILER_OUTPUT').glob('ascend_pytorch_profiler_*.db'));assert len(ds)==1
 c=sqlite3.connect(ds[0].resolve().as_uri()+'?mode=ro',uri=True);c.execute('PRAGMA query_only=ON')
 def task(tid,start):
  xs=c.execute('SELECT startNs,endNs,deviceId,streamId,taskId,modelId FROM TASK WHERE deviceId=? AND streamId=47 AND taskId=? AND startNs=?',(dev,tid,start)).fetchall();assert len(xs)==1 and xs[0][5]==4294967295;return xs[0]
 pad=task(pad_id,pad_start);gate=task(gate_id,gate_start);lo=int(pad[1]);hi=int(gate[0]);assert lo<hi
 sql='SELECT t.startNs,t.endNs,t.deviceId,t.streamId,t.taskId,s.value,n.value FROM TASK t LEFT JOIN STRING_IDS s ON s.id=t.taskType LEFT JOIN COMPUTE_TASK_INFO i ON i.globalTaskId=t.globalTaskId LEFT JOIN STRING_IDS n ON n.id=i.name WHERE t.deviceId=? AND ((t.startNs<? AND t.endNs>?) OR (t.startNs>=? AND t.startNs<? AND t.startNs=t.endNs)) ORDER BY t.startNs'
 rows=c.execute(sql,(dev,hi,lo,lo,hi)).fetchall();assert len(rows)==3 and all(int(r[3])==47 for r in rows)
 assert [r[6] for r in rows]==['aclnnEqScalar_EqualAiCore_Equal','aclnnInplaceFillScalar_FillAiCore_Fill','aclnnSWhere_SelectV2AiCore_SelectV2']
 comm=c.execute('SELECT opId FROM COMMUNICATION_OP WHERE deviceId=? AND startNs<? AND endNs>? LIMIT 501',(dev,hi,lo)).fetchall();assert len(comm)<501
 points=c.execute('SELECT timestampNs FROM COMMUNICATION_TASK_INFO WHERE timestampNs>=? AND timestampNs<? LIMIT 501',(lo,hi)).fetchall();assert len(points)<501
 mstx=c.execute('SELECT COUNT(*) FROM MSTX_EVENTS WHERE startNs<? AND endNs>?',(hi,lo)).fetchone()[0]
 c.close()
 out.append({'case':name,'window_ns':hi-lo,'all_stream_task_count':len(rows),'tasks':[{'start_offset_ns':int(r[0])-lo,'end_offset_ns':int(r[1])-lo,'stream':r[3],'task':r[4],'task_type':r[5],'compute_name':r[6]} for r in rows],'matching_communication_ops':len(comm),'communication_time_points_all_db':len(points),'mstx_nonzero_overlap_all_db':mstx})
print(json.dumps({'scope':'three windows in two saved rank DBs, half-open pad-end to gate-start; existing exported rows only','cases':out,'limits':['TASK all streams on selected device only; missing trace records remain unknown','COMMUNICATION_TASK_INFO/MSTX have no device identity; MSTX point events excluded','Timestamp overlap across different tracks requires calibration; not proof of idle or causality','No historical binary or tensor identity established']},indent=2))
