#!/usr/bin/env python3
"""Inspect selected CANN launch-thread events from three saved scopes, no NPU."""
import argparse,json,sqlite3
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--task-root',required=True);a=ap.parse_args();root=Path(a.task_root)/'diagnostic-20261029/runtime/diagnostic-traces'
cases=[('EP2-L15','dp0_pp0_tp2_dcp0_ep2_rank2_1113_20261029023525648_ascend_pt',2,43942,1793241334299714963,1793241334305544006),('EP3-L15','dp0_pp0_tp3_dcp0_ep3_rank3_2128_20261029023525650_ascend_pt',3,43942,1793241334309640915,1793241334316357695),('EP2-L28','dp0_pp0_tp2_dcp0_ep2_rank2_1113_20261029023525648_ascend_pt',2,44673,1793241334589297557,1793241334589411535)]
out=[]
for label,rank,device,first,pad_start,gate_start in cases:
 ds=list((root/rank/'ASCEND_PROFILER_OUTPUT').glob('ascend_pytorch_profiler_*.db'));assert len(ds)==1
 c=sqlite3.connect(ds[0].resolve().as_uri()+'?mode=ro',uri=True);c.execute('PRAGMA query_only=ON')
 launches=[]
 for tid in range(first,first+5):
  ts=c.execute('SELECT taskId,startNs,connectionId FROM TASK WHERE deviceId=? AND streamId=47 AND taskId=? AND startNs>=? AND startNs<=?',(device,tid,pad_start,gate_start)).fetchall();assert len(ts)==1
  conn=ts[0][2];events=c.execute('SELECT a.startNs,a.endNs,a.globalTid,s.value FROM CANN_API a LEFT JOIN STRING_IDS s ON s.id=a.name WHERE a.connectionId=? AND s.value=?',(conn,'launch')).fetchall();assert len(events)==1
  e=events[0];launches.append({'task':tid,'connection':conn,'start_ns':int(e[0]),'duration_ns':int(e[1])-int(e[0]),'thread':e[2]})
 assert len({x['thread'] for x in launches})==1
 th=launches[0]['thread'];lo=launches[0]['start_ns'];hi=launches[-1]['start_ns'];assert lo<hi
 sql='SELECT a.startNs,a.endNs,a.connectionId,s.value FROM CANN_API a LEFT JOIN STRING_IDS s ON s.id=a.name WHERE a.globalTid=? AND a.startNs>=? AND a.startNs<? ORDER BY a.startNs LIMIT 1001'
 events=c.execute(sql,(th,lo,hi)).fetchall();assert len(events)<1001
 q='SELECT a.startNs,a.endNs,a.connectionId,s.value FROM PYTORCH_API a LEFT JOIN STRING_IDS s ON s.id=a.name WHERE a.globalTid=? AND a.startNs>=? AND a.startNs<? ORDER BY a.startNs LIMIT 1001'
 py=c.execute(q,(th,lo,hi)).fetchall();assert len(py)<1001
 for index,(api,tiling,dequeue) in enumerate((('aclnnEqScalar','Equal_Tiling','Dequeue@aclnnEqScalar'),('aclnnInplaceFillScalar','Fill_Tiling','Dequeue@aclnnInplaceFillScalar'),('aclnnSWhere','SelectV2_Tiling','Dequeue@aclnnSWhere')),start=1):
  matching_api=[x for x in events if x[3]==api];matching_tiling=[x for x in events if x[3]==tiling];matching_dequeue=[x for x in py if x[3]==dequeue]
  assert len(matching_api)==len(matching_tiling)==len(matching_dequeue)==1,(label,api,tiling,dequeue)
  x,y,z=matching_api[0],matching_tiling[0],matching_dequeue[0];launch=launches[index]
  assert int(z[0])<=int(x[0])<=int(y[0])<int(y[1])<=launch['start_ns'] and launch['start_ns']+launch['duration_ns']<=int(x[1])<=int(z[1]),(label,api)
 c.close()
 out.append({'case':label,'launch_thread':th,'pad_to_gate_launch_span_ns':hi-lo,'launches':[{**v,'start_offset_ns':v['start_ns']-lo} for v in launches],'cann_events':[{'offset_ns':int(s)-lo,'duration_ns':int(e)-int(s),'connection':conn,'name':name} for s,e,conn,name in events],'pytorch_events':[{'offset_ns':int(s)-lo,'duration_ns':int(e)-int(s),'connection':conn,'name':name} for s,e,conn,name in py]})
print(json.dumps({'scope':'three saved scopes; within each launch thread only','cases':out,'limits':['Connection links launch to TASK but not tensor identity','Tiling to aclnn attributed by nested intervals and name, not direct flow','Host/device clock relation and critical path unproved; durations include profiling and blocking']},indent=2))
