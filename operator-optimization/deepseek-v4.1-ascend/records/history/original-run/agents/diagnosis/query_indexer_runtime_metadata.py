#!/usr/bin/env python3
"""Inspect three fixed EP8 source-validated layer24 windows, SQLite read-only."""
import argparse,json,sqlite3
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--db',required=True);a=ap.parse_args()
p=Path(a.db).resolve(strict=True);c=sqlite3.connect(p.as_uri()+'?mode=ro',uri=True);c.row_factory=sqlite3.Row;c.execute('PRAGMA query_only=ON')
windows=[(1,1793152958896320834,1793152958897203736),(2,1793152960208017846,1793152960209362219),(3,1793152960284475459,1793152960285774513)]
q='''SELECT t.startNs,t.endNs,t.globalTaskId,t.connectionId,t.modelId,t.streamId,t.taskId,
 s.value AS op_type,i.attrInfo,i.inputShapes,i.inputDataTypes,i.outputShapes,i.opState
FROM TASK t JOIN COMPUTE_TASK_INFO i ON i.globalTaskId=t.globalTaskId JOIN STRING_IDS s ON s.id=i.opType
WHERE t.modelId=? AND t.streamId=? AND CAST(t.startNs AS INTEGER)>=? AND CAST(t.endNs AS INTEGER)<=? AND s.value=?'''
def string_for_id(id):
 if not isinstance(id,int):return None
 r=c.execute('SELECT value FROM STRING_IDS WHERE id=?',(id,)).fetchone()
 return None if r is None else r[0][:2000]
out=[]
for run,start,end in windows:
 rows=c.execute(q,(49,89,start,end,'VllmAscendQuantLightningIndexerV2')).fetchall()
 if len(rows)!=1:raise RuntimeError('Layer24 run%d expected unique indexer, got %d'%(run,len(rows)))
 t=dict(rows[0]); decoded={k:string_for_id(t[k]) for k in ('attrInfo','inputShapes','inputDataTypes','outputShapes')}
 ca=c.execute('''SELECT a.startNs,a.endNs,s.value AS name FROM CANN_API a LEFT JOIN STRING_IDS s ON s.id=a.name WHERE a.connectionId=? LIMIT 21''',(t['connectionId'],)).fetchall()
 if len(ca)>20:raise RuntimeError('CANN API rows capped; refine')
 out.append({'run':run,'window_ns':[start,end],'task':t,'decoded_string_fields':decoded,'cann_api_connection_candidates':[dict(x) for x in ca]})
print(json.dumps({'db':str(p),'samples':out,'scope':'EP8 3 interior captured target layer24 windows only',
 'limits':['shape or metadata pointer is not a tensor value','CANN connection may be shared across graph replays; no request or dynamic value association is inferred',
 'no device execution; absent fields here do not imply absent from all profiler outputs']},indent=2,ensure_ascii=False,default=str))
c.close()
