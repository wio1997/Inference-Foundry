#!/usr/bin/env python3
"""Read one fixed saved EP0 profiler stack; pass --task-root and --db in stage."""
import argparse,hashlib,json,sqlite3
from decimal import Decimal
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--task-root',required=True);ap.add_argument('--db',required=True);a=ap.parse_args()
p=Path(a.task_root)/'diagnostic-20261029/target-layer-all-kernels-16rank.jsonl'
assert hashlib.sha256(p.read_bytes()).hexdigest()=='0493115eb0a8c9191297f32ee4f845dc69d5d60ec09fce6deb96ec1faeb9a3c4'
selected=[]
with p.open() as f:
 for line in f:
  x=json.loads(line)
  if x.get('kind')!='rank' or x['data']['ep']!=0:continue
  for l in x['data']['layers']:
   if l['layer']!=0 or not l['scheduler_digest'].startswith('7ddf7ae98a00'):continue
   for g in l['operator_groups']:
    if g['operator_name']=='aclnnConstantPadNd_PadV3AiCpu_PadV3':selected.extend(g['tasks'])
assert len(selected)==1
t=selected[0];assert t['csv']['Task ID']=='45438' and t['csv']['Stream ID']=='45'
assert int(Decimal(t['host_start_us'])*1000)==1793241331834541680 and t['host_pid']=='736' and t['host_tid']=='736'
assert t['flow_id']=='1793241333374500830'
c=sqlite3.connect(Path(a.db).resolve().as_uri()+'?mode=ro',uri=True)
row=c.execute('SELECT a.startNs,a.endNs,a.globalTid,a.connectionId,a.callchainId,s.value FROM PYTORCH_API a JOIN STRING_IDS s ON s.id=a.name WHERE a.startNs=? AND a.globalTid=? AND s.value=?',('1793241331834541680',3161095930592,'aclnnConstantPadNd')).fetchall()
assert len(row)==1 and row[0][3:5]==(496472,1198793)
stack=c.execute('SELECT pc.stackDepth,s.value FROM PYTORCH_CALLCHAINS pc LEFT JOIN STRING_IDS s ON s.id=pc.stack WHERE pc.id=? ORDER BY pc.stackDepth',(row[0][4],)).fetchall()
need=('functional.py(5418): pad','prepare_finalize.py(338): pad_and_split_input_ids','moe_comm_method.py(327): pad_and_split_input_ids','fused_topk_router.py(169): _compute_routing','model.py(1592): forward')
assert all(any(frame and n in frame for _,frame in stack) for n in need)
print(json.dumps({'scope':'one saved EP0 L0 host API linked by exact thread/time to previously matched flow','flow_id':t['flow_id'],'stream':t['csv']['Stream ID'],'task_id':t['csv']['Task ID'],'host_start_ns':str(row[0][0]),'host_end_ns':str(row[0][1]),'global_tid':row[0][2],'connection_id':row[0][3],'callchain_id':row[0][4],'selected_stack':[{'depth':d,'frame':n} for d,frame in stack for n in need if frame and n in frame],'limitations':['Historical installed source/binary and tensor storage identity not proved','Only this one API callchain is bound; no cross-clock time subtraction','No claim of path or performance for other 319 tasks']},indent=2))
