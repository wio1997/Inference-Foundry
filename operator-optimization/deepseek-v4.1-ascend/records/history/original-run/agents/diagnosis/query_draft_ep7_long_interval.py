#!/usr/bin/env python3
"""EP7 one predetermined draft interval: all overlapping CSV + device tasks."""
import argparse,csv,hashlib,json,sqlite3
from decimal import Decimal
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--task-root',required=True);ap.add_argument('--db',required=True);ap.add_argument('--csv',required=True);a=ap.parse_args()
T=Path(a.task_root);db=Path(a.db).resolve(strict=True);cp=Path(a.csv).resolve(strict=True);prior=T/'agents/diagnosis/draft-query-metadata-ep7.json'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
expected={db:'8b8e72a61e2521f0d91050bc53ab852ddf26dd4e40e2d5ec5d3973652fd807c6',cp:'9aa04e4cbbe7edc2160befcec0b133fdaf3b3e2571a1d2f1d7c92c6998b9bdc1',prior:'91bd06a12087247de6265ae5c5db915af8f86973d9ff78089b22e973b9545cf0'}
for p,h in expected.items():
 if sha(p)!=h:raise RuntimeError('hash mismatch: '+p.name)
d=json.loads(prior.read_text());match=[x for x in d['outputs'] if x['window']['label']=='v3_first_unclassified_no_exported_smla']
if len(match)!=1 or not match[0]['selected_events_complete']:raise RuntimeError('target selection missing or incomplete')
events=[x['target'] for x in match[0]['selected_events']]
left=[x for x in events if x['type']=='ZerosLike' and x['stream']=='47' and x['input_shapes']=='"1,64,512"']
right=[x for x in events if x['type']=='MatMulV2' and x['stream']=='47' and x['input_shapes']=='"5,1024;5120,1024"']
if len(left)!=1 or len(right)!=1:raise RuntimeError('target anchors not unique')
start=left[0]['end_ns'];end=right[0]['start_ns']
if not start<end or end-start!=2252635:raise RuntimeError('unexpected fixed interval')
def union(intervals):
 ints=sorted((max(start,a),min(end,b)) for a,b in intervals if a<end and b>start)
 total=0;finish=start
 for a,b in ints:
  if b>finish:total+=b-max(a,finish);finish=b
 return total
csvrows=[]
with cp.open(newline='') as f:
 for r in csv.DictReader(f):
  a=int(Decimal(r['Start Time(us)'].strip())*1000);b=a+int(Decimal(r['Duration(us)'].strip())*1000)
  if a<end and b>start:
   csvrows.append({'type':r['Type'],'model':r['Model ID'],'stream':r['Stream ID'],'task':r['Task ID'],
    'start_ns':a,'end_ns':b,'input_shapes':r['Input Shapes'],'output_shapes':r['Output Shapes']})
if len(csvrows)>500:raise RuntimeError('CSV overlap exceeds 500 row cap')
numeric=[r for r in csvrows if r['stream'].isdigit()];unknown=[r for r in csvrows if not r['stream'].isdigit()]
c=sqlite3.connect(db.as_uri()+'?mode=ro',uri=True);c.row_factory=sqlite3.Row;c.execute('PRAGMA query_only=ON');c.execute('PRAGMA temp_store=MEMORY')
tasks=[dict(x) for x in c.execute('''SELECT t.startNs,t.endNs,t.streamId,t.modelId,t.taskId,t.globalTaskId,t.connectionId,
   t.taskType,s.value AS task_type_name FROM TASK t LEFT JOIN STRING_IDS s ON s.id=t.taskType
   WHERE t.deviceId=7 AND t.startNs<? AND t.endNs>? ORDER BY t.startNs,t.streamId LIMIT 1001''',(end,start))]
if len(tasks)>1000:raise RuntimeError('TASK overlap exceeds 1000 row cap')
print(json.dumps({'scope':'EP7 fixed half-open interval between selected ZerosLike end and next selected MatMul start',
 'source_sha256':{p.name:h for p,h in expected.items()},'start_ns':start,'end_ns':end,'duration_ns':end-start,
 'left_anchor':left[0],'right_anchor':right[0], 'csv_rows_all_streams':csvrows,'device_task_rows_all_streams':tasks,
 'csv_numeric_stream_union_ns':union([(r['start_ns'],r['end_ns']) for r in numeric]),
 'csv_nonnumeric_stream_union_ns':union([(r['start_ns'],r['end_ns']) for r in unknown]),
 'csv_all_rows_union_ns':union([(r['start_ns'],r['end_ns']) for r in csvrows]),
 'limits':['All rows are temporal overlaps, not necessarily part of the draft block or same request.',
 'CSV N/A streams can be duplicate or aggregate records; do not sum numeric and N/A unions.',
 'CSV gaps are not device-idle proof; DB TASK, profiler collection, and sources may be incomplete or have different meaning.',
 'Event/notify wait labels or temporal overlap alone do not establish producer, dependency, causality or network duration.',
 'Only one rank and one interval; no cross-rank clock arithmetic, no NPU workload.']},ensure_ascii=False,indent=2));c.close()
