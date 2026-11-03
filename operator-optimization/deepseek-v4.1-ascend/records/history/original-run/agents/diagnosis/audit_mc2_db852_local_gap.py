#!/usr/bin/env python3
"""Read all saved TASK rows overlapping one EP3 same-stream local window."""
from pathlib import Path
import json,shlex,subprocess,hashlib
root=Path(__file__).resolve().parents[2];src=root/'agents/diagnosis/mc2-db852-extremes.json';d=json.loads(src.read_text())
x=next(x for x in d['rows'] if x['ep']==3 and x['layer']==15);n=next(x for x in d['last_matched_same_scope_neighbors'] if x['ep']==3)
base='/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029/runtime/diagnostic-traces'
code=r'''import json,sqlite3,pathlib,sys
from decimal import Decimal
base,x,n=sys.argv[1],json.loads(sys.argv[2]),json.loads(sys.argv[3]);folder=pathlib.Path(base)/x['rank']/'ASCEND_PROFILER_OUTPUT';files=list(folder.glob('ascend_pytorch_profiler_*.db'));assert len(files)==1
c=sqlite3.connect(files[0].as_uri()+'?mode=ro',uri=True);c.execute('PRAGMA query_only=ON')
start=int((Decimal(n['last_matched_start_us'])+Decimal(n['last_matched_duration_us']))*1000);end=int(Decimal(x['start_us'])*1000)
rows=c.execute('select t.globalTaskId,t.taskId,t.streamId,t.modelId,t.startNs,t.endNs,s.value,o.value,st.value from TASK t left join COMPUTE_TASK_INFO i on i.globalTaskId=t.globalTaskId left join STRING_IDS s on s.id=i.name left join STRING_IDS o on o.id=i.opType left join STRING_IDS st on st.id=t.taskType where t.streamId=? and ((t.startNs < ? and t.endNs > ?) or (t.startNs >= ? and t.startNs < ? and t.startNs=t.endNs)) order by t.startNs',(int(x['stream']),end,start,start,end)).fetchall()
schemas={t:c.execute('PRAGMA table_info('+t+')').fetchall() for t in ['TASK','MSTX_EVENTS']}
fullrows=[c.execute('select * from TASK where globalTaskId=?',(r[0],)).fetchall() for r in rows]
other=c.execute("select t.streamId,st.value,count(*) from TASK t left join STRING_IDS st on st.id=t.taskType where t.deviceId=? and ((t.startNs < ? and t.endNs > ?) or (t.startNs >= ? and t.startNs < ? and t.startNs=t.endNs)) group by t.streamId,st.value order by count(*) desc",(int(x['ep']),end,start,start,end)).fetchall()
names=[r[0] for r in c.execute("select name from sqlite_master where type='table' and (upper(name) like '%EVENT%' or upper(name) like '%WAIT%' or upper(name) like '%SYNC%') order by name")]
print(json.dumps({'gap_start_ns':start,'gap_end_ns':end,'rows':rows,'event_table_names':names,'other_stream_type_counts':other,'schemas':schemas,'fullrows':fullrows}));c.close()'''
cmd=shlex.join(['docker','exec','wio-vllm-v41-repro169-stage-20261028','python3','-c',code,base,json.dumps(x),json.dumps(n)])
r=subprocess.run(['/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh',cmd],capture_output=True,text=True,timeout=50,check=True)
o=json.loads(r.stdout);o['source_sha256']=hashlib.sha256(src.read_bytes()).hexdigest();o['target']={'ep':3,'layer':15,'stream':x['stream'],'task':x['task']}
(root/'agents/diagnosis/mc2-db852-ep3-gap.json').write_text(json.dumps(o,indent=2)+'\n')
print('gap',o['gap_end_ns']-o['gap_start_ns'],'ns; TASK rows',len(o['rows']));print('event tables:',o['event_table_names'])
for row in o['rows'][:25]:print(row)
