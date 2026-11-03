#!/usr/bin/env python3
"""Read the existing rank-3 SQLite only; compare the gap with saved host/overlap records."""
import hashlib, json, shlex, subprocess
from pathlib import Path
root=Path(__file__).resolve().parents[2]
source=root/'agents/diagnosis/mc2-db852-ep3-gap.json'
gap=json.loads(source.read_text())
code=r'''import sqlite3,pathlib,json,sys
start,end=map(int,sys.argv[1:]);base=pathlib.Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029/runtime/diagnostic-traces/dp0_pp0_tp3_dcp0_ep3_rank3_2128_20261029023525650_ascend_pt/ASCEND_PROFILER_OUTPUT')
files=list(base.glob('ascend_pytorch_profiler_*.db'));assert len(files)==1
c=sqlite3.connect(files[0].as_uri()+'?mode=ro',uri=True);c.execute('pragma query_only=on');c.row_factory=sqlite3.Row
out={}
for table in ['SESSION_TIME_INFO']:
 out[table]=[dict(x) for x in c.execute('select * from '+table)]
out['extents']={}
for table in ['TASK','CANN_API','PYTORCH_API','COMMUNICATION_OP','MSTX_EVENTS','OVERLAP_ANALYSIS']:
 lo,hi=c.execute('select min(startNs),max(endNs) from '+table).fetchone();n=c.execute('select count(*) from '+table+' where startNs<? and endNs>?',(end,start)).fetchone()[0];out['extents'][table]={'min_start_ns':lo,'max_end_ns':hi,'overlapping_rows':n}
for table in ['CANN_API','PYTORCH_API']:
 out[table]=[dict(x) for x in c.execute('select a.startNs-? as startOffsetNs,a.endNs-? as endOffsetNs,a.globalTid,a.connectionId,s.value as name from '+table+' a left join STRING_IDS s on s.id=a.name where a.startNs<? and a.endNs>? order by a.startNs',(start,start,end,start))]
out['overlap']=[dict(x) for x in c.execute('select a.startNs-? as startOffsetNs,a.endNs-? as endOffsetNs,a.type,e.name from OVERLAP_ANALYSIS a left join ENUM_OVERLAP_ANALYSIS_TYPE e on e.id=a.type where a.startNs<? and a.endNs>?',(start,start,end,start))]
out['target_tasks']=[dict(x) for x in c.execute('select * from TASK where streamId=47 and taskId in (40490,40491) and startNs between ? and ? order by startNs',(start-10000000,end+10000000))]
out['task_launches']=[dict(x) for x in c.execute('select a.startNs-? as startOffsetNs,a.endNs-? as endOffsetNs,a.globalTid,a.connectionId,s.value as name from CANN_API a left join STRING_IDS s on s.id=a.name where a.connectionId in (95811,95824) order by a.startNs',(start,start))]
print(json.dumps(out));c.close()'''
cmd=shlex.join(['docker','exec','wio-vllm-v41-repro169-stage-20261028','python3','-c',code,str(gap['gap_start_ns']),str(gap['gap_end_ns'])])
r=subprocess.run(['/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh',cmd],capture_output=True,text=True,timeout=50,check=True)
o=json.loads(r.stdout);o['gap_start_ns']=gap['gap_start_ns'];o['gap_end_ns']=gap['gap_end_ns'];o['gap_input_sha256']=hashlib.sha256(source.read_bytes()).hexdigest()
(root/'agents/diagnosis/mc2-db852-ep3-host-gap.json').write_text(json.dumps(o,indent=2)+'\n')
print('gap ns',o['gap_end_ns']-o['gap_start_ns'],'overlap',o['overlap']);print('CANN launches',o['task_launches']);print('table extents',o['extents'])
