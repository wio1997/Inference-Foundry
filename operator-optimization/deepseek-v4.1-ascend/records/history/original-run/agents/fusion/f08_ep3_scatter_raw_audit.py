#!/usr/bin/env python3
import pathlib,ijson,json,hashlib,sqlite3
from decimal import Decimal as D
p=pathlib.Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029/runtime/diagnostic-traces/dp0_pp0_tp3_dcp0_ep3_rank3_2128_20261029023525650_ascend_pt/ASCEND_PROFILER_OUTPUT')
targets=[('short',D('1793241328966065.515'),D('16.68'),13226),('long',D('1793241329714147.715'),D('265.06'),13468)]
results={x[0]:{'overlap':[],'same_task_id':[]} for x in targets};flow=[]
with (p/'trace_view.json').open('rb') as fp:
 for e in ijson.items(fp,'item'):
  if not isinstance(e,dict):continue
  if e.get('ph')=='X' and str(e.get('pid'))=='2179523' and str(e.get('tid'))=='35':
   start=D(str(e.get('ts',0))); end=start+D(str(e.get('dur',0))); args=e.get('args',{});
   for name,t,d,tid in targets:
    if start<t+d and end>t:results[name]['overlap'].append(e)
    if str(args.get('Task Id',args.get('task_id',args.get('Task ID',''))))==str(tid):results[name]['same_task_id'].append(e)
  if str(e.get('id')) in ('1793241328966065515','1793241329714147715'):flow.append(e)
db=p/'ascend_pytorch_profiler_3.db'
con=sqlite3.connect(db.as_uri()+'?mode=ro',uri=True);con.row_factory=sqlite3.Row
for name,t,d,tid in targets:
 rows=[dict(r) for r in con.execute('select * from TASK where streamId=? and taskId=?',(35,tid))];results[name]['db_task_rows']=rows
print(json.dumps({'results':results,'flow':flow,'note':'exact target track only; overlap X includes target, not evidence of full model activity; db task IDs can be reused'},default=str))
