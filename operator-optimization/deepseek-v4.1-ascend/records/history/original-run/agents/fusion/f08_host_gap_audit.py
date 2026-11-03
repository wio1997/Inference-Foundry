#!/usr/bin/env python3
"""Read-only EP3 saved profiler DB: correlate six selected device tasks and host launches."""
import json,sqlite3,pathlib
P=pathlib.Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029/runtime/diagnostic-traces/dp0_pp0_tp3_dcp0_ep3_rank3_2128_20261029023525650_ascend_pt/ASCEND_PROFILER_OUTPUT/ascend_pytorch_profiler_3.db')
c=sqlite3.connect(P.as_uri()+'?mode=ro',uri=True);c.row_factory=sqlite3.Row
result={}
for label,ids in [('short',(13224,13225,13226)),('long',(13466,13467,13468))]:
 rows=[dict(x) for x in c.execute('select * from TASK where deviceId=3 and streamId=35 and taskId in (?,?,?) order by startNs',ids)];assert [x['taskId'] for x in rows]==list(ids)
 launches=[]
 for task in rows:
  a=[dict(x) for x in c.execute('select a.*,s.value as name_text from CANN_API a left join STRING_IDS s on s.id=a.name where connectionId=?',(task['connectionId'],))];assert len(a)==1 and a[0]['name_text']=='launch';launches.append(a[0])
 assert len({a['globalTid'] for a in launches})==1
 enclosing=[]
 for launch in launches:
  api=[dict(x) for x in c.execute("select a.startNs,a.endNs,s.value as name from CANN_API a join STRING_IDS s on s.id=a.name where a.globalTid=? and a.startNs<=? and a.endNs>=? and s.value like 'aclnn%'",(launch['globalTid'],launch['startNs'],launch['endNs']))]
  enclosing.append(api)
 gaps=[]
 for i in range(2):
  left,right=rows[i]['endNs'],rows[i+1]['startNs']; assert right>=left
  tasks=[dict(x) for x in c.execute('select * from TASK where deviceId=3 and startNs<? and endNs>? order by startNs',(right,left))]
  api=[dict(x) for x in c.execute('select a.startNs,a.endNs,s.value as name from CANN_API a left join STRING_IDS s on s.id=a.name where a.globalTid=? and a.startNs<? and a.endNs>? order by a.startNs',(launches[0]['globalTid'],right,left))]
  gaps.append({'start_ns':left,'end_ns':right,'duration_ns':right-left,'next_launch_start_minus_previous_task_end_ns':launches[i+1]['startNs']-left,'next_task_start_minus_launch_end_ns':right-launches[i+1]['endNs'],'recorded_device3_tasks':tasks,'same_host_thread_api':api})
 result[label]={'tasks':rows,'launches':launches,'enclosing_aclnn_apis':enclosing,'gaps':gaps}
print(json.dumps(result,indent=2))
