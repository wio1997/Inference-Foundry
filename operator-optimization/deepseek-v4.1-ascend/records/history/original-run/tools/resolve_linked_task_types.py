"""Resolve per-database task-type IDs through its own string table."""
from pathlib import Path
from collections import defaultdict,Counter
import json,sqlite3
T=Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41')
ROOT=Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-20261028/diagnostic/runtime/diagnostic-traces')
O=T/'graph-linked-task-type-resolution';O.mkdir(exist_ok=False)
calls=json.loads((T/'graph-submit-task-link-validation/call_results.json').read_text());ranks=defaultdict(set)
for x in calls:
 for task in x['linked_tasks']:ranks[x['rank']].add(task[5])
resolved=[]
for rank,ids in ranks.items():
 db=next((ROOT/rank/'ASCEND_PROFILER_OUTPUT').glob('ascend_pytorch_profiler_*.db'))
 c=sqlite3.connect('file:'+str(db)+'?mode=ro',uri=True);c.execute('PRAGMA query_only=ON')
 for i in sorted(ids):
  rows=list(c.execute('SELECT id,value FROM STRING_IDS WHERE id=?',(i,)))
  resolved.append({'rank':rank,'task_type_id':i,'string_table_matches':rows})
 c.close()
names=Counter()
for call in calls:
 for task in call['linked_tasks']:
  choices=[r for r in resolved if r['rank']==call['rank'] and r['task_type_id']==task[5]]
  assert len(choices)==1
  vals=choices[0]['string_table_matches'];names[str(vals[0][1]) if len(vals)==1 else 'UNRESOLVED']+=1
doc={'status':'TASK_TYPE_STRING_TABLE_LOOKUP_ONLY','linked_call_count':len(calls),'resolved_name_counts':dict(names),
     'resolved_by_rank':resolved,
     'warning':'A string-table match is not by itself proof of full runtime semantics or causality.',
     'no_device_work':True}
assert len(json.dumps(doc).encode())<32768
(O/'summary.json').write_text(json.dumps(doc,indent=2)+'\n');print(json.dumps(doc,indent=2))
