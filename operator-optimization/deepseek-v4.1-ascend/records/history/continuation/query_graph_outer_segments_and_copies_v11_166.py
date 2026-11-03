from pathlib import Path
from collections import defaultdict,Counter
from datetime import datetime,timezone
import json,hashlib,sqlite3,socket,time
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
A=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/profile-readonly-20261031')
def pinned(p,sha):
 x=p.stat();b=p.read_bytes();y=p.stat();assert hashlib.sha256(b).hexdigest()==sha;assert (x.st_size,x.st_mtime_ns)==(y.st_size,y.st_mtime_ns);return b
d=json.loads(pinned(A/'ep3-all-compute-origins-v9-v2.json','fdce699c6cf2fdf15b9ba85c5e9e5a5cd7893d2b34a0f69b0321e719a9ef6b9d'))
g=json.loads(pinned(A/'ep3-vector-overlay-and-graph-blocks-v10-v3.json','587b81f07b1b8c3057b9b7486bec6d58ce7aa946909d9385d8d4915bd6012c84'))
eager=json.loads(pinned(A/'ep3-hc-boundary-task-stack-evidence.json','cc35da2333fb92a982bacfacac4f6323567fc6b5f40e08c3ce7366944a53df05'))
tm={t['task_record_rowid']:t for t in d['tasks']};segments=[];definition_locations=defaultdict(Counter)
for e in g['graph_executions']:
 blocks=e['layer_blocks'];assigned={rid for b in blocks for rid in b['task_record_rowids']};groups=defaultdict(list)
 for rid in e['task_record_rowids']:
  if rid in assigned:continue
  t=tm[rid];where=[]
  if t['endNs']<=blocks[0]['startNs']:where=['BEFORE_FIRST_MARKER']
  elif t['startNs']>=blocks[-1]['endNs']:where=['AFTER_LAST_MARKER']
  else:
   for left,right in zip(blocks,blocks[1:]):
    if left['endNs']<=t['startNs'] and t['endNs']<=right['startNs']:
     where.append('GAP_'+str(left['observed_layer_block_ordinal'])+'_'+str(right['observed_layer_block_ordinal']))
  label=where[0] if len(where)==1 else 'UNRESOLVED_CROSSING'
  groups[label].append(t);definition_locations[t['globalTaskId']][label]+=1
 for label,ts in groups.items():
  ts.sort(key=lambda t:(t['startNs'],t['streamId'],t['taskId']))
  segments.append(dict(execution=e['ordinal'],location=label,records=len(ts),definition_ids=[t['globalTaskId'] for t in ts],
   type_sequence=[t['op_type'] for t in ts],tasks=ts))
samples=[]
for x in eager['items']:
 if '_L0_L1' not in x['label'] and '_L13_L14' not in x['label']:continue
 ts=[t for t in x['task_rows'] if t.get('kernel_name')]
 samples.append(dict(label=x['label'],records=len(ts),type_sequence=[t['op_type'] for t in ts],
  tasks=[dict(type=t['op_type'],input_shapes=t['input_shapes'],output_shapes=t['output_shapes'],source_frames=[f['frame'] for a in t.get('api_candidates',[]) for f in a.get('frames',[]) if '/engram/' in f['frame'] or '/models/deepseek_v41/model.py(' in f['frame']]) for t in ts]))
DB=Path(d['summary']['database']);before=DB.stat();h=hashlib.sha256()
with DB.open('rb') as f:
 for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
assert h.hexdigest()==d['summary']['database_sha256']
t0=time.monotonic();db=sqlite3.connect(DB.as_uri()+'?mode=ro&immutable=1',uri=True);db.row_factory=sqlite3.Row
db.execute('PRAGMA query_only=ON');db.set_progress_handler(lambda:1 if time.monotonic()-t0>100 else 0,100000)
tables=[r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")]
copy_tables=[t for t in tables if 'MEMCPY' in t.upper() or 'MEMORY' in t.upper()]
schemas={t:[dict(r) for r in db.execute('PRAGMA table_info("'+t+'")')] for t in copy_tables}
table_counts={t:db.execute('SELECT COUNT(*) FROM "'+t+'"').fetchone()[0] for t in copy_tables}
table_examples={t:[dict(r) for r in db.execute('SELECT * FROM "'+t+'" LIMIT 2')] for t in copy_tables}
copy_tasks=[dict(r) for r in db.execute('''SELECT t.rowid AS task_record_rowid,t.*,s.value AS task_type_name FROM TASK t JOIN STRING_IDS s ON s.id=t.taskType
 WHERE t.deviceId=3 AND (s.value LIKE '%MEMCPY%' OR s.value LIKE '%MEMSET%') ORDER BY t.startNs''')]
assert len(copy_tasks)<10000
for t in copy_tasks:
 hits=[e['ordinal'] for e in g['graph_executions'] if t['modelId']==e['model_id'] and e['control_pair_start_ns']<=t['startNs'] and t['endNs']<=e['control_pair_end_ns']]
 t['same_model_control_pair_candidates']=hits
db.close();after=DB.stat();assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
summary=dict(created_utc=datetime.now(timezone.utc).isoformat(),database_sha256=d['summary']['database_sha256'],database_stat_unchanged=True,
 outer_compute_records=len(segments) and sum(x['records'] for x in segments),segments=len(segments),
 first_execution_groups=[dict(location=x['location'],records=x['records'],type_sequence=x['type_sequence']) for x in segments if x['execution']==0],
 graph_definition_location_variants=dict(Counter(len(v) for v in definition_locations.values())),
 same_capture_source_bound_eager_windows=[dict(label=x['label'],records=x['records'],type_sequence=x['type_sequence']) for x in samples],
 memory_tables=copy_tables,memory_table_counts=table_counts,all_MEMCPY_MEMSET_TASK_rows=len(copy_tasks),
 memory_task_model_counts=dict(Counter(str(t['modelId']) for t in copy_tasks)),
 same_model_graph_memory_TASK_counts=dict(Counter(str(k) for t in copy_tasks for k in t['same_model_control_pair_candidates'])),
 no_target_source_read=True,no_device_or_inference_work=True,whole_modules_closed=0,overall_complete=False,
 limits=['Gap labels are exact device interval containment, not proof of producer/consumer/tensor identity.',
 'Repeated definitions may have different locations if timing overlaps; all such cases retained.',
 'Saved eager window source bindings define a semantic comparison, not a same-tensor or same-shape identity.',
 'MEMCPY TASK and memory table schemas expose only recorded fields. Missing pointers/bytes/values are not assumed available.',
 'Compute CSV coverage is separate from all noncompute copy/event/communication coverage.'])
out=A/'ep3-outer-segments-and-memory-tasks-v11.json';assert not out.exists()
full=dict(summary=summary,outer_segments=segments,definition_location_counts={str(k):dict(v) for k,v in definition_locations.items()},eager_source_bound_windows=samples,
 memory_schemas=schemas,memory_table_examples=table_examples,memory_TASK_rows=copy_tasks)
payload=json.dumps(full,ensure_ascii=False,indent=2)+'\n';out.write_text(payload)
public=dict(summary=summary,full_result=str(out),result_sha256=hashlib.sha256(payload.encode()).hexdigest(),memory_schemas=schemas,memory_table_examples=table_examples,
 first_outer_task_groups=[dict(location=x['location'],tasks=[{k:t[k] for k in ['globalTaskId','op_type','input_shapes','output_shapes','streamId']} for t in x['tasks']]) for x in segments if x['execution']==0],display_truncated=False)
while len(json.dumps(public,ensure_ascii=False).encode())>30000 and public['first_outer_task_groups']:
 public['first_outer_task_groups']=public['first_outer_task_groups'][:-1];public['display_truncated']=True
if len(json.dumps(public,ensure_ascii=False).encode())>30000:public['memory_table_examples']={};public['display_truncated']=True
assert len(json.dumps(public,ensure_ascii=False).encode())<=32000
print(json.dumps(public,ensure_ascii=False))
