"""Use exact CPU nesting plus vendor connection IDs to bind replay/model envelopes."""
import collections,concurrent.futures,json
import sqlite3
from pathlib import Path
C=Path('/data/tiankuan/wio/migration-171-20261102/builtin-cohort-v164-228ad25fcf9b')
H=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/runtime/analysis/builtin-v164-official-db')
def union_ns(intervals):
 total=0;end=None
 for a,b in sorted(intervals):
  if end is None or a>end:total+=max(0,b-a);end=b
  elif b>end:total+=b-end;end=b
 return total
def supplements():
 scopes={};ends={};bindings=[]
 for p in (C/'runtime/diagnostics/supplement').glob('*.jsonl'):
  for line in p.open():
   r=json.loads(line);kind=r.get('kind')
   if kind=='scope_begin':
    marker=r.get('identity',{}).get('profile_marker')
    if marker:
     assert marker not in scopes
     scopes[marker]=r
   elif kind=='scope_end':ends[(r['process_uuid'],r['scope_id'])]=r
   elif kind=='wrapper_graph_binding':bindings.append(r)
 return scopes,ends,bindings
SCOPES,ENDS,BINDINGS=supplements()
def one(db):
 c=sqlite3.connect('file:'+str(db)+'?mode=ro&immutable=1',uri=True);c.row_factory=sqlite3.Row
 s=dict(c.execute('select id,value from STRING_IDS'));tasks=[dict(r) for r in c.execute('select rowid,* from TASK')]
 compute={r['globalTaskId']:dict(r) for r in c.execute('select * from COMPUTE_TASK_INFO')}
 markers=[]
 for r in c.execute('select rowid,* from PYTORCH_API'):
  name=s.get(r['name'],'')
  if name.startswith('wio_shared/') and name.endswith('/replay'):
   markers.append({**dict(r),'marker':name,'startNs':int(r['startNs']),'endNs':int(r['endNs'])})
 apis={r['connectionId']:dict(r) for r in c.execute('select * from CANN_API') if s.get(r['name'])=='aclmdlRIExecuteAsync'}
 bridges=[];issues=[]
 for t in tasks:
  if s.get(t['taskType'])!='MODEL_EXECUTE':continue
  api=apis.get(t['connectionId'])
  if not api:issues.append({'task_rowid':t['rowid'],'reason':'NO_EXACT_RUNTIME_CONNECTION'});continue
  matches=[r for r in markers if r['globalTid']==api['globalTid'] and r['startNs']<=api['startNs']<=api['endNs']<=r['endNs']]
  if len(matches)!=1:issues.append({'task_rowid':t['rowid'],'reason':'NO_UNIQUE_SAME_THREAD_REPLAY_NESTING','matches':len(matches)});continue
  marker=matches[0];scope=SCOPES.get(marker['marker'])
  end=ENDS.get((scope['process_uuid'],scope['scope_id'])) if scope else None
  if not scope or not end or scope['identity'].get('capture_boundary_state')!='CAPTURE_END_RETURNED':
   issues.append({'task_rowid':t['rowid'],'reason':'MISSING_CLOSED_CAPTURED_REPLAY_SCOPE'});continue
  # Keep the raw scope-end result visible for review;never assume return=GPU completion.
  waits=[r for r in tasks if s.get(r['taskType'])=='MODEL_WAIT_COMPLETE' and
         all(r[k]==t[k] for k in ['connectionId','deviceId','globalPid','streamId','modelId'])]
  if len(waits)!=1:issues.append({'task_rowid':t['rowid'],'reason':'NO_UNIQUE_VENDOR_MODEL_COMPLETION_RECORD'});continue
  w=waits[0]
  if t['modelId']==4294967295 or not(t['startNs']<=w['endNs']):
   issues.append({'task_rowid':t['rowid'],'reason':'INVALID_MODEL_OR_ENVELOPE'});continue
  bridges.append({'profile':db.parent.parent.name,'process_uuid':scope['process_uuid'],'pid':scope['pid'],'tid':scope['tid'],
    'profile_marker':marker['marker'],'CPU_marker_rowid':marker['rowid'],'CPU_global_tid':marker['globalTid'],
    'CPU_marker_start_ns':marker['startNs'],'CPU_marker_end_ns':marker['endNs'],
    'runtime_api':{**api,'name_text':s[api['name']]},'graph_identity':scope['identity'],'scope_end_record':end,
    'native_model_id':t['modelId'],'device_id':t['deviceId'],'native_global_pid':t['globalPid'],
    'launch_stream_id':t['streamId'],'launch_task_id':t['taskId'],'launch_task_rowid':t['rowid'],
    'completion_task_rowid':w['rowid'],'envelope_start_ns':t['startNs'],'envelope_end_ns':w['endNs']})
 assignment=collections.Counter();families=collections.Counter()
 for b in bridges:b['compute_intervals']=[];b['operators']={};b['assigned_compute_tasks']=0
 for t in tasks:
  x=compute.get(t['globalTaskId'])
  if not x or t['modelId']==4294967295:continue
  matches=[b for b in bridges if b['native_model_id']==t['modelId'] and b['device_id']==t['deviceId'] and b['native_global_pid']==t['globalPid'] and
    b['envelope_start_ns']<=t['startNs']<=t['endNs']<=b['envelope_end_ns']]
  assignment['unique' if len(matches)==1 else 'missing' if not matches else 'ambiguous']+=1
  if len(matches)!=1:continue
  b=matches[0];op=s.get(x['opType'],'UNKNOWN');g=b['operators'].setdefault(op,{'count':0,'sum_duration_ns':0})
  g['count']+=1;g['sum_duration_ns']+=t['endNs']-t['startNs'];b['assigned_compute_tasks']+=1
  b['compute_intervals'].append((t['startNs'],t['endNs']));families[op]+=1
 for b in bridges:
  intervals=b.pop('compute_intervals');b['compute_active_union_ns']=union_ns(intervals);b['envelope_duration_ns']=b['envelope_end_ns']-b['envelope_start_ns']
 c.close()
 return {'profile':db.parent.parent.name,'replay_markers':len(markers),'bridges':bridges,'issues':issues,'captured_compute_assignment':dict(assignment),'captured_families':dict(families)}
if __name__=='__main__':
 dbs=sorted((H/'profiles').glob('*/ASCEND_PROFILER_OUTPUT/ascend_pytorch_profiler*.db'));assert len(dbs)==32
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(one,dbs))
 counts=collections.Counter()
 for r in rows:counts.update(r['captured_compute_assignment'])
 out={'status':'REPLAY_SCOPE_VENDOR_MODEL_ENVELOPE_BRIDGE','profiles':rows,'scope':'New171 v164 cohort only',
  'join_contract':['Exact same vendor DB namespace.','Same-thread Runtime API fully nested in one exact replay record_function scope.','Marker name exactly equals supplementary process UUID/scope identity.','Runtime connection ID exactly matches MODEL_EXECUTE and one MODEL_WAIT_COMPLETE with same model/device/pid/stream.',
   'Child task assignment requires same model/device/process and unique full containment in vendor execution envelope;no nearest-time fallback.'],
  'bridges':sum(len(r['bridges']) for r in rows),'captured_compute_assignment':dict(counts),
  'limits':['CPU scope return is not GPU completion.','Envelope membership is not each captured operator source/operand/reader mapping.','No original169/166history backfill or physical-MFU proof.','Native model IDs are local to process/device/lifetime,not a global graph identity.'],
  'historical_native_or_phase_promotions':0,'whole_modules_closed':0,'overall_complete':False}
 (H/'REPLAY-MODEL-BRIDGE-v166.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps({'status':out['status'],'profiles':len(rows),'bridges':out['bridges'],'assignment':dict(counts),'issues':sum(len(r['issues']) for r in rows)}))
