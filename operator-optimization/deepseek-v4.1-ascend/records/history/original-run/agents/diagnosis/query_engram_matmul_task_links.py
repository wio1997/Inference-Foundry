#!/usr/bin/env python3
"""Read EP8 profiler database for six specific matmuls, no device execution."""
import argparse,hashlib,json,sqlite3
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--task-root',required=True);ap.add_argument('--db',required=True);a=ap.parse_args()
T=Path(a.task_root); wp=T/'captured-layer-attribution/module-call-metrics.json';ep=T/'agents/diagnosis/engram-boundary-controls.json'
windows=json.loads(wp.read_text());samples=json.loads(ep.read_text())['samples']
src=[r for r in samples if '_ep8_' in r['rank'] and r['after_layer'] in (0,13)];assert len(src)==6
targets=[]
for r in src:
 x=[z for z in windows if z['rank']==r['rank'] and z['run_in_model']==r['run'] and z['layer']==r['after_layer']];assert len(x)==1
 events=[z for z in r['events'] if z['type']=='MatMulV2' and z['input_shapes']=='"3,6144;25600,6144"' and z['input_dtypes']=='DT_BF16;DT_BF16'];assert len(events)==1
 e=events[0];base=x[0]['window_end_ns'];targets.append({'run':r['run'],'after_layer':r['after_layer'],'stream':int(e['stream']),'model':int(e['model']),
     'start_ns':base+e['event_start_rel_ns'],'end_ns':base+e['event_end_rel_ns']})
path=Path(a.db).resolve(strict=True);h=hashlib.sha256()
with path.open('rb') as f:
 for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
dbsha=h.hexdigest()
assert dbsha=='67d14f7f386658f8c8c5d3c9059dff9b0f1decb200eba621a3432fcf1e23de58'
c=sqlite3.connect(path.as_uri()+'?mode=ro',uri=True);c.row_factory=sqlite3.Row;c.execute('PRAGMA query_only=ON');c.execute('PRAGMA temp_store=MEMORY')
tables={r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}
assert {'TASK','COMPUTE_TASK_INFO','STRING_IDS','CANN_API','PYTORCH_API','CONNECTION_IDS'}<=tables
def bounded(sql,args,limit=100):
 a=[dict(r) for r in c.execute(sql,args).fetchmany(limit+1)]
 return {'rows':a[:limit],'truncated':len(a)>limit,'limit':limit}
def name(i):
 if i is None:return None
 a=c.execute('SELECT value FROM STRING_IDS WHERE id=?',(i,)).fetchone()
 return a[0] if a else None
out=[]
for t in targets:
 matched=bounded('''SELECT t.* FROM TASK t JOIN COMPUTE_TASK_INFO i ON i.globalTaskId=t.globalTaskId
 LEFT JOIN STRING_IDS s ON s.id=i.opType
 WHERE t.streamId=? AND t.modelId=? AND CAST(t.startNs AS INTEGER)=? AND CAST(t.endNs AS INTEGER)=?
 AND s.value='MatMulV2' ''',(t['stream'],t['model'],t['start_ns'],t['end_ns']),10)
 if matched['truncated'] or len(matched['rows'])!=1:raise RuntimeError('TASK exact start/end/model/stream/type not unique: '+str(t))
 task=matched['rows'][0];gid=task['globalTaskId'];conn=task['connectionId']
 info=c.execute('SELECT * FROM COMPUTE_TASK_INFO WHERE globalTaskId=?',(gid,)).fetchone();assert info is not None
 info=dict(info)
 decoded={k:name(info.get(k)) for k in ('name','opType','inputFormats','inputDataTypes','inputShapes','outputFormats','outputDataTypes','outputShapes','attrInfo')}
 replays=bounded('SELECT startNs,endNs,streamId,taskId,modelId,connectionId FROM TASK WHERE globalTaskId=? ORDER BY CAST(startNs AS INTEGER)',(gid,),30)
 cann=bounded('''SELECT a.startNs,a.endNs,a.type,a.globalTid,a.connectionId,s.value AS name FROM CANN_API a
 LEFT JOIN STRING_IDS s ON s.id=a.name WHERE a.connectionId=?''',(conn,),30)
 connsets=bounded('SELECT id,connectionId FROM CONNECTION_IDS WHERE connectionId=? ORDER BY id',(conn,),30)
 torch=[]; ids=[q['id'] for q in connsets['rows']]
 if not connsets['truncated']:
  for ident in ids:
   torch.append({'connection_set_id':ident,'matches':bounded('''SELECT a.startNs,a.endNs,a.globalTid,a.connectionId,a.sequenceNumber,a.fwdThreadId,a.callchainId,a.type,
       s.value AS name FROM PYTORCH_API a LEFT JOIN STRING_IDS s ON s.id=a.name WHERE a.connectionId=? ORDER BY CAST(a.startNs AS INTEGER)''',(ident,),30)})
 direct=bounded('''SELECT a.startNs,a.endNs,a.globalTid,a.connectionId,a.sequenceNumber,a.callchainId,a.type,s.value AS name
  FROM PYTORCH_API a LEFT JOIN STRING_IDS s ON s.id=a.name WHERE a.connectionId=? ORDER BY CAST(a.startNs AS INTEGER)''',(conn,),30)
 n=c.execute('SELECT COUNT(*) FROM TASK WHERE connectionId=?',(conn,)).fetchone()[0]
 out.append({'target':t,'exact_task':task,'compute_fields_raw':info,'compute_decoded_fields':decoded,'replays_by_globalTaskId':replays,
  'tasks_sharing_connectionId_count':n,'cann_api_exact_connectionId':cann,'connection_id_set_lookup':connsets,
  'pytorch_via_connection_set':torch,'pytorch_direct_same_numeric_id':direct})
chains={table:[dict(x) for x in c.execute('PRAGMA table_info("'+table.replace('"','""')+'")')] for table in sorted(tables) if any(s in table.upper() for s in ('CALLCHAIN','STACK','FRAME'))}
c.close()
print(json.dumps({'scope':'Six uniquely timestamp-matched large MatMulV2 tasks on EP8, 3 target samples x before layers 1 and 14',
 'db_sha256':dbsha,'window_sha256':hashlib.sha256(wp.read_bytes()).hexdigest(),'interval_sha256':hashlib.sha256(ep.read_bytes()).hexdigest(),
 'stack_related_table_schemas':chains,'samples':out,
 'limitations':['Exact device-time/stream/model match identifies task row only, not source-level tensor or execution semantics.',
  'Repeated globalTaskId is graph/replay evidence only in these selected samples; attribution and cross-process relationships require documented semantics.',
  'Connection and Python rows are candidate associations, not an automatically proven direct tensor/launch relationship; compare clock domains only with documented calibration.',
  'Missing exact records does not prove missing host launch or producer; graph replay and exporter configuration may omit them.',
  'Task input shapes/dtypes are metadata, not actual tensor values or weight identity.']},ensure_ascii=False,indent=2,default=str))
