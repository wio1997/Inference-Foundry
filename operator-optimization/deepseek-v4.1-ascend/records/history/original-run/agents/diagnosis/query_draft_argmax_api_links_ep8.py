#!/usr/bin/env python3
"""Fixed, bounded EP8 TASK-to-profiler-API lookup; no NPU operations."""
import argparse, hashlib, json, sqlite3
from pathlib import Path
P=argparse.ArgumentParser(); P.add_argument('--db',required=True); P.add_argument('--anchors-json',required=True); a=P.parse_args()
expected_db='67d14f7f386658f8c8c5d3c9059dff9b0f1decb200eba621a3432fcf1e23de58'
expected_anchor='a20fd5761c9ca88b5161ea624c267c248342a21d230bb69a7f83ee589fdc0666'
p=Path(a.db).resolve(strict=True)
h=hashlib.sha256()
with p.open('rb') as f:
 for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
if h.hexdigest()!=expected_db: raise RuntimeError('unexpected DB hash')
if hashlib.sha256(a.anchors_json.encode()).hexdigest()!=expected_anchor: raise RuntimeError('unexpected anchor hash')
x=json.loads(a.anchors_json)
if len(x['intervals'])!=8 or len(x['matched_anchor_rows'])!=10: raise RuntimeError('unexpected anchor count')
con=sqlite3.connect(p.as_uri()+'?mode=ro',uri=True); con.row_factory=sqlite3.Row; con.execute('PRAGMA query_only=ON'); con.execute('PRAGMA temp_store=MEMORY')
def limited(sql,args,cap):
 rows=con.execute(sql,args).fetchmany(cap+1)
 return [dict(r) for r in rows[:cap]],len(rows)>cap
safe={'aten::embedding','aten::matmul','aten::mm','aten::linear','aten::add','aten::add_','aten::argmax','aten::slice','aten::copy_','aten::index_select','aten::to','aten::_to_copy','aten::as_strided','aten::view','aten::reshape'}
targets=[]
for g in x['intervals']:
 if g['ordinal'] not in (6,7) or g['gap_index'] not in (1,2,3,4) or len(g['events'])!=10: raise RuntimeError('unexpected interval')
 for e in g['events']:
  if e['model']!='4294967295' or e['stream']!='47': raise RuntimeError('unexpected model/stream')
  targets.append({'ordinal':g['ordinal'],'gap':g['gap_index'],'type':e['type'],'model':int(e['model']),'stream':int(e['stream']),'task':int(e['task_id']),'start_ns':g['start_ns']+e['start_rel_ns'],'end_ns':g['start_ns']+e['end_rel_ns']})
for e in x['matched_anchor_rows']:
 if e['ordinal'] not in (6,7) or e['model']!='4294967295' or e['stream']!='47' or e['type']!='ArgMaxV2': raise RuntimeError('unexpected anchor')
 targets.append({'ordinal':e['ordinal'],'gap':None,'type':e['type'],'model':int(e['model']),'stream':int(e['stream']),'task':int(e['task_id']),'start_ns':e['start_ns'],'end_ns':e['end_ns']})
if len(targets)!=90 or len({(e['model'],e['stream'],e['task'],e['start_ns'],e['end_ns']) for e in targets})!=90: raise RuntimeError('targets are not unique')
result=[]
for t in targets:
 tasks,task_cut=limited('SELECT connectionId,globalTaskId FROM TASK WHERE modelId=? AND streamId=? AND taskId=? AND startNs=? AND endNs=?',(t['model'],t['stream'],t['task'],t['start_ns'],t['end_ns']),3)
 links=[]
 for task in tasks:
  cid=task['connectionId']; cann,ccut=limited('SELECT connectionId FROM CANN_API WHERE connectionId=?',(cid,),3) if cid is not None else ([],False)
  groups,gcut=limited('SELECT id FROM CONNECTION_IDS WHERE connectionId=?',(cid,),20) if cid is not None else ([],False)
  api=[]
  for group in groups:
   candidates,cut=limited('SELECT p.sequenceNumber,p.callchainId,n.value AS api_name FROM PYTORCH_API p LEFT JOIN STRING_IDS n ON n.id=p.name WHERE p.connectionId=?',(group['id'],),20)
   clean=[]
   for c in candidates:
    name=c['api_name']; clean.append({'name':name if name in safe else None,'other_name_sha256':None if name in safe or name is None else hashlib.sha256(str(name).encode()).hexdigest(),'sequence_number':c['sequenceNumber'],'callchain_present':c['callchainId'] is not None})
   api.append({'connection_set_id':group['id'],'candidate_count_shown':len(clean),'truncated':cut,'candidates':clean})
  links.append({'connection_id':cid,'global_task_id':task['globalTaskId'],'cann_count_shown':len(cann),'cann_truncated':ccut,'connection_set_count_shown':len(groups),'sets_truncated':gcut,'pytorch_via_sets':api})
 result.append({'target':t,'task_count_shown':len(tasks),'task_truncated':task_cut,'links':links})
con.close()
print(json.dumps({'scope':'EP8 fixed 80 interval events and 10 bounding ArgMax only','db_sha256':expected_db,'anchors_sha256':expected_anchor,'rows':result,'limitations':['All links are candidate export associations, not proof of tensor, request, Python caller, causality, or unique ID across replay.','Only fixed EP8 events and capped matches queried; explicit truncation means counts are not total.','No cross-clock subtraction, user inputs, token values, weights, shapes or stack contents returned.','No NPU or target-code operations.']},ensure_ascii=False,indent=2))
