#!/usr/bin/env python3
"""Inspect attrInfo of two exact known graph MC2 tasks, no HCCL data."""
import argparse,hashlib,json,re,sqlite3
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--task-root',required=True);ap.add_argument('--trace-root',required=True);a=ap.parse_args()
T=Path(a.task_root);R=Path(a.trace_root);prior=T/'agents/diagnosis/16rank-mc2-other-stream.json';manifest=T/'evidence/used-raw-source-sha256.json'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
for p,h in [(prior,'d423aa676cf7deb950321d3d32cf77845579589ac7ae4626714a82ec034f40cb'),(manifest,'cd5bac3b56c51fa2d7ce13199523ff9f88b94c4edf4c74131d4aedbdf2fc7c2a')]:
 if sha(p)!=h:raise RuntimeError('input mismatch')
x=json.loads(prior.read_text());m=json.loads(manifest.read_text());ranks=['dp0_pp0_tp0_dcp0_ep0_rank0_728_20261028020238584_ascend_pt','dp1_pp0_tp0_dcp0_ep8_rank0_727_20261028020238580_ascend_pt'];out=[]
for rank in ranks:
 q=[s for s in x['samples'] if s['rank']==rank and s['run']==1]
 if len(q)!=1:raise RuntimeError('ambiguous prior')
 q=q[0];files=[R/z['relative_path'] for z in m['records'] if z['relative_path'].startswith(rank+'/ASCEND_PROFILER_OUTPUT/ascend_pytorch_profiler_') and z['relative_path'].endswith('.db')]
 if len(files)!=1:raise RuntimeError('ambiguous db')
 db=files[0];expected=next(z['sha256'] for z in m['records'] if z['relative_path']==str(db.relative_to(R)))
 if sha(db)!=expected:raise RuntimeError('db identity mismatch')
 c=sqlite3.connect(db.resolve().as_uri()+'?mode=ro',uri=True);c.row_factory=sqlite3.Row;c.execute('PRAGMA query_only=ON')
 tasks=[dict(z) for z in c.execute('''SELECT t.globalTaskId,t.connectionId,t.startNs,t.endNs,t.modelId,t.streamId,t.taskId,i.attrInfo,s.value AS op_type FROM TASK t JOIN COMPUTE_TASK_INFO i ON i.globalTaskId=t.globalTaskId LEFT JOIN STRING_IDS s ON s.id=i.opType WHERE t.startNs=? AND t.endNs=? AND t.modelId=? AND t.streamId=? LIMIT 4''',(q['mc2_start_ns'],q['mc2_end_ns'],int(q['model']),int(q['main_stream'])))]
 if len(tasks)!=1 or tasks[0]['op_type']!='DispatchFFNCombine':raise RuntimeError('task identity mismatch')
 t=tasks[0];attr=t['attrInfo'];values=[]
 if isinstance(attr,int):values=[r[0] for r in c.execute('SELECT value FROM STRING_IDS WHERE id=? LIMIT 3',(attr,))]
 if len(values)>1:raise RuntimeError('attr string ID ambiguous')
 raw=values[0] if values and isinstance(values[0],str) else None
 entries=[]
 if raw is not None:
  if len(raw)>10000:raise RuntimeError('attribute string exceeds bound')
  for item in raw.split('|'):
   name,sep,value=item.partition(':')
   if not sep:continue
   safe=name if re.fullmatch(r'attr_[0-9]{1,3}|group|group_name|world_size|rank_size',name,re.I) else 'UNRECOGNIZED_LABEL'
   entry={'label':safe,'value_sha256':hashlib.sha256(value.encode()).hexdigest(),'numeric':int(value) if safe.lower() in ('world_size','rank_size') and re.fullmatch(r'[0-9]{1,6}',value) else None,'value_length':len(value)}
   entries.append(entry)
 out.append({'rank':rank,'run':1,'db_sha256':expected,'task':t,'attribute_resolved':raw is not None,'attribute_sha256':hashlib.sha256(raw.encode()).hexdigest() if raw is not None else None,'entries':entries})
 c.close()
print(json.dumps({'scope':'Two exact layer0 MC2 tasks and their COMPUTE_TASK_INFO attrInfo; no communication table or new capture','prior_sha256':sha(prior),'samples':out,'limits':['Profiler attr_N-to-operator-ABI index mapping not established: do not infer group or world size by index alone.','Same attribute hash does not establish actual communicator object, participant mapping, same invocation, or request.','Graph globalTaskId and attributes may be reused; exact start/end identifies sample but not dynamic replay contents.','Nonmatching/unresolved attribute is not proof of no HCCL communicator.','Raw attribute strings withheld; no NPU activity.']},ensure_ascii=False,indent=2))
