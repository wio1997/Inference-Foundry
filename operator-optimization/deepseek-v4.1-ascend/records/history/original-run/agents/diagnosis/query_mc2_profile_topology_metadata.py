#!/usr/bin/env python3
"""Two fixed profile DBs: bounded metadata keys and explicit topology labels."""
import argparse,hashlib,json,re,sqlite3
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--task-root',required=True);ap.add_argument('--trace-root',required=True);a=ap.parse_args()
T=Path(a.task_root);R=Path(a.trace_root);manifest=T/'evidence/used-raw-source-sha256.json'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
if sha(manifest)!='cd5bac3b56c51fa2d7ce13199523ff9f88b94c4edf4c74131d4aedbdf2fc7c2a':raise RuntimeError('manifest mismatch')
m=json.loads(manifest.read_text());ranks=['dp0_pp0_tp0_dcp0_ep0_rank0_728_20261028020238584_ascend_pt','dp1_pp0_tp0_dcp0_ep8_rank0_727_20261028020238580_ascend_pt'];out=[]
allowed={'group','group_name','groupname','group_id','groupid','rank','rank_id','rankid','rank_size','ranksize','rank_list','ranklist','ranks','members','group_members','world_size','worldsize','parallel_group','process_group','ep_group','mc2_group'}
integer_fields={'rank','rank_id','rankid','rank_size','ranksize','world_size','worldsize'};list_fields={'ranks','rank_list','ranklist','members','group_members'}
for rank in ranks:
 files=[z for z in m['records'] if z['relative_path'].startswith(rank+'/ASCEND_PROFILER_OUTPUT/ascend_pytorch_profiler_') and z['relative_path'].endswith('.db')]
 if len(files)!=1:raise RuntimeError('ambiguous DB manifest')
 p=R/files[0]['relative_path']
 if sha(p)!=files[0]['sha256']:raise RuntimeError('database mismatch')
 c=sqlite3.connect(p.resolve().as_uri()+'?mode=ro',uri=True);c.row_factory=sqlite3.Row;c.execute('PRAGMA query_only=ON')
 meta=[dict(z) for z in c.execute('SELECT name,value FROM META_DATA LIMIT 101').fetchall()]
 if len(meta)>100:raise RuntimeError('metadata row bound exceeded')
 rankmap=[dict(z) for z in c.execute('SELECT rankId,deviceId FROM RANK_DEVICE_MAP LIMIT 101').fetchall()]
 if len(rankmap)>100:raise RuntimeError('rank map bound exceeded')
 rows=[]
 for z in meta:
  name=z['name'];value=z['value']
  if not isinstance(name,str) or not isinstance(value,str):rows.append({'name_type':type(name).__name__,'value_type':type(value).__name__,'parsed':False});continue
  if len(name)>256 or len(value)>1048576:raise RuntimeError('metadata size limit')
  labels=[]
  def visit(obj,path=(),depth=0):
   state[0]+=1
   if state[0]>10000:raise RuntimeError('JSON node bound exceeded')
   if depth>10:state[1]=True;return
   if isinstance(obj,dict):
    for k,v in obj.items():
     if not isinstance(k,str):continue
     norm=k.lower().replace('-','_');new=path+(k,)
     if norm in allowed:
      ent={'label':norm,'path_sha256':hashlib.sha256('/'.join(new).encode()).hexdigest(),'value_type':type(v).__name__,'value_sha256':hashlib.sha256(json.dumps(v,sort_keys=True,ensure_ascii=False).encode()).hexdigest()}
      if norm in integer_fields and type(v) is int and 0<=v<=65535:ent['integer']=v
      if norm in list_fields and isinstance(v,list) and len(v)<=64 and all(type(y) is int and 0<=y<=65535 for y in v):ent['integers']=v
      labels.append(ent)
      if len(labels)>200:raise RuntimeError('label bound exceeded')
     if isinstance(v,(dict,list)):visit(v,new,depth+1)
   elif isinstance(obj,list):
    if len(obj)>128:state[1]=True
    for i,v in enumerate(obj[:128]):
     if isinstance(v,(dict,list)):visit(v,path+(str(i),),depth+1)
  state=[0,False]
  try:parsed=json.loads(value);status='json';visit(parsed)
  except json.JSONDecodeError:status='non_json'
  norm=name.lower().replace('-','_')
  if norm in allowed:
   ent={'label':norm,'path_sha256':hashlib.sha256(name.encode()).hexdigest(),'value_type':'metadata_value','value_sha256':hashlib.sha256(value.encode()).hexdigest()}
   if status=='json' and norm in integer_fields and type(parsed) is int and 0<=parsed<=65535:ent['integer']=parsed
   if status=='json' and norm in list_fields and isinstance(parsed,list) and len(parsed)<=64 and all(type(y) is int and 0<=y<=65535 for y in parsed):ent['integers']=parsed
   labels.append(ent)
  rows.append({'name_sha256':hashlib.sha256(name.encode()).hexdigest(),'name_labels':[w for w in ('group','member','rank','world','parallel','process') if w in name.lower()],'value_sha256':hashlib.sha256(value.encode()).hexdigest(),'value_length':len(value),'value_format':status,'top_level_type':type(parsed).__name__ if status=='json' else None,'nodes_checked':state[0],'walk_truncated':state[1],'topology_labels':labels})
 c.close();out.append({'rank':rank,'db_sha256':files[0]['sha256'],'rank_device_map':rankmap,'metadata_rows':rows})
print(json.dumps({'scope':'Only EP0 and EP8 META_DATA/RANK_DEVICE_MAP; bounded JSON labels only','manifest_sha256':sha(manifest),'samples':out,'limits':['Explicit group/rank labels can describe unrelated groups or metadata; no MC2 communicator binding is inferred.', 'Only these two DBs, their META_DATA rows and numeric RANK_DEVICE_MAP are examined; no other logs or sources.', 'Non-JSON, alternate spellings, strings containing serialized data, or depth/array caps may hide data.', 'Only allowed label names, hashes and small numeric rank-like values are emitted; no raw text or request payload.', 'No HCCL table scans, no NPU work.']},ensure_ascii=False,indent=2))
