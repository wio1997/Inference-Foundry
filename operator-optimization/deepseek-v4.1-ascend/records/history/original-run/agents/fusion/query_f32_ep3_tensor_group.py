#!/usr/bin/env python3
"""Fixed EP3 call IDs and communication group metadata, no tensor values."""
import sqlite3,pathlib,json,time,hashlib,re
p=pathlib.Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029/runtime/diagnostic-traces/dp0_pp0_tp3_dcp0_ep3_rank3_2128_20261029023525650_ascend_pt/ASCEND_PROFILER_OUTPUT/ascend_pytorch_profiler_3.db')
c=sqlite3.connect(p.as_uri()+'?mode=ro',uri=True);c.row_factory=sqlite3.Row;start=time.monotonic();c.set_progress_handler(lambda:int(time.monotonic()-start>25),100000)
ids=[431371,431372,431373,431383];q='SELECT connectionId,name,inputDtypes,inputShapes,startNs,endNs FROM PYTORCH_API WHERE connectionId IN (?,?,?,?)';rows=[dict(x) for x in c.execute(q,ids)]
need=sorted({r[k] for r in rows for k in ('name','inputDtypes','inputShapes') if r[k] is not None});lookup={}
if need: lookup={r[0]:r[1] for r in c.execute('SELECT id,value FROM STRING_IDS WHERE id IN ('+','.join('?' for _ in need)+')',need)}
for r in rows:
 for k in ('name','inputDtypes','inputShapes'):r[k]=lookup.get(r[k])
op=c.execute('SELECT o.count,o.rankSize,o.dataType,e.name dtype FROM COMMUNICATION_OP o LEFT JOIN ENUM_HCCL_DATA_TYPE e ON o.dataType=e.id WHERE o.opId=246').fetchone()
row=c.execute('SELECT s.value FROM COMMUNICATION_OP o JOIN STRING_IDS s ON s.id=o.groupName WHERE o.opId=246').fetchone();target=row[0] if row else None
metas=c.execute('SELECT value FROM META_DATA WHERE name=? LIMIT 2',('parallel_group_info',)).fetchall();matched=[];duplicate_keys=[]
def check_pairs(pairs):
 d={}
 for k,v in pairs:
  if k in d:duplicate_keys.append(hashlib.sha256(k.encode()).hexdigest())
  d[k]=v
 return d
if len(metas)==1 and target is not None:
 obj=json.loads(metas[0][0],object_pairs_hook=check_pairs);assert isinstance(obj,dict) and len(obj)<100
 for comm,x in obj.items():
  if not isinstance(x,dict):continue
  name=x.get('group_name');is_key=comm==target;is_name=name==target
  if is_key or is_name:
   cat=(re.fullmatch(r'(ep|mc2|tp|dp)(?::[0-9]+)?|default_group',name,re.I).group(0).split(':')[0].lower() if isinstance(name,str) and re.fullmatch(r'(ep|mc2|tp|dp)(?::[0-9]+)?|default_group',name,re.I) else 'OTHER')
   matched.append({'matches_communicator_key':is_key,'matches_group_name':is_name,'category':cat,'group_rank':x.get('group_rank'),'global_ranks':x.get('global_ranks')})
meta_scan=[]
for n,v in c.execute('SELECT name,value FROM META_DATA'):
 if isinstance(v,str) and isinstance(target,str) and target in v:
  meta_scan.append({'name_is_parallel_group_info':n=='parallel_group_info','value_sha256':hashlib.sha256(v.encode()).hexdigest()})
 if len(meta_scan)>50:raise RuntimeError('unexpected group metadata match count')
print(json.dumps({'metadata_rows_containing_group_key':meta_scan,'fixed_host_calls':rows,'communication_fields':dict(op) if op else None,'dtype_id11_mapping_rows':c.execute('SELECT COUNT(*) FROM ENUM_HCCL_DATA_TYPE WHERE id=11').fetchone()[0],'metadata_json_duplicate_key_count':len(duplicate_keys),'groupName_string_present':target is not None,'groupName_sha256':hashlib.sha256(target.encode()).hexdigest() if isinstance(target,str) else None,'metadata_rows':len(metas),'matched_groups':matched,'limits':'shape only, no tensor pointers/storage IDs; same string is a clue subject to 9.2 exporter contract; absence of match is not no group'},indent=2))
