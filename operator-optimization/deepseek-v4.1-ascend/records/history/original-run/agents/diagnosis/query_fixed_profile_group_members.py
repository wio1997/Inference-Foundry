#!/usr/bin/env python3
"""Read exact parallel_group_info records from two frozen profile DBs."""
import argparse,hashlib,json,re,sqlite3
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--task-root',required=True);ap.add_argument('--trace-root',required=True);a=ap.parse_args()
T=Path(a.task_root);R=Path(a.trace_root);prior=T/'agents/diagnosis/mc2-profile-topology-metadata.json';attrs=T/'agents/diagnosis/mc2-group-attr.json';m=T/'evidence/used-raw-source-sha256.json'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
for p,h in [(prior,'195aeec3aa1377c95655ed07a7e9c867b283d1187bbd842664a9e0947d6bff16'),(attrs,'a3f4e5f78ac1724e5a816a4da1b7d4fc2a1bb889078697a06b163200c690ee60'),(m,'cd5bac3b56c51fa2d7ce13199523ff9f88b94c4edf4c74131d4aedbdf2fc7c2a')]:
 if sha(p)!=h:raise RuntimeError('input mismatch')
px=json.loads(prior.read_text());ax=json.loads(attrs.read_text());manifest=json.loads(m.read_text());out=[];target_name='parallel_group_info'
for r in px['samples']:
 rank=r['rank'];matches=[z for z in manifest['records'] if z['relative_path'].startswith(rank+'/ASCEND_PROFILER_OUTPUT/ascend_pytorch_profiler_') and z['relative_path'].endswith('.db')]
 if len(matches)!=1:raise RuntimeError('ambiguous manifest DB')
 db=R/matches[0]['relative_path']
 if sha(db)!=r['db_sha256'] or sha(db)!=matches[0]['sha256']:raise RuntimeError('DB mismatch')
 prior_rows=[z for z in r['metadata_rows'] if z.get('name_sha256')==hashlib.sha256(target_name.encode()).hexdigest()]
 if len(prior_rows)!=1:raise RuntimeError('prior metadata not unique')
 entry=[z for z in ax['samples'] if z['rank']==rank];assert len(entry)==1
 attr_hashes=[z['value_sha256'] for z in entry[0]['entries'] if z['label']=='attr_0'];assert len(attr_hashes)==1
 c=sqlite3.connect(db.resolve().as_uri()+'?mode=ro',uri=True);c.row_factory=sqlite3.Row;c.execute('PRAGMA query_only=ON')
 rows=[dict(z) for z in c.execute('SELECT value FROM META_DATA WHERE name=? LIMIT 3',(target_name,))];c.close()
 if len(rows)!=1 or not isinstance(rows[0]['value'],str):raise RuntimeError('metadata missing/nonunique')
 raw=rows[0]['value'];digest=hashlib.sha256(raw.encode()).hexdigest()
 if digest!=prior_rows[0]['value_sha256'] or len(raw)>100000:raise RuntimeError('metadata changed')
 obj=json.loads(raw)
 if not isinstance(obj,dict) or len(obj)>50:raise RuntimeError('unexpected group metadata structure')
 groups=[]
 for comm,info in obj.items():
  if not isinstance(comm,str) or not isinstance(info,dict):raise RuntimeError('unexpected group entry')
  name=info.get('group_name');gr=info.get('group_rank');members=info.get('global_ranks')
  valid=(isinstance(name,str) and type(gr) is int and isinstance(members,list) and len(members)<=128 and all(type(v) is int and 0<=v<65536 for v in members))
  if not valid:raise RuntimeError('unexpected fields')
  match=re.fullmatch(r'(ep|mc2|tp|dp)(?::[0-9]+)?|default_group',name,re.I)
  comm_hash=hashlib.sha256(comm.encode()).hexdigest();cat=match.group(0).split(':')[0].lower() if match else 'OTHER'
  groups.append({'comm_name_sha256':comm_hash,'comm_name_utf8_hash_equals_attr0':comm_hash==attr_hashes[0], 'group_name_category':cat,'group_name_sha256':hashlib.sha256(name.encode()).hexdigest(),'group_rank':gr,'global_ranks':members,'metadata_field_keys':sorted(k for k in info if k in ('group_name','group_rank','global_ranks'))})
 out.append({'rank':rank,'db_sha256':sha(db),'metadata_sha256':digest,'rank_device_map':r['rank_device_map'],'groups':groups})
print(json.dumps({'scope':'Exactly the fixed parallel_group_info record in two profiler DBs; explicit fields only','samples':out,'limits':['Profiler source supplies a possible producer; actual capture/export version, group registration and coverage must be bound independently.', 'HCCL communicator key vs attr_0 hash equality is a byte-comparison clue only: profiler attr_N-to-ABI mapping/serialization not established.', 'Name category is allowlisted exact syntax only, OTHER is not evidence of any group type.', 'Even verified group membership cannot prove same request or same collective execution.', 'No raw communicator/group names or request text, no NPU access.']},ensure_ascii=False,indent=2))
