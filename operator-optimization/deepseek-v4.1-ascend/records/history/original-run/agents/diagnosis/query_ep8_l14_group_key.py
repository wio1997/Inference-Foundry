#!/usr/bin/env python3
"""CPU-only query exact op 209 group key and metadata in frozen Oct 29 EP8 DB."""
import hashlib,json,pathlib,sqlite3,re
p=pathlib.Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029/runtime/diagnostic-traces/dp1_pp0_tp0_dcp0_ep8_rank0_737_20261029023525648_ascend_pt/ASCEND_PROFILER_OUTPUT/ascend_pytorch_profiler_8.db')
h=hashlib.sha256()
with p.open('rb') as f:
 for b in iter(lambda:f.read(1048576),b''):h.update(b)
assert h.hexdigest()=='e07ee2428be27c6129afd6dd0be8b700830048f30f13411b701e97c5d88e2486'
c=sqlite3.connect(p.as_uri()+'?mode=ro',uri=True);c.execute('PRAGMA query_only=ON')
rows=c.execute('SELECT s.value,o.rankSize FROM COMMUNICATION_OP o LEFT JOIN STRING_IDS s ON s.id=o.groupName WHERE o.opId=?',(209,)).fetchall();assert len(rows)==1 and isinstance(rows[0][0],str)
target,size=rows[0];assert target=='group_name_5' and size==8
metas=c.execute('SELECT value FROM META_DATA WHERE name=?',('parallel_group_info',)).fetchall()
dup=[]
def pairs(xs):
 d={}
 for k,v in xs:
  if k in d:dup.append(hashlib.sha256(k.encode()).hexdigest())
  d[k]=v
 return d
matches=[];n_groups=None
if len(metas)==1:
 obj=json.loads(metas[0][0],object_pairs_hook=pairs);assert isinstance(obj,dict) and len(obj)<=100
 n_groups=len(obj)
 for comm,g in obj.items():
  if isinstance(g,dict) and (comm==target or g.get('group_name')==target):
   name=g.get('group_name');m=re.fullmatch(r'(ep|mc2|tp|dp)(?::[0-9]+)?|default_group',name,re.I) if isinstance(name,str) else None
   members=g.get('global_ranks');assert isinstance(members,list) and len(members)<=128 and all(type(v) is int and 0<=v<65536 for v in members) and len(members)==len(set(members))
   matches.append({'global_ranks':members,'match_key':comm==target,'match_display_name':name==target,'category':m.group(0).split(':')[0].lower() if m else 'OTHER','group_rank':g.get('group_rank'),'member_count':len(g['global_ranks']) if isinstance(g.get('global_ranks'),list) else None})
rankmap=c.execute('SELECT rankId,deviceId FROM RANK_DEVICE_MAP WHERE deviceId=?',(8,)).fetchall()
meta_contains=[]
for n,v in c.execute('SELECT name,value FROM META_DATA'):
 if isinstance(v,str) and target in v:meta_contains.append({'name_is_parallel_group_info':n=='parallel_group_info','value_sha256':hashlib.sha256(v.encode()).hexdigest()})
print(json.dumps({'scope':'only op 209 and metadata of exact frozen Oct29 EP8 profile, no device execution','db_sha256':h.hexdigest(),'group_key_sha256':hashlib.sha256(target.encode()).hexdigest(),'op_rank_size':size,'parallel_group_info_count':len(metas),'parallel_group_info_json_sha256':hashlib.sha256(metas[0][0].encode()).hexdigest() if len(metas)==1 else None,'json_duplicate_key_count':len(dup),'group_count':n_groups,'device8_rank_map':rankmap,'unique_device8_rank_in_matching_members':len(rankmap)==1 and len(matches)==1 and rankmap[0][0] in matches[0]['global_ranks'],'rank_at_group_rank_index_matches_device_rank':len(rankmap)==1 and len(matches)==1 and type(matches[0]['group_rank']) is int and 0<=matches[0]['group_rank']<len(matches[0]['global_ranks']) and matches[0]['global_ranks'][matches[0]['group_rank']]==rankmap[0][0],'exact_matches':matches,'metadata_rows_containing_key':meta_contains,'limits':['Byte-exact metadata match is subject to exporter/producer linkage and registration completeness.','Absence of metadata match does not prove absence of communicator or members.','No claim of same collective instance or request. Global rank mapping is private evidence; do not publish.']}))
