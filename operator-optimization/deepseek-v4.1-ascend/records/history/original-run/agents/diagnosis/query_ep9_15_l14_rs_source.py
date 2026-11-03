#!/usr/bin/env python3
"""Inside stage: exact per-rank AI_CPU source associations in saved DB and sidecar."""
import pathlib,sqlite3,hashlib,json,collections,re
spec=json.loads(SPEC_JSON)
base=pathlib.Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029/runtime')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
out=[]
for r in spec:
 ep=r['ep'];db=base/'diagnostic-traces'/r['rank']/'ASCEND_PROFILER_OUTPUT'/('ascend_pytorch_profiler_%d.db'%ep);side=base/'metadata'/('events-%d.jsonl'%r['pid'])
 assert not pathlib.Path(str(db)+'-wal').exists() and sha(db)==r['db_sha'] and sha(side)==r['side_sha']
 c=sqlite3.connect(db.as_uri()+'?mode=ro',uri=True);c.row_factory=sqlite3.Row;c.execute('PRAGMA query_only=ON');c.execute('PRAGMA automatic_index=OFF')
 def q(sql,ps=()):return [dict(x) for x in c.execute(sql,ps)]
 selected=[];scope_ids=set()
 for x in r['items']:
  tasks=q('SELECT * FROM TASK WHERE deviceId=? AND globalPid=? AND streamId=? AND taskId=? AND startNs=?',(ep,r['pid'],x['streamId'],x['taskId'],x['startNs']))
  assert len(tasks)==1 and tasks[0]['connectionId']==x['connectionId'] and tasks[0]['endNs']==x['endNs']
  con=x['connectionId'];links=q('SELECT * FROM CONNECTION_IDS WHERE id IN (SELECT id FROM CONNECTION_IDS WHERE connectionId=?) ORDER BY id,connectionId',(con,));host_ids=sorted(set(z['id'] for z in links if z['connectionId']==con));allcons=sorted(set(z['connectionId'] for z in links))
  hosts=[]
  for hid in host_ids:
   hh=q('SELECT p.startNs,p.endNs,p.globalTid,p.connectionId,s.value name FROM PYTORCH_API p LEFT JOIN STRING_IDS s ON p.name=s.id WHERE p.connectionId=?',(hid,))
   for h in hh:
    h['scopes']=q('SELECT p.startNs,p.endNs,p.globalTid,p.connectionId,s.value name FROM PYTORCH_API p LEFT JOIN STRING_IDS s ON p.name=s.id WHERE p.globalTid=? AND CAST(p.startNs AS INTEGER)<=? AND CAST(p.endNs AS INTEGER)>=? ORDER BY CAST(p.endNs AS INTEGER)-CAST(p.startNs AS INTEGER)',(h['globalTid'],int(h['startNs']),int(h['endNs'])))
    for s in h['scopes']:
     n=s['name'] or ''
     if n.startswith('wio_diag:DeepseekV41DecoderLayer.forward:'):scope_ids.add(n.rsplit(':',1)[-1])
   hosts+=hh
  comm=[];api=[];reverse=[]
  for cc in allcons:
   reverse.append({'connectionId':cc,'ids':q('SELECT * FROM CONNECTION_IDS WHERE connectionId=? ORDER BY id',(cc,))})
   api+=q('SELECT a.startNs,a.endNs,a.connectionId,s.value name FROM CANN_API a LEFT JOIN STRING_IDS s ON a.name=s.id WHERE a.connectionId=?',(cc,))
   comm+=q('SELECT p.*,s.value nameVal,sg.value groupVal FROM COMMUNICATION_OP p LEFT JOIN STRING_IDS s ON p.opName=s.id LEFT JOIN STRING_IDS sg ON p.groupName=sg.id WHERE p.connectionId=?',(cc,))
  selected.append({'task':tasks[0],'connection_links':links,'reverse_links':reverse,'hosts':hosts,'cann_api':api,'comm_ops':comm})
 metas=q("SELECT value FROM META_DATA WHERE name=?",('parallel_group_info',)); rankmap=q('SELECT rankId,deviceId FROM RANK_DEVICE_MAP WHERE deviceId=?',(ep,)); duplicates=[]
 def hook(xs):
  d={}
  for k,v in xs:
   if k in d:duplicates.append(k)
   d[k]=v
  return d
 obj=json.loads(metas[0]['value'],object_pairs_hook=hook) if len(metas)==1 else {}
 assert isinstance(obj,dict) and len(obj)<=100
 for item in selected:
  for op in item['comm_ops']:
   target=op['groupVal'];v=obj.get(target);match=None
   if isinstance(v,dict):
    name=v.get('group_name');m=re.fullmatch(r'(ep|mc2|tp|dp)(?::[0-9]+)?|default_group',name,re.I) if isinstance(name,str) else None
    members=v.get('global_ranks'); assert isinstance(members,list) and len(members)<=128 and all(type(z) is int and 0<=z<65536 for z in members) and len(members)==len(set(members))
    group_rank=v.get('group_rank');match={'category':m.group(0).split(':')[0].lower() if m else 'OTHER','group_rank':group_rank,'member_count':len(members),'global_ranks':members,'rank_at_group_rank_matches_device':len(rankmap)==1 and type(group_rank) is int and 0<=group_rank<len(members) and members[group_rank]==rankmap[0]['rankId'],'rank_size_matches':len(members)==op['rankSize']}
   op['metadata_match']=match
 metadata={'parallel_group_info_rows':len(metas),'duplicate_key_count':len(duplicates),'rank_device_rows':rankmap}
 c.close()
 side_entries=[]
 with side.open() as f:
  for line in f:
   v=json.loads(line)
   if v.get('call_id') in scope_ids:
    side_entries.append({k:v.get(k) for k in ('kind','pid','tid','call_id','label','layer_index','outcome')})
 assert sha(db)==r['db_sha'] and sha(side)==r['side_sha'] and not pathlib.Path(str(db)+'-wal').exists()
 out.append({'ep':ep,'db_sha256':r['db_sha'],'sidecar_sha256':r['side_sha'],'sidecar_entries':side_entries,'group_metadata_integrity':metadata,'items':selected})
print(json.dumps({'scope':'independent rank-local exact TASK connection links and enclosing host-thread scopes; no dependency, same transaction or cross-rank clock inference','ranks':out},indent=2))
