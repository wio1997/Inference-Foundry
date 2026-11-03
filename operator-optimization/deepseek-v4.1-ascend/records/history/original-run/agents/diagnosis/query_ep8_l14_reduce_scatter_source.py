#!/usr/bin/env python3
"""Read existing EP8 DB and sidecar only; output a narrowly selected association."""
import hashlib,json,pathlib,sqlite3
base=pathlib.Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029/runtime')
p=base/'diagnostic-traces/dp1_pp0_tp0_dcp0_ep8_rank0_737_20261029023525648_ascend_pt/ASCEND_PROFILER_OUTPUT/ascend_pytorch_profiler_8.db'
side=base/'metadata/events-737.jsonl'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for x in iter(lambda:f.read(1024*1024),b''):h.update(x)
 return h.hexdigest()
assert sha(p)=='e07ee2428be27c6129afd6dd0be8b700830048f30f13411b701e97c5d88e2486'
assert sha(side)=='ec095ff0bcdc9a0a2d545985c36bbcc75630f1181c1b8d08838221295fe4320f'
c=sqlite3.connect(p.as_uri()+'?mode=ro',uri=True);c.row_factory=sqlite3.Row;c.execute('PRAGMA query_only=ON');c.execute('PRAGMA automatic_index=OFF')
def q(sql,ps=()):return [dict(r) for r in c.execute(sql,ps)]
anchor=q('select * from TASK where deviceId=8 and globalPid=737 and streamId=47 and taskId=44217 and startNs=?',(1793241329140513188,));assert len(anchor)==1
aicpu=q('select * from TASK where deviceId=8 and globalPid=737 and streamId=92 and taskId=3999 and startNs=?',(1793241329205648107,));assert len(aicpu)==1 and aicpu[0]['connectionId']==98566
links=q('select * from CONNECTION_IDS where id in (select id from CONNECTION_IDS where connectionId=?) order by id,connectionId',(98566,));assert links==[{'id':175592,'connectionId':98554},{'id':175592,'connectionId':98566}]
host=q('select p.startNs,p.endNs,p.globalTid,p.connectionId,s.value name from PYTORCH_API p left join STRING_IDS s on p.name=s.id where p.connectionId=?',(175592,));assert len(host)==1 and host[0]['name']=='HcclReduceScatter'
h=host[0]
scopes=q('select p.startNs,p.endNs,p.globalTid,p.connectionId,s.value name from PYTORCH_API p left join STRING_IDS s on p.name=s.id where p.globalTid=? and cast(p.startNs as integer)<=? and cast(p.endNs as integer)>=? order by cast(p.endNs as integer)-cast(p.startNs as integer)',(h['globalTid'],int(h['startNs']),int(h['endNs'])))
assert sum(x['name']=='wio_diag:DeepseekV41DecoderLayer.forward:737-29' for x in scopes)==1
apis=q('select a.*,s.value nameVal from CANN_API a left join STRING_IDS s on a.name=s.id where a.connectionId in (?,?) order by a.connectionId',(98554,98566));assert len(apis)==2 and all(x['nameVal']=='launch' for x in apis)
tasks=q('select a.*,s.value taskTypeName from TASK a left join STRING_IDS s on a.taskType=s.id where a.connectionId in (?,?) order by a.startNs',(98554,98566))
comm=q('select p.*,s.value nameVal,sg.value groupVal from COMMUNICATION_OP p left join STRING_IDS s on p.opName=s.id left join STRING_IDS sg on p.groupName=sg.id where p.connectionId in (?,?,?)',(175592,98554,98566));assert len(comm)==1 and comm[0]['connectionId']==98554
selected=[]
with side.open() as f:
 for line in f:
  row=json.loads(line)
  if row.get('call_id')=='737-29':
   selected.append({k:row.get(k) for k in ('kind','pid','tid','call_id','label','layer_index','outcome')})
assert len(selected)==2 and selected[0]['layer_index']==23 and selected[0]['label']=='DeepseekV41DecoderLayer.forward'
start=anchor[0]['startNs'];end=anchor[0]['endNs']
relative={label:{'start_from_anchor_ns':x['startNs']-start,'end_from_anchor_ns':x['endNs']-start,'start_from_anchor_end_ns':x['startNs']-end} for label,x in [('aicpu',aicpu[0]),('comm_op',comm[0])]}
print(json.dumps({'db_sha256':sha(p),'sidecar_sha256':sha(side),'anchor':anchor[0],'aicpu':aicpu[0],'connection_links':links,'host':host,'host_scopes':scopes,'apis':apis,'tasks':tasks,'communication_op':comm,'sidecar_call':selected,'relative':relative,'limits':'DB connection associations, nesting and exported timestamps do not establish physical transaction, same request, producer-consumer dependence, event handles, causation, cross-track clock accuracy, or actual link-transfer duration. Sidecar does not independently specify main versus draft.'},indent=2))
