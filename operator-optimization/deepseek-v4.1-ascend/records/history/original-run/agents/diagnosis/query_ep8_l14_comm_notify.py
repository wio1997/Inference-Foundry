#!/usr/bin/env python3
"""Inspect specific stored communication and task identities, read only."""
import pathlib,hashlib,sqlite3,json
p=pathlib.Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029/runtime/diagnostic-traces/dp1_pp0_tp0_dcp0_ep8_rank0_737_20261029023525648_ascend_pt/ASCEND_PROFILER_OUTPUT/ascend_pytorch_profiler_8.db')
h=hashlib.sha256()
with p.open('rb') as f:
 for b in iter(lambda:f.read(1048576),b''):h.update(b)
assert h.hexdigest()=='e07ee2428be27c6129afd6dd0be8b700830048f30f13411b701e97c5d88e2486'
c=sqlite3.connect(p.as_uri()+'?mode=ro',uri=True);c.row_factory=sqlite3.Row;c.execute('pragma query_only=on');c.execute('pragma automatic_index=off')
def q(s,v=()):return [dict(x) for x in c.execute(s,v)]
tables=q("select name from sqlite_master where type='table' and (lower(name) like '%comm%' or lower(name) like '%event%' or lower(name) like '%notify%') order by name")
schemas={t['name']:q('pragma table_info("'+t['name']+'")') for t in tables}
comm=q('select * from COMMUNICATION_OP where opId=209');assert len(comm)==1 and comm[0]['connectionId']==98554
children=q('select * from COMMUNICATION_TASK_INFO where opId=209 order by globalTaskId')
ids=sorted({x['globalTaskId'] for x in children if x['globalTaskId'] is not None});refs={str(i):q('select * from TASK where globalTaskId=?',(i,)) for i in ids}
related=q('select * from COMMUNICATION_TASK_INFO where globalTaskId in (2538,2874,71682,71683,71684,20240,20241) order by globalTaskId,opId')
assert len(children)==168 and len(ids)==168 and all(len(refs[str(i)])==1 for i in ids)
assert all(x['timestampNs']==refs[str(x['globalTaskId'])][0]['startNs'] for x in children)
start=1793241329140513188;end=1793241329300879144
assert all(not (t['startNs']<end and t['endNs']>start) for v in refs.values() for t in v)
assert 2874 not in ids
assert all(t['connectionId']!=98566 for v in refs.values() for t in v)
earliest_delay=min(v[0]['startNs'] for v in refs.values())-end
assert earliest_delay==226807529
notify_ids=sorted({x['notifyId'] for x in children if x['notifyId'] not in (0,None)})
marks=','.join('?' for _ in notify_ids)
notify_reuse=q('select notifyId,opId,count(*) n from COMMUNICATION_TASK_INFO where notifyId in ('+marks+') group by notifyId,opId order by notifyId,opId',notify_ids)
print(json.dumps({'summary':{'child_count':len(children),'unique_task_ids':len(ids),'matching_task_count_each':1,'timestamp_mismatches':0,'overlapping_anchor':0,'earliest_delay_after_anchor_ns':earliest_delay,'aicpu_task_id_2874_among_child_ids':False,'child_tasks_with_connection_98566':0},'notify_reuse':notify_reuse,'db_sha256':h.hexdigest(),'tables':tables,'schemas':schemas,'comm':comm,'children':children,'task_refs':refs,'related_global_task_id':related},indent=2))
