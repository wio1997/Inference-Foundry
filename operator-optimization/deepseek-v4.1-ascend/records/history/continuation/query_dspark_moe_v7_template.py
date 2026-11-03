from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timezone
import sqlite3, socket, json, hashlib

assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
DB=Path('/data/tiankuan/wio/migration-166-20261030/source169-project/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029/runtime/diagnostic-traces/dp0_pp0_tp3_dcp0_ep3_rank3_2128_20261029023525650_ascend_pt/ASCEND_PROFILER_OUTPUT/ascend_pytorch_profiler_3.db')
SHA='084b8718888e33271435ec79d7573c95ca1615f3e7484c807ea24590fad234c4'
before=DB.stat()
h=hashlib.sha256()
with DB.open('rb') as f:
    for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
assert h.hexdigest()==SHA
db=sqlite3.connect(DB.as_uri()+'?mode=ro&immutable=1',uri=True)
db.row_factory=sqlite3.Row
db.execute('PRAGMA query_only=ON')
SELECT='''SELECT t.startNs,t.endNs,t.deviceId,t.streamId,t.taskId,t.globalTaskId,t.connectionId,t.modelId,
 n.value AS kernel_name,op.value AS op_type,ish.value AS input_shapes,osh.value AS output_shapes,
 idt.value AS input_dtypes,odt.value AS output_dtypes,at.value AS attrs,c.blockNum,c.mixBlockNum
 FROM TASK t JOIN COMPUTE_TASK_INFO c ON c.globalTaskId=t.globalTaskId
 JOIN STRING_IDS n ON n.id=c.name LEFT JOIN STRING_IDS op ON op.id=c.opType
 LEFT JOIN STRING_IDS ish ON ish.id=c.inputShapes LEFT JOIN STRING_IDS osh ON osh.id=c.outputShapes
 LEFT JOIN STRING_IDS idt ON idt.id=c.inputDataTypes LEFT JOIN STRING_IDS odt ON odt.id=c.outputDataTypes
 LEFT JOIN STRING_IDS at ON at.id=c.attrInfo'''
anchor_tasks=[]
for a in ANCHORS:
    rows=[dict(r) for r in db.execute(SELECT+' WHERE t.deviceId=3 AND t.streamId=? AND t.taskId=? AND t.startNs=? AND t.endNs=?',
                                    (int(a['stream']),int(a['task_id']),a['start_ns'],a['end_ns']))]
    assert len(rows)==1,(a['label'],'nonunique_anchor',len(rows))
    anchor_tasks.append(dict(label=a['label'],scope_id=a['scope_id'],layer=a['layer'],digest=a['digest'],task=rows[0]))
dspark_frames=[dict(r) for r in db.execute('''SELECT c.id,c.stackDepth,s.value AS frame FROM PYTORCH_CALLCHAINS c JOIN STRING_IDS s ON s.id=c.stack
 WHERE s.value LIKE '%/models/deepseek_v41/dspark.py(%' ORDER BY c.id,c.stackDepth''')]
dspark_ids={r['id'] for r in dspark_frames}
assert 0<len(dspark_ids)<250000,('dspark_source_chain_budget',len(dspark_ids))
api_sql='''SELECT p.startNs,p.endNs,p.globalTid,p.connectionId,p.callchainId,n.value AS name
 FROM PYTORCH_API p LEFT JOIN STRING_IDS n ON n.id=p.name'''
dspark_apis_all=[dict(r) for r in db.execute(api_sql) if r['callchainId'] in dspark_ids]
assert len(dspark_apis_all)<250000
host_ids={r['connectionId'] for r in dspark_apis_all if r['connectionId'] is not None}
extra_device={a['task']['connectionId'] for a in anchor_tasks}
links=[dict(r) for r in db.execute('SELECT id,connectionId FROM CONNECTION_IDS') if r['id'] in host_ids or r['connectionId'] in extra_device]
linked_host_ids={r['id'] for r in links}
dspark_apis=[r for r in dspark_apis_all if r['connectionId'] in linked_host_ids]
assert len(dspark_apis)<50000,('linked_dspark_api_budget',len(dspark_apis))
device_ids={r['connectionId'] for r in links if r['id'] in host_ids}
assert 0<len(device_ids)<20000,('device_connection_budget',len(device_ids))
extra_host={r['id'] for r in links if r['connectionId'] in extra_device}
extra_apis=[dict(r) for r in db.execute(api_sql+' WHERE p.connectionId IN ('+','.join(str(int(x)) for x in extra_host)+')')] if extra_host else []
apis={json.dumps(r,sort_keys=True):r for r in dspark_apis+extra_apis}
apis=list(apis.values())
chain_ids={r['callchainId'] for r in apis if r['callchainId'] is not None}
assert len(chain_ids)<50000
frames=defaultdict(list)
for r in db.execute('SELECT c.id,c.stackDepth,s.value AS frame FROM PYTORCH_CALLCHAINS c JOIN STRING_IDS s ON s.id=c.stack WHERE c.id IN ('+','.join(str(int(x)) for x in chain_ids)+') ORDER BY c.id,c.stackDepth'):
    if r['stackDepth']<4 or any(s in r['frame'] for s in ['deepseek_v41','moe_comm_method','triton','rope_dsv4','spec_decode/dspark','layernorm','vocab_parallel_embedding']):
        frames[r['id']].append(dict(stackDepth=r['stackDepth'],frame=r['frame']))
api_by_host=defaultdict(list)
for a in apis:api_by_host[a['connectionId']].append(a)
host_by_device=defaultdict(set)
for l in links:host_by_device[l['connectionId']].add(l['id'])


def attach(row):
    row['api_candidates']=[a for host in sorted(host_by_device[row['connectionId']]) for a in api_by_host[host]]
    return row


tasks=[attach(dict(r)) for r in db.execute(SELECT+' WHERE t.deviceId=3 AND t.connectionId IN ('+','.join(str(int(x)) for x in sorted(device_ids))+') ORDER BY t.startNs,t.streamId LIMIT 50001')]
assert len(tasks)<=50000,'compute_task_budget'
for a in anchor_tasks:attach(a['task'])
db.close()
after=DB.stat()
assert before.st_size==after.st_size and before.st_mtime_ns==after.st_mtime_ns
signatures=defaultdict(list)
for t in tasks:
    relevant=sorted({f['frame'] for a in t['api_candidates'] for f in frames[a['callchainId']] if '/models/deepseek_v41/dspark.py(' in f['frame']})
    key=json.dumps([t['op_type'],t['input_shapes'],t['output_shapes'],relevant],ensure_ascii=False)
    signatures[key].append(t)
groups=[]
for key,ts in signatures.items():
    op,ish,osh,fs=json.loads(key)
    groups.append(dict(op_type=op,input_shapes=ish,output_shapes=osh,dspark_frames=fs,count=len(ts),
                       first_task=dict(taskId=ts[0]['taskId'],startNs=ts[0]['startNs'],endNs=ts[0]['endNs'],connectionId=ts[0]['connectionId']),
                       last_start_ns=max(t['startNs'] for t in ts),api_count_hist=dict(Counter(len(t['api_candidates']) for t in ts))))
groups.sort(key=lambda x:x['first_task']['startNs'])
source_count=Counter(f['frame'] for a in dspark_apis for f in frames[a['callchainId']] if '/models/deepseek_v41/dspark.py(' in f['frame'])
result=dict(created_utc=datetime.now(timezone.utc).isoformat(),database=str(DB),database_sha256=SHA,database_stat_unchanged=True,
            dspark_source_chain_count=len(dspark_ids),dspark_all_api_count=len(dspark_apis_all),
            dspark_source_frame_counts=dict(source_count),dspark_api_count=len(dspark_apis),dspark_device_connections=len(device_ids),
            task_count=len(tasks),op_type_counts=dict(Counter(t['op_type'] for t in tasks)),signature_groups=groups,
            tasks=tasks,moe_anchors=anchor_tasks,callchains={str(k):v for k,v in frames.items()},
            no_device_or_inference_work=True,
            limits=['All saved DSpark source-frame/API connection matches in this one database; not all ranks, phases or unexported/graph-only work.',
                    'API/connection membership is not tensor/storage identity. Reused connections or multiple candidate APIs remain ambiguous.',
                    'Source-frame names/line numbers identify recorded submission sites, not historical loaded file hashes or native kernel stages.',
                    'Shape/count/signature groups are not layer-group identity, actual slot values or request membership.',
                    'Callchains retain depths0-3 and declared semantic source frames; omitted runtime-wrapper frames are not asserted absent.',
                    'Unlinked DSpark host/API records are counted but do not acquire a device task identity.',
                    'No join to166 expert-count snapshots or same-call native internal phase timing is made.'])
out=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/profile-readonly-20261031/ep3-dspark-and-moe-linked-tasks-v7-v2.json')
assert not out.exists(),'refuse_overwrite'
payload=json.dumps(result,ensure_ascii=False,indent=2)+'\n'
assert len(payload.encode())<100*1024*1024,'derived_result_budget'
out.write_text(payload)


def anchor_short(a):
    t=a['task']
    return dict(label=a['label'],scope_id=a['scope_id'],layer=a['layer'],task={k:v for k,v in t.items() if k!='api_candidates'},
                api=[dict(name=x['name'],connectionId=x['connectionId'],callchainId=x['callchainId'],
                          selected_frames=[f for f in frames[x['callchainId']] if any(s in f['frame'] for s in ['moe_comm','fused_moe','model.py('])][:7]) for x in t['api_candidates']])


public={k:result[k] for k in ['created_utc','database_sha256','database_stat_unchanged','dspark_source_chain_count','dspark_all_api_count','dspark_source_frame_counts','dspark_api_count','dspark_device_connections','task_count','op_type_counts','no_device_or_inference_work','limits']}
public.update(full_result=str(out),result_sha256=hashlib.sha256(payload.encode()).hexdigest(),signature_group_count=len(groups),
              signature_display=groups[:40],signature_display_truncated=len(groups)>40,moe_anchors=[anchor_short(a) for a in anchor_tasks])
while len(json.dumps(public,ensure_ascii=False).encode())>30000 and public['signature_display']:
    public['signature_display']=public['signature_display'][:-1]
    public['signature_display_truncated']=True
assert len(json.dumps(public,ensure_ascii=False).encode())<=32000,'public_summary_budget'
print(json.dumps(public,ensure_ascii=False))
