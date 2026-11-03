from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib, json, socket, sqlite3, time

assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
P=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/hostprof-20261030T212812Z-360e0c72')
def pinned(name,sha):
    b=(P/'analysis'/name).read_bytes();assert hashlib.sha256(b).hexdigest()==sha;return json.loads(b)
x=pinned('prepare-field-origins-v12.json','c23b75f3fbd3ae3d3d7152f6882524e899bade53001ea2f1befa09ce943f091f')
c=pinned('saved-prepare-consumers-v13.json','6ba784cac70931d7764111832ba40fc8ca3927f80c58d63ab70341594fab62eb')
OUT=P/'analysis/CP-dead-producer-device-tasks-v13.json';assert not OUT.exists()
rootmap={r['root_key']:r for r in x['prepare_roots']};batchmap={r['root_key']:r for r in c['batch_initialization_and_suffix']}
boundaries=defaultdict(list)
for r in x['selected_API_records']:
    if r['name']=='acl_memcpy_host_to_device'and 'dsa_v41_cp.py(131)'in r['project_frames'][0]['frame']:boundaries[r['root_key']].append(r)
for rs in boundaries.values():rs.sort(key=lambda r:r['startNs'])
def query_in(db,sql,values):
    out=[];values=sorted(set(values))
    for start in range(0,len(values),700):
        batch=values[start:start+700];out.extend(dict(r)for r in db.execute(sql.replace('{ids}',','.join('?'for _ in batch)),batch))
    return out
all_apis=[];all_tasks=[];t0=time.monotonic()
source_frames=['/vllm-workspace/vllm-ascend/vllm_ascend/attention/context_parallel/dsa_v41_cp.py('+str(i)+'): build'for i in [139,140,141,142]]
for di,spec in enumerate(x['databases']):
    assert time.monotonic()-t0<145,'bounded query time'
    db=sqlite3.connect(Path(spec['path']).as_uri()+'?mode=ro&immutable=1',uri=True);db.row_factory=sqlite3.Row;db.execute('PRAGMA query_only=ON')
    db.set_progress_handler(lambda:int(time.monotonic()-t0>155),100000)
    frame_ids=[r[0]for r in db.execute('SELECT id FROM STRING_IDS WHERE value IN ('+','.join('?'for _ in source_frames)+')',source_frames)]
    ids=query_in(db,'SELECT DISTINCT id FROM PYTORCH_CALLCHAINS WHERE stack IN ({ids})',frame_ids)
    apis=query_in(db,'''SELECT p.rowid AS api_rowid,CAST(p.startNs AS INTEGER)AS startNs,CAST(p.endNs AS INTEGER)AS endNs,p.globalTid,p.connectionId,p.callchainId,s.value AS name
      FROM PYTORCH_API p JOIN STRING_IDS s ON s.id=p.name WHERE p.callchainId IN ({ids})''',[r['id']for r in ids])
    relevant=[]
    for a in apis:
        roots=[r for r in x['prepare_roots']if r['database_index']==di and r['root']['globalTid']==a['globalTid']and r['root']['startNs']<=a['startNs']and a['endNs']<=r['root']['endNs']]
        if len(roots)!=1:continue
        root=roots[0];qs=boundaries[root['root_key']]
        positions=[j for j,q in enumerate(qs)if q['startNs']<=a['startNs']and(j+1==len(qs)or a['startNs']<qs[j+1]['startNs'])]
        if len(positions)!=1:continue
        ordinal=positions[0];local=[r for r in batchmap[root['root_key']]['initialization_copies']if r['scope']=='CP_LOCAL']
        a.update(database_index=di,root_key=root['root_key'],CP_build_ordinal=ordinal,after_observed_root_batch_init=len(local)==1 and a['startNs']>local[0]['endNs'],
                 dead_value_branch=ordinal>0 and len(local)==1 and a['startNs']>local[0]['endNs'])
        relevant.append(a)
    links=query_in(db,'SELECT id,connectionId FROM CONNECTION_IDS WHERE id IN ({ids})',[a['connectionId']for a in relevant])
    devices={r['connectionId']for r in links};host_by_dev=defaultdict(set)
    for r in links:host_by_dev[r['connectionId']].add(r['id'])
    tasks=query_in(db,'''SELECT t.rowid AS task_record_rowid,t.globalTaskId,t.startNs,t.endNs,t.streamId,t.taskId,t.connectionId,t.modelId,op.value AS op_type,n.value AS kernel_name,
      ish.value AS input_shapes,odt.value AS output_dtypes,c.blockNum,c.mixBlockNum
      FROM TASK t JOIN COMPUTE_TASK_INFO c ON c.globalTaskId=t.globalTaskId
      LEFT JOIN STRING_IDS op ON op.id=c.opType LEFT JOIN STRING_IDS n ON n.id=c.name LEFT JOIN STRING_IDS ish ON ish.id=c.inputShapes LEFT JOIN STRING_IDS odt ON odt.id=c.outputDataTypes
      WHERE t.connectionId IN ({ids})''',devices)
    byhost=defaultdict(list)
    for a in relevant:byhost[a['connectionId']].append(a)
    for task in tasks:
        candidates=[a for hid in host_by_dev[task['connectionId']]for a in byhost[hid]]
        task.update(database_index=di,source_API_rowids=sorted({a['api_rowid']for a in candidates}),root_keys=sorted({a['root_key']for a in candidates}),
                    CP_build_ordinals=sorted({a['CP_build_ordinal']for a in candidates}),dead_branch_values=sorted({a['dead_value_branch']for a in candidates}))
        task['dead_producer_task']=len(task['root_keys'])==1 and task['dead_branch_values']==[True]
    all_apis.extend(relevant);all_tasks.extend(tasks);db.close()
assert len({(t['database_index'],t['task_record_rowid'])for t in all_tasks})==len(all_tasks),'execution identity preserved'
def union(intervals):
    end=None;total=0
    for a,b in sorted(intervals):
        if end is None or a>end:total+=b-a;end=b
        elif b>end:total+=b-end;end=b
    return total
root_stats=[]
for key,root in rootmap.items():
    ts=[t for t in all_tasks if t['dead_producer_task']and t['root_keys']==[key]]
    if ts:root_stats.append(dict(root_key=key,tasks=len(ts),type_counts=dict(Counter(t['op_type']for t in ts)),kernel_interval_union_ns=union([(t['startNs'],t['endNs'])for t in ts]),
        stream_counts=dict(Counter(str(t['streamId'])for t in ts)),interval_union_is_not_latency_gain=True))
dead=[t for t in all_tasks if t['dead_producer_task']]
summary=dict(created_utc=datetime.now(timezone.utc).isoformat(),databases=32,source_bound_APIs=len(all_apis),linked_compute_executions=len(all_tasks),
    dead_producer_compute_executions=len(dead),dead_producer_op_type_counts=dict(Counter(t['op_type']for t in dead)),live_or_ambiguous_type_counts=dict(Counter(t['op_type']for t in all_tasks if not t['dead_producer_task'])),
    roots_with_dead_producer_tasks=len(root_stats),per_root_dead_task_hist=dict(Counter(r['tasks']for r in root_stats)),
    ambiguous_execution_links=sum(len(t['root_keys'])!=1 or len(t['dead_branch_values'])!=1 for t in all_tasks),
    no_new_target_source_read=True,no_device_or_service_work=True,whole_modules_closed=0,overall_complete=False,
    limits=['Tasks are joined by native CONNECTION_IDS relation to full-source-bound APIs,then exact CPUroot/buildintervals;not nearest kernel timestamps.',
            'Dead-value classification applies to source proven batch-hit localGPUlength subtree only. Preserve firstmiss,queryoffsetbuffers andall othermetadata.',
            'Kernel interval union is observed profiled device work,not unprofiled CPUcost,request-critical-path gain orattainable speedup.'])
payload=json.dumps(dict(summary=summary,APIs=all_apis,compute_executions=all_tasks,per_root_device_work=root_stats,source_frames=source_frames),ensure_ascii=False,separators=(',',':'))+'\n';OUT.write_text(payload)
print(json.dumps(dict(summary=summary,first_root=root_stats[:1],full_result=str(OUT),result_sha256=hashlib.sha256(payload.encode()).hexdigest()),ensure_ascii=False))
