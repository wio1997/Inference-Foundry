from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime, timezone
import json, hashlib, sqlite3, socket, time, re

assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
DB=Path('/data/tiankuan/wio/migration-166-20261030/source169-project/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029/runtime/diagnostic-traces/dp0_pp0_tp3_dcp0_ep3_rank3_2128_20261029023525650_ascend_pt/ASCEND_PROFILER_OUTPUT/ascend_pytorch_profiler_3.db')
SHA='084b8718888e33271435ec79d7573c95ca1615f3e7484c807ea24590fad234c4'
OUT=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/profile-readonly-20261031/ep3-all-compute-origins-v9.json')
assert not OUT.exists()
t0=time.monotonic(); before=DB.stat();h=hashlib.sha256()
with DB.open('rb') as f:
    for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
assert h.hexdigest()==SHA
db=sqlite3.connect(DB.as_uri()+'?mode=ro&immutable=1',uri=True);db.row_factory=sqlite3.Row
db.execute('PRAGMA query_only=ON')
db.set_progress_handler(lambda:1 if time.monotonic()-t0>140 else 0,100000)
def chunks(xs,n=800):
    xs=sorted(xs)
    for i in range(0,len(xs),n):yield xs[i:i+n]
def select_in(sql,values):
    out=[]
    for xs in chunks(values):
        out.extend(dict(r) for r in db.execute(sql.replace('{ids}',','.join('?' for _ in xs)),xs))
        assert len(out)<=1500000,'query row budget'
    return out
tasks=[dict(r) for r in db.execute('''SELECT t.globalTaskId,t.startNs,t.endNs,t.deviceId,t.streamId,t.taskId,t.connectionId,t.modelId,
 n.value AS kernel_name,op.value AS op_type,ish.value AS input_shapes,osh.value AS output_shapes,
 idt.value AS input_dtypes,odt.value AS output_dtypes,c.blockNum,c.mixBlockNum
 FROM TASK t JOIN COMPUTE_TASK_INFO c ON c.globalTaskId=t.globalTaskId
 LEFT JOIN STRING_IDS n ON n.id=c.name LEFT JOIN STRING_IDS op ON op.id=c.opType
 LEFT JOIN STRING_IDS ish ON ish.id=c.inputShapes LEFT JOIN STRING_IDS osh ON osh.id=c.outputShapes
 LEFT JOIN STRING_IDS idt ON idt.id=c.inputDataTypes LEFT JOIN STRING_IDS odt ON odt.id=c.outputDataTypes
 WHERE t.deviceId=3 ORDER BY t.startNs,t.streamId LIMIT 200001''')]
assert 0<len(tasks)<=200000
assert len({t['globalTaskId'] for t in tasks})==len(tasks),'compute globalTaskId uniqueness'
connections={t['connectionId'] for t in tasks if t['connectionId'] is not None}
links=select_in('SELECT id,connectionId FROM CONNECTION_IDS WHERE connectionId IN ({ids})',connections)
host_ids={r['id'] for r in links}
apis=select_in('''SELECT p.rowid AS api_rowid,p.startNs,p.endNs,p.globalTid,p.connectionId,p.callchainId,n.value AS name
 FROM PYTORCH_API p LEFT JOIN STRING_IDS n ON n.id=p.name WHERE p.connectionId IN ({ids})''',host_ids)
assert len({a['api_rowid'] for a in apis})==len(apis)
chain_ids={a['callchainId'] for a in apis if a['callchainId'] is not None}
frame_rows=select_in('''SELECT c.id,c.stackDepth,c.stack AS frame_id,s.value AS frame
 FROM PYTORCH_CALLCHAINS c JOIN STRING_IDS s ON s.id=c.stack WHERE c.id IN ({ids}) ORDER BY c.id,c.stackDepth''',chain_ids)
frame_dict={};chains=defaultdict(list)
for r in frame_rows:
    frame_dict[r['frame_id']]=r['frame'];chains[r['id']].append(dict(stackDepth=r['stackDepth'],frame_id=r['frame_id']))
for fs in chains.values():fs.sort(key=lambda f:f['stackDepth'])
del frame_rows
apis_by_host=defaultdict(list)
for a in apis:apis_by_host[a['connectionId']].append(a['api_rowid'])
hosts_by_device=defaultdict(set)
for r in links:hosts_by_device[r['connectionId']].add(r['id'])
api_map={a['api_rowid']:a for a in apis}
phase_needles={'DUMMY_ANCESTOR':[': dummy_run',': _dummy_run',': execute_dummy_batch'],
               'PROPOSAL_ANCESTOR':[': propose',': _propose'],
               'MODEL_EXECUTE_ANCESTOR':[': execute_model'],
               'PROFILE_ANCESTOR':[': profile_run'],
               'CAPTURE_ANCESTOR':[': capture_model',': _capture_model',': capture']}
chain_context={};project_file_by_api={};project_leaf_by_api={}
parse=re.compile(r'^(.*?)\((\d+)\): (.*)$')
for cid,fs in chains.items():
    texts=[frame_dict[f['frame_id']] for f in fs]
    chain_context[cid]=[key for key,needles in phase_needles.items() if any(needle in x for needle in needles for x in texts)]
for a in apis:
    texts=[frame_dict[f['frame_id']] for f in chains.get(a['callchainId'],[])]
    project=[x for x in texts if '/vllm_ascend/' in x or '/vllm/vllm/' in x]
    project_file_by_api[a['api_rowid']]=sorted({parse.match(x).group(1) if parse.match(x) else x for x in project})
    project_leaf_by_api[a['api_rowid']]=project[0] if project else None
type_rows=defaultdict(list);project_files=Counter();project_leaves=Counter();context_counts=Counter()
for t in tasks:
    ids=sorted({i for host in hosts_by_device.get(t['connectionId'],[]) for i in apis_by_host.get(host,[])})
    t['api_rowids']=ids
    context=sorted({x for i in ids for x in chain_context.get(api_map[i]['callchainId'],[])})
    t['recorded_host_context_flags']=context
    t['context_is_actual_device_phase']=False
    type_rows[t['op_type']].append(t);context_counts['+'.join(context) or 'NO_NAMED_CONTEXT']+=1
for a in apis:
    project_files.update(project_file_by_api[a['api_rowid']])
    if project_leaf_by_api[a['api_rowid']]:project_leaves[project_leaf_by_api[a['api_rowid']]]+=1
summaries=[]
for typ,ts in sorted(type_rows.items(),key=lambda x:str(x[0])):
    summaries.append(dict(op_type=typ,tasks=len(ts),linked_tasks=sum(bool(t['api_rowids']) for t in ts),
        multi_api_tasks=sum(len(t['api_rowids'])>1 for t in ts),
        linked_with_no_callchain_tasks=sum(bool(t['api_rowids']) and not any(chains.get(api_map[i]['callchainId']) for i in t['api_rowids']) for t in ts),
        recorded_context_hist=dict(Counter('+'.join(t['recorded_host_context_flags']) or 'NO_NAMED_CONTEXT' for t in ts)),
        project_leaf_counts=dict(Counter(project_leaf_by_api[i] or 'NO_PROJECT_FRAME' for t in ts for i in t['api_rowids'])),
        examples=[dict(globalTaskId=t['globalTaskId'],taskId=t['taskId'],startNs=t['startNs'],modelId=t['modelId'],api_rowids=t['api_rowids']) for t in ts[:3]]))
all_task_type_counts=[dict(r) for r in db.execute('''SELECT s.value AS task_type,COUNT(*) AS count FROM TASK t LEFT JOIN STRING_IDS s ON s.id=t.taskType WHERE t.deviceId=3 GROUP BY t.taskType''')]
db.close();after=DB.stat();assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
summary=dict(created_utc=datetime.now(timezone.utc).isoformat(),database=str(DB),database_sha256=SHA,database_stat_unchanged=True,
    compute_tasks=len(tasks),compute_types=len(summaries),linked_compute_tasks=sum(bool(t['api_rowids']) for t in tasks),
    unlinked_compute_tasks=sum(not t['api_rowids'] for t in tasks),multi_api_compute_tasks=sum(len(t['api_rowids'])>1 for t in tasks),
    linked_APIs=len(apis),distinct_callchains=len(chain_ids),callchains_with_frames=len(chains),frame_rows=sum(len(v) for v in chains.values()),
    distinct_frame_strings=len(frame_dict),project_source_files=len(project_files),project_leaf_locations=len(project_leaves),
    all_task_type_counts=all_task_type_counts,recorded_host_context_counts=dict(context_counts),
    no_target_source_read=True,no_device_or_inference_work=True,whole_modules_closed=0,overall_complete=False,
    limits=['Complete TASK/COMPUTE relation in this single fixed EP3 database only; not the16-rank official two-window capture.',
            'A linked API is a recorded submission candidate, not tensor identity or proof of exact historical loaded source bytes.',
            'Host callstack dummy/proposal/capture context is not actual device replay/client phase; modelId alone does not resolve it.',
            'Multiple API matches and missing callchains remain explicit; no nearest-time fallback is invented.',
            'First project frame is only a display/index convention pending stackDepth-direction validation; full chains preserved.',
            'All TASK kinds counted, but semantic origin association here covers COMPUTE_TASK_INFO, not COMM/event/memcpy completeness.'])
full=dict(summary=summary,type_summaries=summaries,project_file_api_counts=dict(project_files),project_leaf_api_counts=dict(project_leaves),
          tasks=tasks,apis=apis,connection_links=links,callchains={str(k):v for k,v in chains.items()},frame_dictionary={str(k):v for k,v in frame_dict.items()})
payload=json.dumps(full,ensure_ascii=False,separators=(',',':'))+'\n';assert len(payload.encode())<200*1024*1024
OUT.write_text(payload)
public=dict(summary=summary,full_result=str(OUT),result_sha256=hashlib.sha256(payload.encode()).hexdigest(),result_bytes=len(payload.encode()),
            project_file_api_counts=dict(project_files),type_counts=[{k:r[k] for k in ['op_type','tasks','linked_tasks','multi_api_tasks','linked_with_no_callchain_tasks']} for r in summaries])
assert len(json.dumps(public,ensure_ascii=False).encode())<=32000
print(json.dumps(public,ensure_ascii=False))
