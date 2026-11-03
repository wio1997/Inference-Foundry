from pathlib import Path
from datetime import datetime,timezone
import sqlite3,hashlib,socket,json

assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
DB=Path('/data/tiankuan/wio/migration-166-20261030/source169-project/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029/runtime/diagnostic-traces/dp0_pp0_tp3_dcp0_ep3_rank3_2128_20261029023525650_ascend_pt/ASCEND_PROFILER_OUTPUT/ascend_pytorch_profiler_3.db')
SHA='084b8718888e33271435ec79d7573c95ca1615f3e7484c807ea24590fad234c4'
before=DB.stat();h=hashlib.sha256()
with DB.open('rb') as f:
    for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
assert h.hexdigest()==SHA
db=sqlite3.connect(DB.as_uri()+'?mode=ro&immutable=1',uri=True);db.row_factory=sqlite3.Row;db.execute('PRAGMA query_only=ON')
TARGETS=[
 ('F11_KEY_PROJECTION_L2',39806,1793241328706472155,1793241328706483955),
 ('F11_KEY_NORM_L2',39807,1793241328706513515,1793241328706529075),
 ('F11_KEY_QUANT_L2',39809,1793241328708263955,1793241328708287295),
 ('F11_KEY_WRITE_L2',39810,1793241328708316515,1793241328708333855),
 ('F11_SCALE_WRITE_L2',39812,1793241328708964855,1793241328708984735),
 ('F23_MERGE_L36',41461,1793241329818661855,1793241329818693315),
 ('F23_HCPOST_L36',41462,1793241329818693335,1793241329818815615),
]
frames_filter=['indexer.py','dsa_v41','cache','fused_moe','moe_runner','routed_experts','moe_comm','prepare_finalize','model.py','reduce','sequence_parallel','hc_post','triton','_foreach']


def get_frames(callchain):
    rows=[dict(x) for x in db.execute('SELECT c.id,c.stackDepth,s.value AS frame FROM PYTORCH_CALLCHAINS c JOIN STRING_IDS s ON s.id=c.stack WHERE c.id=? ORDER BY c.stackDepth',(callchain,))]
    selected=[r for r in rows if any(s in (r['frame'] or '') for s in frames_filter)]
    return {'all_frames':rows,'selected_frames':selected,'selected_frame_count':len(selected)}


def api_for_connection(connection):
    ids=[x[0] for x in db.execute('SELECT id FROM CONNECTION_IDS WHERE connectionId=?',(connection,))]
    if not ids:return []
    rows=[dict(x) for x in db.execute('SELECT p.startNs,p.endNs,p.globalTid,p.connectionId,p.callchainId,s.value AS name FROM PYTORCH_API p LEFT JOIN STRING_IDS s ON s.id=p.name WHERE p.connectionId IN ('+','.join('?'*len(ids))+') ORDER BY p.startNs',ids)]
    assert len(rows)<30
    for r in rows:r['frames']=get_frames(r['callchainId']) if r['callchainId'] is not None else {'all_frames':[],'selected_frames':[],'selected_frame_count':0}
    return rows


items=[]
for label,tid,s,e in TARGETS:
    rows=[dict(r) for r in db.execute('''SELECT t.startNs,t.endNs,t.deviceId,t.streamId,t.taskId,t.globalTaskId,t.connectionId,t.modelId,n.value AS kernel_name,ty.value AS task_type_name
       FROM TASK t LEFT JOIN COMPUTE_TASK_INFO c ON c.globalTaskId=t.globalTaskId LEFT JOIN STRING_IDS n ON n.id=c.name LEFT JOIN STRING_IDS ty ON ty.id=t.taskType
       WHERE t.deviceId=3 AND t.streamId=47 AND t.taskId=? AND t.startNs=? AND t.endNs=?''',(tid,s,e))]
    named=[r for r in rows if r['kernel_name']];assert len(named)==1,(label,'nonunique_task',len(rows))
    items.append({'label':label,'exact_task_rows':rows,'api_candidates':api_for_connection(named[0]['connectionId'])})
start_device=1793241329811411635;end_device=1793241329812070995
comm=[dict(x) for x in db.execute('''SELECT c.*,n.value AS operation_name,g.value AS group_name FROM COMMUNICATION_OP c
    LEFT JOIN STRING_IDS n ON n.id=c.opName LEFT JOIN STRING_IDS g ON g.id=c.groupName
    WHERE c.deviceId=3 AND c.startNs<? AND c.endNs>? ORDER BY c.startNs''',(end_device,start_device))]
assert len(comm)<50
host=[dict(x) for x in db.execute('''SELECT p.startNs,p.endNs,p.globalTid,p.connectionId,p.callchainId,s.value AS name
    FROM PYTORCH_API p JOIN STRING_IDS s ON s.id=p.name
    WHERE p.globalTid=9139690408016 AND CAST(p.startNs AS INTEGER)>=1793241329141824170
    AND CAST(p.startNs AS INTEGER)<=1793241329147301570
    AND (s.value LIKE '%Hccl%' OR s.value LIKE '%all_to_all%' OR s.value LIKE '%alltoall%') ORDER BY p.startNs''')]
assert len(host)<30
for x in host:x['frames']=get_frames(x['callchainId']) if x['callchainId'] is not None else {'all_frames':[],'selected_frames':[],'selected_frame_count':0}
db.close();after=DB.stat();assert before.st_size==after.st_size and before.st_mtime_ns==after.st_mtime_ns
result={'created_utc':datetime.now(timezone.utc).isoformat(),'database':str(DB),'database_sha256':SHA,'database_stat_unchanged':True,'items':items,
        'F15':{'device_interval_ns':[start_device,end_device],'communication_ops':comm,'host_communication_candidates':host},
        'no_device_or_inference_work':True,
        'limits':['Connection links and stored submission stacks identify recorded call locations; not historical loaded file hashes, tensor identity or numerical semantics.',
                  'One archived rank and two selected scopes; broader sidecar candidates do not inherit all these stack bindings.',
                  'Device interval and host interval separately chosen from same scope exports; clocks not subtracted.',
                  'Communication interval intersection is not proof of a tensor dependency, critical path or releasable time.']}
out=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/profile-readonly-20261031/ep3-cache-merge-stack-evidence.json')
assert not out.exists(),'refuse_overwrite'
out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')


def public_api(r):
    return {**{k:v for k,v in r.items() if k!='frames'},'selected_frames':r['frames']['selected_frames'][:20],
            'selected_frame_count':r['frames']['selected_frame_count'],'frame_display_truncated':r['frames']['selected_frame_count']>20}


print(json.dumps({'created_utc':result['created_utc'],'database_sha256':SHA,'database_stat_unchanged':True,'full_result':str(out),
                  'items':[{'label':x['label'],'exact_task_rows':x['exact_task_rows'],'api_candidates':[public_api(a) for a in x['api_candidates']]} for x in items],
                  'F15':{'device_interval_ns':[start_device,end_device],'communication_ops':comm,'host_communication_candidates':[public_api(x) for x in host]},
                  'no_device_or_inference_work':True,'limits':result['limits']},ensure_ascii=False))
