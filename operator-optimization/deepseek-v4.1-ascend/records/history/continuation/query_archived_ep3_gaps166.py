from pathlib import Path
from collections import Counter,defaultdict
from datetime import datetime,timezone
import sqlite3,socket,hashlib,json,time

assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
DB=Path('/data/tiankuan/wio/migration-166-20261030/source169-project/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029/runtime/diagnostic-traces/dp0_pp0_tp3_dcp0_ep3_rank3_2128_20261029023525650_ascend_pt/ASCEND_PROFILER_OUTPUT/ascend_pytorch_profiler_3.db')
EXPECTED='084b8718888e33271435ec79d7573c95ca1615f3e7484c807ea24590fad234c4'
before=DB.stat();h=hashlib.sha256()
with DB.open('rb') as f:
    for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
assert h.hexdigest()==EXPECTED,('profile_db_sha_mismatch',h.hexdigest())
con=sqlite3.connect(DB.as_uri()+'?mode=ro&immutable=1',uri=True);con.row_factory=sqlite3.Row
con.execute('PRAGMA query_only=ON')
SELECT='''SELECT t.startNs,t.endNs,t.deviceId,t.streamId,t.taskId,t.globalTaskId,t.globalPid,t.connectionId,t.modelId,t.contextId,t.taskType,
 ty.value AS task_type_name, nm.value AS kernel_name, op.value AS op_type
 FROM TASK t LEFT JOIN STRING_IDS ty ON ty.id=t.taskType
 LEFT JOIN COMPUTE_TASK_INFO ci ON ci.globalTaskId=t.globalTaskId
 LEFT JOIN STRING_IDS nm ON nm.id=ci.name LEFT JOIN STRING_IDS op ON op.id=ci.opType'''
targets=[
 ('F22_LONG_L4',43276,1793241333677204175,1793241333677233735,43277,1793241333678998935,1793241333679051855),
 ('F22_SHORT_L38',41551,1793241329845050015,1793241329845092195,41552,1793241329845093055,1793241329845168495),
 ('F17_LONG_L14',43845,1793241334237441635,1793241334237751795,43846,1793241334240121875,1793241334240424275),
]


def union(intervals):
    end=None;total=0
    for a,b in sorted(intervals):
        if b<=a:continue
        if end is None or a>end:total+=b-a;end=b
        elif b>end:total+=b-end;end=b
    return total


def api_candidates(anchor):
    connection=anchor['connectionId']
    ids=[x[0] for x in con.execute('SELECT id FROM CONNECTION_IDS WHERE connectionId=?',(connection,))]
    out={'device_connection':connection,'framework_candidate_connection_ids':ids}
    if ids:
        sql='SELECT p.startNs,p.endNs,p.globalTid,p.connectionId,p.callchainId,s.value AS name FROM PYTORCH_API p LEFT JOIN STRING_IDS s ON s.id=p.name WHERE p.connectionId IN ('+','.join('?'*len(ids))+') ORDER BY p.startNs LIMIT 21'
        rows=[dict(r) for r in con.execute(sql,ids)];out['pytorch_api_candidates']=rows[:20];out['pytorch_candidate_display_truncated']=len(rows)>20
    return out


items=[]
for label,aid,astart,aend,bid,bstart,bend in targets:
    anchor_a=[dict(r) for r in con.execute(SELECT+' WHERE t.deviceId=3 AND t.streamId=47 AND t.taskId=? AND t.startNs=? AND t.endNs=?',(aid,astart,aend))]
    anchor_b=[dict(r) for r in con.execute(SELECT+' WHERE t.deviceId=3 AND t.streamId=47 AND t.taskId=? AND t.startNs=? AND t.endNs=?',(bid,bstart,bend))]
    na=[x for x in anchor_a if x['kernel_name']];nb=[x for x in anchor_b if x['kernel_name']]
    assert len(na)==len(nb)==1,(label,'anchor_not_unique',len(anchor_a),len(anchor_b),len(na),len(nb))
    records=[dict(r) for r in con.execute(SELECT+' WHERE t.deviceId=3 AND t.startNs<? AND t.endNs>? ORDER BY t.startNs,t.streamId LIMIT 5001',(bstart,aend))]
    assert len(records)<=5000,(label,'interval_row_budget_exceeded')
    grouped=defaultdict(list)
    for row in records:
        grouped[row['kernel_name'] or row['task_type_name'] or 'UNRESOLVED'].append(row)
    aggregates=[]
    for name,rows in sorted(grouped.items()):
        aggregates.append({'name':name,'records':len(rows),'stream_ids':sorted({r['streamId'] for r in rows}),
                           'clipped_record_union_ns':union([(max(aend,r['startNs']),min(bstart,r['endNs'])) for r in rows])})
    item={'label':label,'gap_ns':bstart-aend,'start_ns':aend,'end_ns':bstart,
          'anchor_a':anchor_a,'anchor_b':anchor_b,'api_a':api_candidates(na[0]),'api_b':api_candidates(nb[0]),
          'overlap_task_rows':records,'overlap_task_count':len(records),'all_name_aggregates':aggregates,
          'all_record_union_ns':union([(max(aend,r['startNs']),min(bstart,r['endNs'])) for r in records]),
          'same_stream47_record_union_ns':union([(max(aend,r['startNs']),min(bstart,r['endNs'])) for r in records if r['streamId']==47]),
          'event_wait_like_records':sum(any(word in (r['task_type_name'] or '').upper() for word in ['EVENT','WAIT','NOTIFY','SYNC']) for r in records)}
    items.append(item)
con.close();after=DB.stat();assert before.st_size==after.st_size and before.st_mtime_ns==after.st_mtime_ns
result={'created_utc':datetime.now(timezone.utc).isoformat(),'host':socket.gethostname(),'database':str(DB),'database_sha256':EXPECTED,
        'database_stat_unchanged':True,'items':items,'no_device_work':True,
        'limits':['Single archived rank database, not all device processes or unexported work.',
                  'TASK intervals/labels, including EVENT WAIT, are exported records and are not exclusive hardware idle/wait/busy measurements.',
                  'No event-object identity or record-to-wait causal edge is proved by proximity or connection.',
                  'Long/short examples are different layers/digests; not controlled equal-work performance comparison.',
                  'Host and device clocks are not subtracted; API connection candidates do not prove tensor edges.']}
out=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/profile-readonly-20261031')
out.mkdir(parents=True,exist_ok=True);path=out/'ep3-gap-task-evidence.json';assert not path.exists(),'refuse_overwrite_prior_receipt'
path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'created_utc':result['created_utc'],'database_sha256':EXPECTED,'database_stat_unchanged':True,'full_result':str(path),
                  'items':[{k:x[k] for k in ['label','gap_ns','overlap_task_count','all_name_aggregates','all_record_union_ns','same_stream47_record_union_ns','event_wait_like_records','anchor_a','anchor_b','api_a','api_b']} for x in items],
                  'no_device_work':True,'limits':result['limits']},ensure_ascii=False))
