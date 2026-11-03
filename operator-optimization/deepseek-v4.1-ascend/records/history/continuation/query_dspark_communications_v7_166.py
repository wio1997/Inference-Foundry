from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timezone
import socket, sqlite3, json, hashlib

assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
A=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/profile-readonly-20261031')
source=A/'ep3-dspark-and-moe-linked-tasks-v7-v2.json'
raw=source.read_bytes()
assert hashlib.sha256(raw).hexdigest()=='e806fc67206b77a7d90d1d70f289eab84656e78fb813c1636509c901d51c4a98'
d=json.loads(raw)
chains={int(k):v for k,v in d['callchains'].items() if any('/models/deepseek_v41/dspark.py(' in f['frame'] for f in v)}
DB=Path(d['database'])
before=DB.stat()
h=hashlib.sha256()
with DB.open('rb') as f:
    for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
assert h.hexdigest()==d['database_sha256']
db=sqlite3.connect(DB.as_uri()+'?mode=ro&immutable=1',uri=True);db.row_factory=sqlite3.Row;db.execute('PRAGMA query_only=ON')
apis=[dict(r) for r in db.execute('''SELECT p.startNs,p.endNs,p.globalTid,p.connectionId,p.callchainId,n.value AS name FROM PYTORCH_API p
 JOIN STRING_IDS n ON n.id=p.name WHERE p.callchainId IN ('''+','.join(str(x) for x in chains)+") AND n.value LIKE 'Hccl%' ORDER BY p.startNs")]
assert len(apis)<1000
host_ids={a['connectionId'] for a in apis}
links=[dict(r) for r in db.execute('SELECT id,connectionId FROM CONNECTION_IDS') if r['id'] in host_ids]
device_ids={r['connectionId'] for r in links}
comm=[dict(r) for r in db.execute('''SELECT c.*,n.value AS name,g.value AS group_name FROM COMMUNICATION_OP c
 JOIN STRING_IDS n ON n.id=c.opName LEFT JOIN STRING_IDS g ON g.id=c.groupName
 WHERE c.deviceId=3 AND c.connectionId IN ('''+','.join(str(x) for x in device_ids)+') ORDER BY c.startNs')]
by_conn=defaultdict(list)
for c in comm:by_conn[c['connectionId']].append(c)
rows=[]
for a in apis:
    fs=chains[a['callchainId']]
    text='\n'.join(f['frame'] for f in fs)
    if 'dspark.py(332): compute_logits' in text:role='F30_base_logits'
    elif 'dspark.py(249): forward' in text:role='F28_embedding'
    elif 'dspark.py(282): combine_hidden_states' in text:role='F27_context_projection'
    elif 'dspark.py(275): forward' in text:role='F30_draft_collapse_output'
    elif 'dspark.py(265): forward' in text:role='draft_decoder'
    else:role='OTHER_DSPARK_CALLSITE'
    ownlinks=[l for l in links if l['id']==a['connectionId']]
    cs=[c for l in ownlinks for c in by_conn[l['connectionId']]]
    unique={tuple(c[k] for k in ['opId','connectionId','startNs','endNs']):c for c in cs}
    rows.append(dict(role=role,api=a,links=ownlinks,communication=list(unique.values()),source_frames=[f for f in fs if '/deepseek_v41/' in f['frame']]))
db.close()
after=DB.stat()
assert before.st_size==after.st_size and before.st_mtime_ns==after.st_mtime_ns
groups=defaultdict(list)
for r in rows:groups[r['role']].append(r)
summary=dict(created_utc=datetime.now(timezone.utc).isoformat(),database_sha256=d['database_sha256'],database_stat_unchanged=True,
             source_profile_result_sha256=hashlib.sha256(raw).hexdigest(),host_HCCL_APIs=len(apis),communication_ops=len(comm),
             per_role={role:dict(api_count=len(rs),api_name_counts=dict(Counter(r['api']['name'] for r in rs)),
                                matched_COMM_count_hist=dict(Counter(len(r['communication']) for r in rs)),
                                rank_size_hist=dict(Counter(c['rankSize'] for r in rs for c in r['communication']))) for role,rs in groups.items()},
             no_device_or_inference_work=True,overall_complete=False,
             limits=['One archived rank; no crossrank participant/tensor attestation.',
                     'COMM linked directly through stored connection IDs; TASK/COMPUTE_TASK_INFO omissions do not prove collective absence.',
                     'Nested native MC2 internal communication is not decomposed by these HCCL API records.'])
out=A/'ep3-dspark-communication-links-v7.json'
assert not out.exists(),'refuse_overwrite'
payload=json.dumps(dict(summary=summary,rows=rows),ensure_ascii=False,indent=2)+'\n';out.write_text(payload)
print(json.dumps(dict(summary=summary,full_result=str(out),result_sha256=hashlib.sha256(payload.encode()).hexdigest(),
                      examples=[dict(role=role,api=rs[0]['api'],links=rs[0]['links'],communication=rs[0]['communication'],source_frames=rs[0]['source_frames']) for role,rs in groups.items()]),ensure_ascii=False))
