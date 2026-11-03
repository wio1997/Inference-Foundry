from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json, hashlib, sqlite3, socket
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
A=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/profile-readonly-20261031')
raw=(A/'ep3-dspark-and-moe-linked-tasks-v7-v2.json').read_bytes();assert hashlib.sha256(raw).hexdigest()=='e806fc67206b77a7d90d1d70f289eab84656e78fb813c1636509c901d51c4a98'
d=json.loads(raw)
selected=[]
for t in d['tasks']:
    fs=[f['frame'] for a in t['api_candidates'] for f in d['callchains'][str(a['callchainId'])]]
    if t['op_type']=='Tile':role='repeat'
    elif t['op_type']=='MatMulV2' and any('dspark.py(290): _project_shared_kv' in f for f in fs):role='context_projection'
    elif t['op_type']=='MatMulV2' and any('dspark.py(332): compute_logits' in f for f in fs):role='base_head'
    else:continue
    assert len(t['api_candidates'])==1
    selected.append(dict(role=role,taskId=t['taskId'],startNs=t['startNs'],endNs=t['endNs'],input_shapes=t['input_shapes'],callchainId=t['api_candidates'][0]['callchainId']))
assert len(selected)==65
DB=Path(d['database']);before=DB.stat();h=hashlib.sha256()
with DB.open('rb') as f:
    for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
assert h.hexdigest()==d['database_sha256']
db=sqlite3.connect(DB.as_uri()+'?mode=ro&immutable=1',uri=True);db.row_factory=sqlite3.Row;db.execute('PRAGMA query_only=ON')
ids=sorted({r['callchainId'] for r in selected});frames=defaultdict(list)
for f in db.execute('SELECT c.id,c.stackDepth,s.value AS frame FROM PYTORCH_CALLCHAINS c JOIN STRING_IDS s ON s.id=c.stack WHERE c.id IN ('+','.join(str(i) for i in ids)+') ORDER BY c.id,c.stackDepth'):
    frames[f['id']].append(dict(stackDepth=f['stackDepth'],frame=f['frame']))
db.close();after=DB.stat();assert before.st_size==after.st_size and before.st_mtime_ns==after.st_mtime_ns
for r in selected:
    fs=frames[r['callchainId']];texts=[f['frame'] for f in fs]
    dummy=any(': dummy_run' in f or ': _dummy_run' in f for f in texts)
    proposal=any(': _propose' in f or ': propose' in f for f in texts)
    r['phase']='DUMMY_ANCESTRY_CONFIRMED' if dummy else 'PROPOSAL_ANCESTRY_CONFIRMED' if proposal else 'ANCESTRY_UNRESOLVED'
    r['frames']=fs
    r['phase_frames']=[f for f in fs if any(s in f['frame'] for s in ['dummy_run','_propose','propose_draft','execute_model','profile_run','capture'])]
summary=dict(created_utc=datetime.now(timezone.utc).isoformat(),database_sha256=d['database_sha256'],database_stat_unchanged=True,
             source_result_sha256=hashlib.sha256(raw).hexdigest(),selected_tasks=len(selected),
             counts=[dict(role=role,phase=phase,count=n) for (role,phase),n in sorted(Counter((r['role'],r['phase']) for r in selected).items())],
             no_device_or_inference_work=True,overall_complete=False,
             limits=['Full recorded parent stack proves dummy/proposal call context, not formal-client phase, request identity or graph replay timing.',
                     'No timing from166 count snapshots is joined to these169 records.'])
out=A/'ep3-dspark-phase-ancestry-v7.json';assert not out.exists()
payload=json.dumps(dict(summary=summary,rows=selected),ensure_ascii=False,indent=2)+'\n';out.write_text(payload)
example={}
for r in selected:example.setdefault((r['role'],r['phase']),r)
print(json.dumps(dict(summary=summary,full_result=str(out),result_sha256=hashlib.sha256(payload.encode()).hexdigest(),
                      repeat_rows=[{k:v for k,v in r.items() if k!='frames'} for r in selected if r['role']=='repeat'],
                      examples=[{k:v for k,v in r.items() if k!='frames'} for r in example.values()]),ensure_ascii=False))
