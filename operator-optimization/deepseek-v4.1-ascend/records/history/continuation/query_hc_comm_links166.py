from pathlib import Path
import sqlite3, socket, hashlib, json

assert socket.gethostname() == 'bm-jn-zs-zone1-910C-64G-10-105'
p = Path('/data/tiankuan/wio/migration-166-20261030/source169-project/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029/runtime/diagnostic-traces/dp0_pp0_tp3_dcp0_ep3_rank3_2128_20261029023525650_ascend_pt/ASCEND_PROFILER_OUTPUT/ascend_pytorch_profiler_3.db')
sha = '084b8718888e33271435ec79d7573c95ca1615f3e7484c807ea24590fad234c4'
h = hashlib.sha256()
before = p.stat()
with p.open('rb') as f:
    for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
assert h.hexdigest() == sha
db = sqlite3.connect(p.as_uri()+'?mode=ro&immutable=1',uri=True)
db.row_factory = sqlite3.Row
db.execute('PRAGMA query_only=ON')
targets = [(102723,'AUX_L37'),(103719,'MTP_RESIDUAL'),(103752,'MTP_PREMIX'),(103954,'DSPARK_CONTEXT_FC')]
items = []
for connection, label in targets:
    comm = [dict(r) for r in db.execute('SELECT c.*,n.value AS name FROM COMMUNICATION_OP c JOIN STRING_IDS n ON n.id=c.opName WHERE c.deviceId=3 AND c.connectionId=?',(connection,))]
    assert len(comm) == 1
    links = [dict(r) for r in db.execute('SELECT * FROM CONNECTION_IDS WHERE connectionId=?',(connection,))]
    assert 0 < len(links) < 20
    ids = sorted({r['id'] for r in links})
    apis = [dict(r) for r in db.execute('SELECT p.startNs,p.endNs,p.globalTid,p.connectionId,p.callchainId,n.value AS name FROM PYTORCH_API p JOIN STRING_IDS n ON n.id=p.name WHERE p.connectionId IN ('+','.join('?'*len(ids))+')',ids)]
    assert len(apis) < 20
    for a in apis:
        a['frames'] = [dict(r) for r in db.execute('SELECT c.stackDepth,s.value AS frame FROM PYTORCH_CALLCHAINS c JOIN STRING_IDS s ON s.id=c.stack WHERE c.id=? ORDER BY c.stackDepth',(a['callchainId'],))]
    items.append(dict(label=label,device_connection=connection,communication=comm[0],links=links,api_candidates=apis))
db.close()
after = p.stat()
assert before.st_size == after.st_size and before.st_mtime_ns == after.st_mtime_ns
result = dict(database_sha256=sha,database_stat_unchanged=True,items=items,no_device_or_inference_work=True,
              limits=['One rank four stored collective connections; not complete crossrank message/tensor/ownership attestation.',
                      'Connection-to-API binding identifies submission sites; dtype/count decoding and per-rank role need stated contract.'])
out = Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/profile-readonly-20261031/ep3-hc-comm-links.json')
assert not out.exists()
out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
for item in items:
    for a in item['api_candidates']:
        a['frames'] = [f for f in a['frames'] if any(x in f['frame'] for x in ['deepseek_v41','parallel_state.py','distributed/communication'])][:12]
result['full_result'] = str(out)
print(json.dumps(result,ensure_ascii=False))
