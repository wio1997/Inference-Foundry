from pathlib import Path
from datetime import datetime,timezone
import socket,json,hashlib,sqlite3

assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
root=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/profile-readonly-20261031')
prior=json.loads((root/'ep3-cache-merge-stack-evidence.json').read_text())
DB=Path(prior['database']);before=DB.stat();h=hashlib.sha256()
with DB.open('rb') as f:
    for x in iter(lambda:f.read(4*1024*1024),b''):h.update(x)
assert h.hexdigest()==prior['database_sha256']=='084b8718888e33271435ec79d7573c95ca1615f3e7484c807ea24590fad234c4'
merge=next(x for x in prior['items'] if x['label']=='F23_MERGE_L36');assert len(merge['api_candidates'])==1
top=merge['api_candidates'][0]['frames']['all_frames'][:8]
db=sqlite3.connect(DB.as_uri()+'?mode=ro&immutable=1',uri=True);db.row_factory=sqlite3.Row;db.execute('PRAGMA query_only=ON')
connections=[dict(x) for x in db.execute('SELECT id,connectionId FROM CONNECTION_IDS WHERE id=431394')]
reverse=[dict(x) for x in db.execute('SELECT id,connectionId FROM CONNECTION_IDS WHERE connectionId=102545')]
comm=[dict(x) for x in db.execute('SELECT opId,connectionId,deviceId,rankSize,count FROM COMMUNICATION_OP WHERE opId=247')]
db.close();after=DB.stat();assert before.st_size==after.st_size and before.st_mtime_ns==after.st_mtime_ns
result={'created_utc':datetime.now(timezone.utc).isoformat(),'DB_SHA':prior['database_sha256'],'DB_stat_unchanged':True,
        'F23_top8_unfiltered_frames':top,'F23_API':merge['api_candidates'][0]['name'],
        'F15_host431394_device_connections':connections,'F15_device102545_reverse_connections':reverse,
        'F15_communication_op247':comm,'F15_host_api_to_device_comm_connection_match':any(x['connectionId']==102545 for x in connections) and any(x['connectionId']==102545 for x in comm),
        'no_device_or_inference_work':True,
        'limits':['Callchain is recorded submission stack, not a loaded-source-hash attestation or tensor identity.',
                  'Explicit communication connection link is local to thisDB/scope; no global cross-rank message identity or timing gain proved.']}
out=root/'ep3-merge-comm-proof.json';assert not out.exists(),'refuse_overwrite';out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({**result,'full_result':str(out)},ensure_ascii=False))
