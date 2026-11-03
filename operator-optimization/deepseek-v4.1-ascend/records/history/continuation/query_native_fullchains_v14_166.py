from pathlib import Path
from collections import defaultdict,Counter
from datetime import datetime,timezone
import hashlib,json,sqlite3,socket,time
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
P=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/hostprof-20261030T212812Z-360e0c72')
ROOT=P/'analysis/native-execution-identity-v14';b=(ROOT/'index.json').read_bytes();assert hashlib.sha256(b).hexdigest()=='a3aa47cdf9170785432ecbc18117a489f3fba37d89959f110f4fcbd8890c7dc4';index=json.loads(b)
OUT=ROOT/'native-compute-fullchains-v14.json';assert not OUT.exists()
t0=time.monotonic();canonical={};dbresults=[];contexts=Counter()
for ref in index['databases']:
    b=Path(ref['path']).read_bytes();assert hashlib.sha256(b).hexdigest()==ref['sha256'];d=json.loads(b);di=d['database_index']
    used={i for t in d['TASK_records']for i in t.get('native_linked_API_rowids',[])}
    apis=[a for a in d['linked_PYTORCH_APIs']if a['api_rowid']in used];ids={a['callchainId']for a in apis if a['callchainId']is not None}
    db=sqlite3.connect(Path(d['database_path']).as_uri()+'?mode=ro&immutable=1',uri=True);db.row_factory=sqlite3.Row;db.execute('PRAGMA query_only=ON');db.set_progress_handler(lambda:int(time.monotonic()-t0>145),100000)
    chains=defaultdict(list)
    for start in range(0,len(ids),600):
        batch=sorted(ids)[start:start+600]
        for r in db.execute('SELECT c.id,c.stackDepth,s.value AS frame FROM PYTORCH_CALLCHAINS c JOIN STRING_IDS s ON s.id=c.stack WHERE c.id IN ('+','.join('?'for _ in batch)+') ORDER BY c.id,c.stackDepth',batch):chains[r['id']].append(dict(stackDepth=r['stackDepth'],frame=r['frame']))
    db.close();cidkeys={}
    for cid,fs in chains.items():
        key=hashlib.sha256(json.dumps(fs,ensure_ascii=False,separators=(',',':')).encode()).hexdigest();canonical[key]=fs;cidkeys[cid]=key
    for a in apis:
        a['chain_key']=cidkeys.get(a['callchainId']);fs=chains.get(a['callchainId'],[]);a['project_frames']=[r['frame']for r in fs if '/vllm_ascend/'in r['frame']or '/vllm/vllm/'in r['frame']]
        contexts[a['project_frames'][0]if a['project_frames']else'NO_PROJECT_FRAME']+=1
    dbresults.append(dict(database_index=di,database_path=d['database_path'],database_sha256=d['database_sha256'],compute_native_API_records=len(apis),fullchains=sum(bool(a['chain_key'])for a in apis),chain_definitions=len(ids),API_records=apis))
summary=dict(created_utc=datetime.now(timezone.utc).isoformat(),databases=len(dbresults),native_linked_compute_API_records=sum(d['compute_native_API_records']for d in dbresults),records_with_fullchains=sum(d['fullchains']for d in dbresults),canonical_fullchains=len(canonical),first_project_locations=len(contexts),no_target_source_read=True,no_device_or_service_work=True,whole_modules_closed=0,overall_complete=False,
    limits=['Exact native-linked compute API chains only; graph records lacking an API link remain represented by native graph associations, not fabricated Python sources.', 'Canonicalization uses the complete ordered frame content including stackDepth; every original API/DB/callchainId retained.', 'First project frame is an index convention; full caller context must assign semantic role. No module ownership follows from a helper filename alone.'])
payload=(json.dumps(dict(summary=summary,databases=dbresults,canonical_fullchains=canonical,first_project_location_counts=dict(contexts)),ensure_ascii=False,separators=(',',':'))+'\n').encode();OUT.write_bytes(payload)
public=dict(summary=summary,full_result=str(OUT),result_sha256=hashlib.sha256(payload).hexdigest(),first_project_location_counts=dict(contexts))
if len(json.dumps(public).encode())>30000:public.pop('first_project_location_counts');public['display_truncated']=True
print(json.dumps(public,ensure_ascii=False))
