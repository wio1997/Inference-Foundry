from pathlib import Path
from collections import defaultdict,Counter
from datetime import datetime,timezone
import json,hashlib,sqlite3,socket,time
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
P=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/hostprof-20261030T212812Z-360e0c72/analysis')
def pin(path,sha):
 b=path.read_bytes();assert hashlib.sha256(b).hexdigest()==sha;return json.loads(b)
idx=pin(P/'native-communications-v15/index.json','20ea4452458001c372c2d2c9b5cd9f4f13d951cfa7c77bb7e5e4bf3faa259016')
OUT=P/'native-communications-v15/communication-fullchains-v15.json';assert not OUT.exists();canonical={};dbrows=[];t0=time.monotonic()
for ref in idx['databases']:
 d=pin(Path(ref['path']),ref['sha256']);di=d['database_index'];apis={a['api_rowid']:a for op in d['COMM_OP_records']for a in op['native_PYTORCH_API_candidates']}
 db=sqlite3.connect(Path(d['database_path']).as_uri()+'?mode=ro&immutable=1',uri=True);db.row_factory=sqlite3.Row;db.execute('PRAGMA query_only=ON');db.set_progress_handler(lambda:int(time.monotonic()-t0>145),100000)
 source=[dict(r)for r in db.execute('SELECT p.rowid AS api_rowid,CAST(p.startNs AS INTEGER)AS startNs,CAST(p.endNs AS INTEGER)AS endNs,p.globalTid,p.connectionId,p.callchainId,s.value AS name_text FROM PYTORCH_API p LEFT JOIN STRING_IDS s ON s.id=p.name WHERE p.callchainId IS NOT NULL')]
 bythread=defaultdict(list)
 for a in source:bythread[a['globalTid']].append(a)
 sources={};need=set()
 for rid,a in apis.items():
  if a['callchainId']is not None:
   sources[rid]=dict(kind='DIRECT_NATIVE_API_SOURCE',candidates=[a]);need.add(a['callchainId'])
  else:
   candidates=[p for p in bythread[a['globalTid']]if p['startNs']<=a['startNs']and a['endNs']<=p['endNs']]
   if candidates:
    width=min(p['endNs']-p['startNs']for p in candidates);candidates=[p for p in candidates if p['endNs']-p['startNs']==width]
   sources[rid]=dict(kind='SAME_THREAD_ENCLOSING_SOURCE_CANDIDATE',candidates=candidates);need.update(p['callchainId']for p in candidates)
 fs=defaultdict(list)
 if need:
  vals=','.join(str(int(i))for i in need)
  for r in db.execute('SELECT c.id,c.stackDepth,s.value AS frame FROM PYTORCH_CALLCHAINS c JOIN STRING_IDS s ON s.id=c.stack WHERE c.id IN ('+vals+') ORDER BY c.id,c.stackDepth'):fs[r['id']].append(dict(stackDepth=r['stackDepth'],frame=r['frame']))
 for group in sources.values():
  for a in group['candidates']:
   frames=fs.get(a['callchainId'],[]);key=hashlib.sha256(json.dumps(frames,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()if frames else None
   if key:canonical[key]=frames
   a['chain_key']=key;a['project_frames']=[r['frame']for r in frames if '/vllm_ascend/'in r['frame']or '/vllm/vllm/'in r['frame']]
 records=[]
 for op in d['COMM_OP_records']:
  records.append(dict(COMM_OP_rowid=op['COMM_OP_rowid'],opId=op['opId'],name=op['opName_text'],count=op['count'],dtype=op['dataType_text'],source_groups=[dict(native_API_rowid=a['api_rowid'],native_API=a,**sources[a['api_rowid']])for a in op['native_PYTORCH_API_candidates']],graph_control_candidates=op['graph_control_container_candidates']))
 dbrows.append(dict(database_index=di,records=records));db.close()
allops=[r for d in dbrows for r in d['records']]
summary=dict(created_utc=datetime.now(timezone.utc).isoformat(),databases=len(dbrows),COMM_OP_records=len(allops),with_direct_fullchain=sum(any(g['kind']=='DIRECT_NATIVE_API_SOURCE'and any(a['chain_key']for a in g['candidates'])for g in r['source_groups'])for r in allops),with_enclosing_source_candidate=sum(any(g['kind']=='SAME_THREAD_ENCLOSING_SOURCE_CANDIDATE'and any(a['chain_key']for a in g['candidates'])for g in r['source_groups'])for r in allops),canonical_fullchains=len(canonical),no_target_source_or_device_work=True,overall_complete=False,limits=['Includes native communication API chains even when their device node is absent from COMPUTE_TASK_INFO. Compute-onlyfullchainindexisnotacommunicationcompletenessproof.', 'Same-threadparentcandidateassociationisdistinctfromdirectnativeAPIchain; allambiguityretained. Graph-onlycollectivesarekeptwithoutfabricatedPythonstacks.'])
payload=(json.dumps(dict(summary=summary,databases=dbrows,canonical_fullchains=canonical),ensure_ascii=False,separators=(',',':'))+'\n').encode();OUT.write_bytes(payload)
leaf=Counter(a['project_frames'][0]for d in dbrows for r in d['records']for g in r['source_groups']for a in g['candidates']if a['project_frames'])
print(json.dumps(dict(summary=summary,first_project_source_counts=dict(leaf),full_result=str(OUT),result_sha256=hashlib.sha256(payload).hexdigest()),ensure_ascii=False))
