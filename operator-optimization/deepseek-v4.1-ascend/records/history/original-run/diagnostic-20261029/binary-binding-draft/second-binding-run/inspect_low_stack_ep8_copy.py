"""Single pre-existing EP8 DB; bounded CPU-only read-only operator audit."""
from pathlib import Path
import sqlite3,json,hashlib,statistics
base=Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-20261028/diagnostic/runtime/diagnostic-traces')
rank=base/'dp1_pp0_tp0_dcp0_ep8_rank0_727_20261028020238580_ascend_pt';info=rank/'profiler_info_8.json';db=rank/'ASCEND_PROFILER_OUTPUT/ascend_pytorch_profiler_8.db'
raw=info.read_bytes();assert hashlib.sha256(raw).hexdigest()=='a249b01a52dc6d9cbbd34953f2573b52d1d9f7280ff2d561b228b4a9417eb544'
assert not Path(str(db)+'-wal').exists() and not Path(str(db)+'-journal').exists();st0=db.stat()
h=hashlib.sha256()
with db.open('rb') as f:
 for b in iter(lambda:f.read(1048576),b''):h.update(b)
c=sqlite3.connect(db.resolve().as_uri()+'?mode=ro&immutable=1',uri=True);c.row_factory=sqlite3.Row;c.execute('PRAGMA query_only=ON')
def rows(sql,args=(),cap=200000):
 x=[dict(r) for r in c.execute(sql,args).fetchmany(cap+1)]
 if len(x)>cap:raise RuntimeError('bounded query exceeded')
 return x
settings=json.loads(raw)['config']; copies=rows('''SELECT a.startNs,a.endNs,a.globalTid,a.connectionId,s.value name FROM CANN_API a JOIN STRING_IDS s ON s.id=a.name WHERE s.value IN ('aclnnInplaceCopy','Slice_Tiling')''')
cs=[x for x in copies if x['name']=='aclnnInplaceCopy'];ss=[x for x in copies if x['name']=='Slice_Tiling']
dqs=rows('''SELECT a.startNs,a.endNs,a.globalTid,a.connectionId,s.value name FROM PYTORCH_API a JOIN STRING_IDS s ON s.id=a.name WHERE a.type=50002 AND s.value='Dequeue@aclnnInplaceCopy' ''')
def ms(r):return (int(r['endNs'])-int(r['startNs']))/1e6
def nested(a,b):return a['globalTid']==b['globalTid'] and int(a['startNs'])<=int(b['startNs']) and int(b['endNs'])<=int(a['endNs'])
joined=[]
for s in ss:
 parent=[a for a in cs if nested(a,s)]
 if len(parent)!=1:continue
 a=parent[0];deq=[d for d in dqs if nested(d,a)]
 joined.append({'slice':s,'slice_ms':ms(s),'copy':a,'copy_ms':ms(a),'dequeue_matches':deq})
joined.sort(key=lambda r:-r['slice_ms'])
for z in joined[:12]:
 z['correlations']=[]
 for d in z['dequeue_matches']:
  ids=rows('SELECT id,connectionId FROM CONNECTION_IDS WHERE id=?',(d['connectionId'],),100)
  for k in ids:
   q=rows('''SELECT a.startNs,a.endNs,a.globalTid,a.connectionId,s.value name FROM CONNECTION_IDS k JOIN PYTORCH_API a ON a.connectionId=k.id JOIN STRING_IDS s ON s.id=a.name WHERE k.connectionId=? AND a.type=50002''',(k['connectionId'],),200)
   enqs=[e for e in q if e['name']=='Enqueue@aclnnInplaceCopy'];out={'id':k['connectionId'],'all_queue_rows':len(q),'enqueues':enqs,'scopes':[]}
   for e in enqs:
    scopes=rows('''SELECT a.startNs,a.endNs,s.value name,dt.value inputDtypes,sh.value inputShapes,a.callchainId FROM PYTORCH_API a JOIN STRING_IDS s ON s.id=a.name LEFT JOIN STRING_IDS dt ON dt.id=a.inputDtypes LEFT JOIN STRING_IDS sh ON sh.id=a.inputShapes WHERE a.type=50001 AND a.globalTid=? AND CAST(a.startNs AS INTEGER)<=? AND CAST(a.endNs AS INTEGER)>=? ORDER BY CAST(a.endNs AS INTEGER)-CAST(a.startNs AS INTEGER)''',(e['globalTid'],int(e['startNs']),int(e['endNs'])),100)
    out['scopes'].append(scopes)
   z['correlations'].append(out)
vals=sorted(x['slice_ms'] for x in joined)
st1=db.stat();assert (st0.st_size,st0.st_mtime_ns)==(st1.st_size,st1.st_mtime_ns)
print(json.dumps({'profile_info_sha256':hashlib.sha256(raw).hexdigest(),'db_sha256':h.hexdigest(),'db_bytes':st1.st_size,'settings':settings,'counts':{'cann_inplace_copy':len(cs),'slice_tiling':len(ss),'dequeues':len(dqs),'slices_with_one_parent_copy':len(joined),'joined_with_one_dequeue':sum(len(z['dequeue_matches'])==1 for z in joined)},'all_slice_ms_distribution':{'n':len(ss),'min':min(map(ms,ss)) if ss else None,'max':max(map(ms,ss)) if ss else None,'ge20ms':sum(ms(x)>=20 for x in ss)},'slice_ms_distribution':{'n':len(vals),'min':min(vals) if vals else None,'median':statistics.median(vals) if vals else None,'max':max(vals) if vals else None,'ge20ms':sum(v>=20 for v in vals)},'longest_matches':joined[:12],'limitations':['all sampled EP8 rows; not a matched workload/shape/tensor group','nested intervals and correlation are within this exported DB','no Python stack or shape attribution assumed','unknown profiler overhead and completeness']},indent=2))
