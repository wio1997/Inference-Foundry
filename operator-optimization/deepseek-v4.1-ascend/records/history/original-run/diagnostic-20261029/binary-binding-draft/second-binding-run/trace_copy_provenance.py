"""Read-only queue correlation and exact same-thread enqueue containment audit."""
from pathlib import Path
import json,sqlite3,hashlib
root=Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029/binary-binding-draft/second-binding-run/offline-ep8-20261029')
status=json.loads((root/'status.json').read_text()); item=next(x for x in status['dbs'] if x['path'].endswith('/ascend_pytorch_profiler_8.db'));db=root/item['path']
assert not Path(str(db)+'-wal').exists() and not Path(str(db)+'-journal').exists()
c=sqlite3.connect(db.resolve().as_uri()+'?mode=ro&immutable=1',uri=True);c.row_factory=sqlite3.Row;c.execute('PRAGMA query_only=ON')
def rows(sql,args=(),limit=500):
 r=[dict(x) for x in c.execute(sql,args).fetchmany(limit+1)]
 if len(r)>limit:raise RuntimeError('bounded rows overflow')
 return r
samples=[]
for x in SAMPLES['results']:
 for y in x['longest_clipped_dequeues']:
  if y['name']=='Dequeue@aclnnInplaceCopy' and y['duration_ms']>=20:
   samples.append({'layer_call':x['call_id'],'start':y['startNs'],'end':y['endNs'],'worker_tid':x['worker_tid_global'],'clipped_ms':y['duration_ms']})
seen=set();out=[]
for v in samples:
 key=(v['start'],v['end'],v['worker_tid'])
 if key in seen:continue
 seen.add(key)
 ds=rows('''SELECT a.startNs,a.endNs,a.globalTid,a.connectionId,s.value name FROM PYTORCH_API a JOIN STRING_IDS s ON s.id=a.name WHERE a.type=50002 AND a.globalTid=? AND CAST(a.startNs AS INTEGER)=? AND CAST(a.endNs AS INTEGER)=? AND s.value='Dequeue@aclnnInplaceCopy' ''',(v['worker_tid'],v['start'],v['end']))
 rec={'observed':v,'exact_dequeues':ds,'correlations':[]}
 if len(ds)!=1 or ds[0]['connectionId'] is None:rec['status']='UNKNOWN_DEQUEUE_IDENTITY';out.append(rec);continue
 deq=ds[0]
 nested=rows('''SELECT a.startNs,a.endNs,a.globalTid,a.type,a.connectionId,s.value name FROM CANN_API a JOIN STRING_IDS s ON s.id=a.name WHERE a.globalTid=? AND CAST(a.startNs AS INTEGER)>=? AND CAST(a.endNs AS INTEGER)<=? AND s.value IN ('aclnnInplaceCopy','Slice_Tiling') ORDER BY CAST(a.startNs AS INTEGER)''',(v['worker_tid'],v['start'],v['end']))
 rec['nested_cann']=nested
 corr=rows('SELECT id,connectionId FROM CONNECTION_IDS WHERE id=?',(deq['connectionId'],))
 for x in corr:
  q=rows('''SELECT a.startNs,a.endNs,a.globalTid,a.connectionId,s.value name FROM CONNECTION_IDS k JOIN PYTORCH_API a ON a.connectionId=k.id JOIN STRING_IDS s ON s.id=a.name WHERE k.connectionId=? AND a.type=50002 ORDER BY CAST(a.startNs AS INTEGER)''',(x['connectionId'],))
  matching=[e for e in q if e['name']=='Enqueue@aclnnInplaceCopy'];dequeues=[e for e in q if e['name']=='Dequeue@aclnnInplaceCopy']
  entry={'correlation':x['connectionId'],'id_rows':len(q),'enqueues':matching,'dequeues':dequeues,'other_names':sorted(set(e['name'] for e in q if e['name'] not in ('Enqueue@aclnnInplaceCopy','Dequeue@aclnnInplaceCopy'))),'scopes':[]}
  for e in matching:
   if int(e['endNs'])>v['start']:
    entry.setdefault('ordering_issues',[]).append('enqueue_after_dequeue')
   scopes=rows('''SELECT a.startNs,a.endNs,a.globalTid,a.type,s.value name,sh.value inputShapes,dt.value inputDtypes,a.connectionId,a.callchainId FROM PYTORCH_API a JOIN STRING_IDS s ON s.id=a.name LEFT JOIN STRING_IDS sh ON sh.id=a.inputShapes LEFT JOIN STRING_IDS dt ON dt.id=a.inputDtypes WHERE a.type=50001 AND a.globalTid=? AND CAST(a.startNs AS INTEGER)<=? AND CAST(a.endNs AS INTEGER)>=? ORDER BY (CAST(a.endNs AS INTEGER)-CAST(a.startNs AS INTEGER)) ASC''',(e['globalTid'],int(e['startNs']),int(e['endNs'])),limit=200)
   for z in scopes:
    if z['name'] in ('aten::to','aten::contiguous','aten::copy_') and z['callchainId'] is not None:
     z['callchain']=rows('SELECT k.stackDepth,s.value FROM PYTORCH_CALLCHAINS k LEFT JOIN STRING_IDS s ON s.id=k.stack WHERE k.id=? ORDER BY k.stackDepth',(z['callchainId'],),limit=128)
   entry['scopes'].append({'enqueue':e,'scopes':scopes})
  rec['correlations'].append(entry)
 rec['status']='CONDITIONAL_UNIQUE_CORRELATION' if len(corr)==1 and len(rec['correlations'][0]['enqueues'])==1 and len(rec['correlations'][0]['dequeues'])==1 and not rec['correlations'][0]['other_names'] and not rec['correlations'][0].get('ordering_issues') else 'AMBIGUOUS_OR_MISSING'
 out.append(rec)
print(json.dumps({'db_sha256_expected':item['sha256'],'sample_count':len(out),'samples':out,'limits':['Unique in exported DB is not proof correlation IDs never reused outside exported records.','Containment of enqueue within Python scope is a host interval observation, not tensor dependency or cause.','Profiler SQLite disk I/O error and missing events prevent completeness.','Profiler enabled stack and shapes and may perturb host latency.']},indent=2))
