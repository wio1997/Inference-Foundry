from pathlib import Path
from collections import defaultdict,Counter
from datetime import datetime,timezone
from decimal import Decimal
import json,hashlib,sqlite3,csv,socket,time
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
A=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/profile-readonly-20261031')
def pinned(p,sha):
 x=p.stat();b=p.read_bytes();y=p.stat();assert hashlib.sha256(b).hexdigest()==sha;assert (x.st_size,x.st_mtime_ns)==(y.st_size,y.st_mtime_ns);return b
d=json.loads(pinned(A/'ep3-all-compute-origins-v9-v2.json','fdce699c6cf2fdf15b9ba85c5e9e5a5cd7893d2b34a0f69b0321e719a9ef6b9d'))
audit=json.loads(pinned(A/'ep3-origin-source-and-csv-audit-v9-v2.json','7124c7e900ed2bfb4d1ed9a29eb335096a5d8f9a6bd2be85b1f43b9d91453541'))
csv_rows=list(csv.DictReader(pinned(Path(audit['summary']['csv_path']),audit['summary']['csv_sha256']).decode('utf-8-sig').splitlines()))
types={'_quantize_indexer_query_kernel','_prepare_indexer_indices_kernel','_pool_kernel','_cache_update_kernel'}
used={r['task_record_rowid'] for r in audit['csv_compute_matches']}
left_db=[t for t in d['tasks'] if t['task_record_rowid'] not in used]
left_csv=[(i,csv_rows[i]) for i in audit['unmatched_csv_row_indices'] if csv_rows[i]['Type'] in types]
assert len(left_db)==len(left_csv)==242 and {t['op_type'] for t in left_db}==types
index=defaultdict(list);time_index=defaultdict(list)
for t in left_db:
 index[(t['op_type'],t['streamId'],t['taskId'],t['modelId'])].append(t)
 time_index[(t['op_type'],t['startNs'])].append(t)
items=[]
for i,r in left_csv:
 s=int(Decimal(r['Start Time(us)'].strip())*1000);e=s+int(Decimal(r['Duration(us)'].strip())*1000)
 base=(r['Type'],int(r['Stream ID']),int(r['Task ID']),int(r['Model ID']))
 candidates=index.get(base,[])
 items.append(dict(csv_row=i,csv={k:r[k] for k in ['Name','Type','Model ID','Stream ID','Task ID','Start Time(us)','Duration(us)','Input Shapes','Output Shapes']},
   csv_start_ns=s,csv_end_ns=e,same_type_stream_task_model_candidates=[dict(task_record_rowid=t['task_record_rowid'],globalTaskId=t['globalTaskId'],connectionId=t['connectionId'],
     startNs=t['startNs'],endNs=t['endNs'],start_delta_ns=s-t['startNs'],end_delta_ns=e-t['endNs'],kernel_name=t['kernel_name']) for t in candidates],
   same_type_exact_start_candidates=[{k:t[k] for k in ['task_record_rowid','globalTaskId','modelId','streamId','taskId','startNs','endNs']} for t in time_index.get((r['Type'],s),[])]))
assert max(len(x['same_type_stream_task_model_candidates']) for x in items)<=32
DB=Path(d['summary']['database']);before=DB.stat();h=hashlib.sha256()
with DB.open('rb') as f:
 for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
assert h.hexdigest()==d['summary']['database_sha256']
t0=time.monotonic();db=sqlite3.connect(DB.as_uri()+'?mode=ro&immutable=1',uri=True);db.row_factory=sqlite3.Row
db.execute('PRAGMA query_only=ON');db.set_progress_handler(lambda:1 if time.monotonic()-t0>100 else 0,100000)
tables=[r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")]
graph_tables=[t for t in tables if any(s in t.upper() for s in ['MODEL','GRAPH'])]
schemas={t:[dict(r) for r in db.execute('PRAGMA table_info("'+t+'")')] for t in graph_tables+['CANN_API']}
graph_table_counts={t:db.execute('SELECT COUNT(*) FROM "'+t+'"').fetchone()[0] for t in graph_tables}
graph_api_names=[dict(r) for r in db.execute('''SELECT s.value AS name,COUNT(*) AS count FROM CANN_API a JOIN STRING_IDS s ON s.id=a.name
 WHERE s.value LIKE '%Capture%' OR s.value LIKE '%Graph%' OR s.value LIKE '%aclmdl%' OR s.value LIKE '%ModelExecute%' GROUP BY a.name''')]
apis=[dict(r) for r in db.execute('''SELECT a.rowid AS cann_api_rowid,a.startNs,a.endNs,a.connectionId,a.globalTid,s.value AS name FROM CANN_API a JOIN STRING_IDS s ON s.id=a.name
 WHERE s.value='aclmdlRIExecuteAsync' ORDER BY a.startNs''')]
assert len(apis)<=256
models=[dict(r) for r in db.execute('''SELECT t.rowid AS task_record_rowid,t.*,s.value AS task_type_name FROM TASK t JOIN STRING_IDS s ON s.id=t.taskType
 WHERE t.deviceId=3 AND s.value IN ('MODEL_EXECUTE','MODEL_WAIT_COMPLETE') ORDER BY t.startNs''')]
links=[dict(r) for r in db.execute('SELECT id,connectionId FROM CONNECTION_IDS') if r['id'] in {a['connectionId'] for a in apis} or r['connectionId'] in {a['connectionId'] for a in apis}]
host_ids={a['connectionId'] for a in apis}|{r['id'] for r in links}|{r['connectionId'] for r in links}
pt=[]
if host_ids:
 pt=[dict(r) for r in db.execute('''SELECT p.rowid AS api_rowid,p.startNs,p.endNs,p.globalTid,p.connectionId,p.callchainId,s.value AS name FROM PYTORCH_API p
 LEFT JOIN STRING_IDS s ON s.id=p.name WHERE p.connectionId IN ('''+','.join('?' for _ in host_ids)+')',sorted(host_ids))]
assert len(pt)<=2000
ids=sorted({a['callchainId'] for a in pt if a['callchainId'] is not None});frames=defaultdict(list)
if ids:
 for r in db.execute('''SELECT c.id,c.stackDepth,s.value AS frame FROM PYTORCH_CALLCHAINS c JOIN STRING_IDS s ON s.id=c.stack WHERE c.id IN ('''+','.join('?' for _ in ids)+') ORDER BY c.id,c.stackDepth',ids):frames[r['id']].append(dict(stackDepth=r['stackDepth'],frame=r['frame']))
api_links=[]
for a in apis:
 device_ids={a['connectionId']}|{r['connectionId'] for r in links if r['id']==a['connectionId']}
 candidates=[m for m in models if m['connectionId'] in device_ids]
 api_links.append(dict(api=a,device_connection_candidates=sorted(device_ids),model_task_candidates=candidates,
  exact_connection_pytorch_apis=[p for p in pt if p['connectionId']==a['connectionId']]))
db.close();after=DB.stat();assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
examples=[]
for typ in sorted(types):examples.append(next(x for x in items if x['csv']['Type']==typ))
summary=dict(created_utc=datetime.now(timezone.utc).isoformat(),database_sha256=d['summary']['database_sha256'],database_stat_unchanged=True,
 csv_unmatched_vectors=len(items),DB_unrepresented_compute_records=len(left_db),same_non_time_key_candidate_count_hist=dict(Counter(len(x['same_type_stream_task_model_candidates']) for x in items)),
 exact_start_candidate_count_hist=dict(Counter(len(x['same_type_exact_start_candidates']) for x in items)),
 graph_tables=graph_tables,graph_table_counts=graph_table_counts,graph_api_names=graph_api_names,graph_execute_APIs=len(apis),model_task_records=len(models),
 graph_API_model_connection_candidate_count_hist=dict(Counter(len(x['model_task_candidates']) for x in api_links)),graph_linked_pytorch_API_records=len(pt),
 no_target_source_read=True,no_device_or_inference_work=True,overall_complete=False,
 limits=['Candidate differences are diagnostic observations, not accepted matches. No nearest-time repair or guessed timestamp offset is used.',
         'Graph API/task joins use stored connection fields; membership of model tasks is distinct from replay/client phase and clock-calibrated latency.',
         'Same type/count is not proof of tensor identity or record equivalence.'])
out=A/'ep3-vector-identities-and-graph-links-v10.json';assert not out.exists()
full=dict(summary=summary,vector_identity_rows=items,unrepresented_DB_tasks=left_db,graph_schemas=schemas,graph_API_links=api_links,
          graph_model_tasks=models,graph_pytorch_APIs=pt,graph_callchains={str(k):v for k,v in frames.items()})
payload=json.dumps(full,ensure_ascii=False,indent=2)+'\n';out.write_text(payload)
public=dict(summary=summary,full_result=str(out),result_sha256=hashlib.sha256(payload.encode()).hexdigest(),vector_examples=examples,
 graph_schema=schemas,first_graph_API_link=api_links[:1],first_graph_pytorch_apis=pt[:4])
assert len(json.dumps(public,ensure_ascii=False).encode())<=32000
print(json.dumps(public,ensure_ascii=False))
