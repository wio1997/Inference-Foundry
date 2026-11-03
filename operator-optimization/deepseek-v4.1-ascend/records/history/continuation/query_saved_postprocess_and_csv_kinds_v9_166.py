from pathlib import Path
from collections import Counter,defaultdict
from decimal import Decimal
from datetime import datetime,timezone
import json,hashlib,socket,sqlite3,csv
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
A=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/profile-readonly-20261031')
def pinned(p,sha):
 before=p.stat();raw=p.read_bytes();after=p.stat();assert hashlib.sha256(raw).hexdigest()==sha;assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns);return raw
d=json.loads(pinned(A/'ep3-all-compute-origins-v9-v2.json','fdce699c6cf2fdf15b9ba85c5e9e5a5cd7893d2b34a0f69b0321e719a9ef6b9d'))
audit=json.loads(pinned(A/'ep3-origin-source-and-csv-audit-v9-v2.json','7124c7e900ed2bfb4d1ed9a29eb335096a5d8f9a6bd2be85b1f43b9d91453541'))
api={a['api_rowid']:a for a in d['apis']};fd=d['frame_dictionary'];chains=d['callchains']
def frames(a):return [fd[str(f['frame_id'])] for f in chains[str(a['callchainId'])]]
groups=defaultdict(list)
for t in d['tasks']:
 for i in t['api_rowids']:
  a=api[i];fs=frames(a)
  selected=[f for f in fs if any(s in f for s in ['llm_base_proposer.py(','/sample/sampler.py(','/sample/rejection_sampler.py(','reject_sample.py('])]
  if not selected:continue
  key=json.dumps([t['op_type'],t['input_shapes'],t['output_shapes'],t['input_dtypes'],t['output_dtypes'],selected],ensure_ascii=False)
  groups[key].append(t)
rows=[]
for key,ts in groups.items():
 typ,ish,osh,idt,odt,fs=json.loads(key)
 rows.append(dict(type=typ,input_shapes=ish,output_shapes=osh,input_dtypes=idt,output_dtypes=odt,source_frames=fs,records=len(ts),
   context_hist=dict(Counter('+'.join(t['recorded_host_context_flags']) or 'NONE' for t in ts)),
   task_records=[dict(rowid=t['task_record_rowid'],startNs=t['startNs'],endNs=t['endNs'],streamId=t['streamId'],taskId=t['taskId']) for t in ts]))
rows.sort(key=lambda r:r['task_records'][0]['startNs'])
selected_line_rows=[r for r in rows if any('llm_base_proposer.py('+str(n)+')' in f for n in [1584,1589,1599,1601,1602] for f in r['source_frames'])]
csv_rows=list(csv.DictReader(pinned(Path(audit['summary']['csv_path']),audit['summary']['csv_sha256']).decode('utf-8-sig').splitlines()))
unmatched=[(i,csv_rows[i]) for i in audit['unmatched_csv_row_indices']]
DB=Path(d['summary']['database']);before=DB.stat();h=hashlib.sha256()
with DB.open('rb') as f:
 for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
assert h.hexdigest()==d['summary']['database_sha256']
db=sqlite3.connect(DB.as_uri()+'?mode=ro&immutable=1',uri=True);db.row_factory=sqlite3.Row;db.execute('PRAGMA query_only=ON')
comm=[dict(r) for r in db.execute('''SELECT c.*,n.value AS op_name FROM COMMUNICATION_OP c LEFT JOIN STRING_IDS n ON n.id=c.opName WHERE c.deviceId=3''')]
by_comm=defaultdict(list)
for c in comm:by_comm[(c['startNs'],c['endNs'],c['op_name'])].append(c)
task_rows=[dict(r) for r in db.execute('''SELECT t.rowid AS task_record_rowid,t.*,s.value AS task_type_name FROM TASK t LEFT JOIN STRING_IDS s ON s.id=t.taskType WHERE t.deviceId=3''')]
by_task=defaultdict(list)
for t in task_rows:by_task[(t['streamId'],t['taskId'],t['modelId'],t['startNs'])].append(t)
kind_rows=[]
for index,r in unmatched:
 start=int(Decimal(r['Start Time(us)'].strip())*1000);end=start+int(Decimal(r['Duration(us)'].strip())*1000)
 cs=by_comm.get((start,end,r['Name']),[])
 ts=[]
 try:ts=by_task.get((int(r['Stream ID']),int(r['Task ID']),int(r['Model ID']),start),[])
 except ValueError:pass
 kind_rows.append(dict(csv_row=index,csv_type=r['Type'],csv_name=r['Name'],csv_core=r.get('Accelerator Core'),
  exact_COMM_matches=[dict(opId=c['opId'],connectionId=c['connectionId'],startNs=c['startNs'],endNs=c['endNs']) for c in cs],
  exact_TASK_matches=[dict(rowid=t['task_record_rowid'],task_type=t['task_type_name'],globalTaskId=t['globalTaskId']) for t in ts]))
db.close();after=DB.stat();assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
kind_summary=[]
for typ in sorted({r['csv_type'] for r in kind_rows}):
 rs=[r for r in kind_rows if r['csv_type']==typ]
 kind_summary.append(dict(type=typ,rows=len(rs),core_hist=dict(Counter(str(r['csv_core']) for r in rs)),exact_one_COMM=sum(len(r['exact_COMM_matches'])==1 for r in rs),
  exact_one_TASK=sum(len(r['exact_TASK_matches'])==1 for r in rs),task_type_hist=dict(Counter(t['task_type'] for r in rs for t in r['exact_TASK_matches']))))
summary=dict(created_utc=datetime.now(timezone.utc).isoformat(),database_sha256=d['summary']['database_sha256'],database_stat_unchanged=True,
  postprocess_signature_groups=len(rows),postprocess_compute_records=sum(r['records'] for r in rows),selected_add_argmax_copy_groups=len(selected_line_rows),
  selected_add_argmax_copy_counts=dict(Counter({typ:sum(r['records'] for r in selected_line_rows if r['type']==typ) for typ in {r['type'] for r in selected_line_rows}})),
  unmatched_csv_kind_audit=kind_summary,COMM_rows=len(comm),TASK_rows=len(task_rows),
  no_target_source_read=True,no_device_or_inference_work=True,overall_complete=False,
  limits=['All postprocess native tasks with the selected recorded source ancestry in one database; ancestry is not formal-client identity.',
          'Exact COMM association uses original name/start/end. A COMM interval and nested physical task durations must not be added as disjoint work.',
          'Unmatched rows remain unmatched; no temporal-nearest matching or source/phase guessing.'])
out=A/'ep3-postprocess-and-csv-kind-audit-v9.json';assert not out.exists()
payload=json.dumps(dict(summary=summary,postprocess_rows=rows,csv_unmatched_kind_rows=kind_rows),ensure_ascii=False,indent=2)+'\n';out.write_text(payload)
public=dict(summary=summary,full_result=str(out),result_sha256=hashlib.sha256(payload.encode()).hexdigest(),
 selected_postprocess_rows=[{k:v for k,v in r.items() if k!='task_records'} for r in selected_line_rows])
assert len(json.dumps(public,ensure_ascii=False).encode())<=32000
print(json.dumps(public,ensure_ascii=False))
