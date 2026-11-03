from pathlib import Path
from collections import defaultdict,Counter
from datetime import datetime,timezone
from decimal import Decimal
import json,hashlib,csv,socket,sqlite3,time
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
A=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/profile-readonly-20261031')
def pinned(p,sha):
 x=p.stat();b=p.read_bytes();y=p.stat();assert hashlib.sha256(b).hexdigest()==sha;assert (x.st_size,x.st_mtime_ns)==(y.st_size,y.st_mtime_ns);return b
d=json.loads(pinned(A/'ep3-all-compute-origins-v9-v2.json','fdce699c6cf2fdf15b9ba85c5e9e5a5cd7893d2b34a0f69b0321e719a9ef6b9d'))
audit=json.loads(pinned(A/'ep3-origin-source-and-csv-audit-v9-v2.json','7124c7e900ed2bfb4d1ed9a29eb335096a5d8f9a6bd2be85b1f43b9d91453541'))
q=json.loads(pinned(A/'ep3-vector-identities-and-graph-links-v10.json','8b7345aedb84ddac61dc2564e088bd584cabdeab24115075b358c14e95330bb7'))
csv_rows=list(csv.DictReader(pinned(Path(audit['summary']['csv_path']),audit['summary']['csv_sha256']).decode('utf-8-sig').splitlines()))
task_map={t['task_record_rowid']:t for t in d['tasks']};new_matches=[]
for x in q['vector_identity_rows']:
 assert len(x['same_type_exact_start_candidates'])==1
 t=task_map[x['same_type_exact_start_candidates'][0]['task_record_rowid']];r=csv_rows[x['csv_row']]
 assert (r['Name'],r['Type'],int(r['Stream ID']),int(r['Task ID']),x['csv_start_ns'],x['csv_end_ns'])==(t['kernel_name'],t['op_type'],t['streamId'],t['taskId'],t['startNs'],t['endNs'])
 assert int(r['Model ID'])==4294967295 and t['modelId']==49
 new_matches.append(dict(csv_row=x['csv_row'],task_record_rowid=t['task_record_rowid'],globalTaskId=t['globalTaskId'],type=t['op_type'],
  exact_fields=['name','type','stream','task','start','end'],CSV_model_id=r['Model ID'],TASK_model_id=t['modelId'],
  evidence_status='EXACT_SIX_NONMODEL_FIELDS_WITH_EXPLICIT_MODEL_METADATA_CONFLICT'))
assert len(new_matches)==len({r['task_record_rowid'] for r in new_matches})==len({r['csv_row'] for r in new_matches})==242
matches=audit['csv_compute_matches']+new_matches
assert len(matches)==len({m['task_record_rowid'] for m in matches})==len(d['tasks'])==36521
csv_by_task={m['task_record_rowid']:csv_rows[m['csv_row']] for m in matches}
def union(intervals):
 end=None;out=0
 for a,b in sorted(intervals):
  if b<=a:continue
  if end is None or a>end:out+=b-a;end=b
  elif b>end:out+=b-end;end=b
 return out
exec_rows=[]
for i,link in enumerate(q['graph_API_links']):
 wait=[x for x in link['model_task_candidates'] if x['task_type_name']=='MODEL_WAIT_COMPLETE']
 execute=[x for x in link['model_task_candidates'] if x['task_type_name']=='MODEL_EXECUTE']
 assert len(wait)==len(execute)==1
 w=wait[0];m=execute[0];assert w['connectionId']==m['connectionId']==link['api']['connectionId'] and w['modelId']==m['modelId']
 wait_only=[t for t in d['tasks'] if t['modelId']==w['modelId'] and w['startNs']<=t['startNs'] and t['endNs']<=w['endNs']]
 lo_control=min(m['startNs'],w['startNs']);hi_control=max(m['endNs'],w['endNs'])
 tasks=[t for t in d['tasks'] if t['modelId']==w['modelId'] and lo_control<=t['startNs'] and t['endNs']<=hi_control]
 markers=sorted([t for t in tasks if t['op_type'] in ['HcPre','HcPost']],key=lambda t:(t['startNs'],t['streamId'],t['taskId']))
 expected=['HcPre','HcPost','HcPre','HcPost']*40
 marker_check=[t['op_type'] for t in markers]==expected
 layers=[]
 if marker_check:
  for k in range(40):
   ms=markers[k*4:(k+1)*4];lo,hi=ms[0]['startNs'],ms[-1]['endNs']
   ts=[t for t in tasks if lo<=t['startNs'] and t['endNs']<=hi]
   counts=dict(Counter(t['op_type'] for t in ts));selected=counts.get('DispatchFFNCombine')==counts.get('SparseFlashMla')==1 and counts.get('HcPre')==counts.get('HcPost')==2
   layers.append(dict(observed_layer_block_ordinal=k,startNs=lo,endNs=hi,compute_records=len(ts),type_counts=counts,
     marker_and_one_attention_one_moe_check=selected,task_record_rowids=[t['task_record_rowid'] for t in ts],
     main_marker_streams=sorted({t['streamId'] for t in ms}),temporal_block_not_all_consumer_proof=True))
 assigned={r for l in layers for r in l['task_record_rowids']}
 perstream=defaultdict(list)
 for t in sorted(tasks,key=lambda t:(t['startNs'],t['streamId'],t['taskId'])):perstream[t['streamId']].append([t['globalTaskId'],t['taskId'],t['op_type']])
 stream_signatures={str(s):dict(records=len(rows),sha256=hashlib.sha256(json.dumps(rows,separators=(',',':')).encode()).hexdigest()) for s,rows in perstream.items()}
 exec_rows.append(dict(ordinal=i,CANN_API=link['api'],MODEL_EXECUTE=m,MODEL_WAIT_COMPLETE=w,model_id=w['modelId'],compute_records=len(tasks),
   wait_only_compute_records=len(wait_only),control_pair_start_ns=lo_control,control_pair_end_ns=hi_control,unique_definitions=len({t['globalTaskId'] for t in tasks}),definition_set_sha256=hashlib.sha256(json.dumps(sorted(t['globalTaskId'] for t in tasks)).encode()).hexdigest(),
   type_counts=dict(Counter(t['op_type'] for t in tasks)),stream_signatures=stream_signatures,
   marker_count=len(markers),marker_sequence_check=marker_check,layer_blocks=layers,
   block_assigned_records=len(assigned),outside_block_records=len(tasks)-len(assigned),
   outside_block_type_counts=dict(Counter(t['op_type'] for t in tasks if t['task_record_rowid'] not in assigned)),
   task_record_rowids=[t['task_record_rowid'] for t in tasks],
   graph_compute_record_union_ns=union([(t['startNs'],t['endNs']) for t in tasks]),
   graph_record_span_ns=max((t['endNs'] for t in tasks),default=w['startNs'])-min((t['startNs'] for t in tasks),default=w['startNs'])))
all_ids=[x for e in exec_rows for x in e['task_record_rowids']]
graph_tasks=[t for t in d['tasks'] if t['modelId']==49]
missed_ids={t['task_record_rowid'] for t in graph_tasks}-set(all_ids)
duplicate_ids=[rid for rid,n in Counter(all_ids).items() if n>1]
container_check=(not missed_ids and not duplicate_ids and all(e['compute_records']==1650 for e in exec_rows))
# Native graph submission has no direct PYTORCH connection in this export.
# Same-thread host-scope containment is saved separately and not promoted to
# a connection proof or formal request identity.
DB=Path(d['summary']['database']);before=DB.stat();h=hashlib.sha256()
with DB.open('rb') as f:
 for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
assert h.hexdigest()==d['summary']['database_sha256']
t0=time.monotonic();db=sqlite3.connect(DB.as_uri()+'?mode=ro&immutable=1',uri=True);db.row_factory=sqlite3.Row
db.execute('PRAGMA query_only=ON');db.set_progress_handler(lambda:1 if time.monotonic()-t0>100 else 0,100000)
host_scopes=[];chain_ids=set()
for e in exec_rows:
 a=e['CANN_API']
 rs=[dict(r) for r in db.execute('''SELECT p.rowid AS api_rowid,p.startNs,p.endNs,p.globalTid,p.connectionId,p.callchainId,p.type,s.value AS name
 FROM PYTORCH_API p LEFT JOIN STRING_IDS s ON s.id=p.name WHERE p.globalTid=? AND p.startNs<=? AND p.endNs>=? ORDER BY p.endNs-p.startNs LIMIT 32''',
  (a['globalTid'],a['startNs'],a['endNs']))]
 host_scopes.append(dict(ordinal=e['ordinal'],candidate_scope_count=len(rs),same_thread_enclosing_pytorch_scopes=rs,connection_proof=False))
 chain_ids.update(r['callchainId'] for r in rs if r['callchainId'] is not None)
frames=defaultdict(list)
if chain_ids:
 for r in db.execute('''SELECT c.id,c.stackDepth,s.value AS frame FROM PYTORCH_CALLCHAINS c JOIN STRING_IDS s ON s.id=c.stack WHERE c.id IN ('''+','.join('?' for _ in chain_ids)+') ORDER BY c.id,c.stackDepth',sorted(chain_ids)):frames[r['id']].append(dict(stackDepth=r['stackDepth'],frame=r['frame']))
db.close();after=DB.stat();assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
META=DB.parents[2]/'metadata/events-2128.jsonl'
mbefore=META.stat();assert mbefore.st_size<=30000000;mraw=META.read_bytes();mafter=META.stat();assert (mbefore.st_size,mbefore.st_mtime_ns)==(mafter.st_size,mafter.st_mtime_ns)
metadata=[json.loads(x) for x in mraw.decode().splitlines()]
selected_meta=[x for x in metadata if x.get('label')=='NPUModelRunner.execute_model' or x.get('kind') in ['profiler_start_anchor','profiler_stop_anchor']]
summary=dict(created_utc=datetime.now(timezone.utc).isoformat(),database_sha256=d['summary']['database_sha256'],database_stat_unchanged=True,
  six_field_vector_matches=len(new_matches),model_field_conflict_hist={'CSV4294967295_TASK49':len(new_matches)},complete_COMPUTE_CSV_overlay_records=len(matches),
  original_full_four_field_matches=len(audit['csv_compute_matches']),model_conflicts_preserved=True,
  graph_executions=len(exec_rows),graph_compute_records=len(all_ids),control_pair_container_check=container_check,missed_graph_records=len(missed_ids),overlapping_graph_record_assignments=len(duplicate_ids),wait_only_counts=[e['wait_only_compute_records'] for e in exec_rows],metadata_path=str(META),metadata_sha256=hashlib.sha256(mraw).hexdigest(),metadata_selected_count=len(selected_meta),metadata_all_event_count=len(metadata),definition_set_counts=dict(Counter(e['definition_set_sha256'] for e in exec_rows)),
  graph_compute_counts=[e['compute_records'] for e in exec_rows],marker_sequence_pass=sum(e['marker_sequence_check'] for e in exec_rows),
  layer_block_counts=[len(e['layer_blocks']) for e in exec_rows],per_block_marker_attention_moe_pass=sum(l['marker_and_one_attention_one_moe_check'] for e in exec_rows for l in e['layer_blocks']),
  assigned_compute_counts=[e['block_assigned_records'] for e in exec_rows],outside_block_counts=[e['outside_block_records'] for e in exec_rows],
  same_thread_host_scope_count_hist=dict(Counter(e['candidate_scope_count'] for e in host_scopes)),
  no_target_source_read=True,no_device_or_inference_work=True,whole_modules_closed=0,overall_complete=False,
  limits=['The242six-field associations explicitly preserve a Model ID disagreement; no blanket sentinel normalization or nearest-time matching.',
   'All18150model49compute records have same-model paired MODEL_EXECUTE-start to MODEL_WAIT_COMPLETE-end containment when control_pair_container_check passes, each associated to a native execute by stored connection IDs. This is export association, not a causal wait-time estimate.',
   'Four-marker blocks and per-stream definition signatures are observed structural evidence; source-layer object identity and all consumers still require scoped contract review.',
   'A task temporally inside a block can have other consumers or ownership. Outside blocks is not idle work.',
   'Same-thread host containment is not a direct connection, formal-client identity, calibrated host/device delay or runtime flag attestation.'])
out=A/'ep3-vector-overlay-and-graph-blocks-v10-v2.json';assert not out.exists()
full=dict(summary=summary,vector_overlay=new_matches,graph_executions=exec_rows,host_scope_candidates=host_scopes,host_scope_callchains={str(k):v for k,v in frames.items()},metadata_selected_events=selected_meta,uncontained_graph_task_examples=[t for t in graph_tasks if t['task_record_rowid'] in missed_ids][:16],duplicate_container_record_ids=duplicate_ids)
payload=json.dumps(full,ensure_ascii=False,indent=2)+'\n';out.write_text(payload)
public=dict(summary=summary,full_result=str(out),result_sha256=hashlib.sha256(payload.encode()).hexdigest(),
 first_execution={k:v for k,v in exec_rows[0].items() if k not in ['task_record_rowids','layer_blocks']},
 first_layer_blocks=[{k:v for k,v in l.items() if k!='task_record_rowids'} for l in exec_rows[0]['layer_blocks'][:3]],
 metadata_first_selected_events=selected_meta[:2],metadata_first_row_keys=list(metadata[0]) if metadata else [],first_host_scopes=host_scopes[:2],first_host_scope_callchains={str(k):v for k,v in list(frames.items())[:3]})
assert len(json.dumps(public,ensure_ascii=False).encode())<=32000
print(json.dumps(public,ensure_ascii=False))
