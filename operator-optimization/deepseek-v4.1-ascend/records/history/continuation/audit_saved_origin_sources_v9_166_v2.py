from pathlib import Path
from collections import Counter,defaultdict
from datetime import datetime,timezone
from decimal import Decimal
import hashlib,json,csv,socket,re
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
A=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/profile-readonly-20261031')
SRC=A/'ep3-all-compute-origins-v9-v2.json';before=SRC.stat();raw=SRC.read_bytes()
assert hashlib.sha256(raw).hexdigest()=='fdce699c6cf2fdf15b9ba85c5e9e5a5cd7893d2b34a0f69b0321e719a9ef6b9d'
d=json.loads(raw);after=SRC.stat();assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
tasks=d['tasks'];api={a['api_rowid']:a for a in d['apis']};chains=d['callchains'];fd=d['frame_dictionary']
parse=re.compile(r'^(.*?)\((\d+)\): (.*)$')
def short_file(p):
 return p.replace('/vllm-workspace/vllm-ascend/','ascend/').replace('/vllm-workspace/vllm/','base/')
def project_frames(cid):
 return [dict(stackDepth=x['stackDepth'],frame=fd[str(x['frame_id'])]) for x in chains.get(str(cid),[]) if '/vllm_ascend/' in fd[str(x['frame_id'])] or '/vllm/vllm/' in fd[str(x['frame_id'])]]
locations=defaultdict(list)
for frame,count in d['project_leaf_api_counts'].items():
 m=parse.match(frame);assert m,frame
 locations[short_file(m.group(1))].append(dict(line=int(m.group(2)),function=m.group(3),api_records=count))
for ls in locations.values():ls.sort(key=lambda x:x['line'])
CSV=Path(d['summary']['database']).parent/'kernel_details.csv';s0=CSV.stat();b=CSV.read_bytes();csv_sha=hashlib.sha256(b).hexdigest()
rows=list(csv.DictReader(b.decode('utf-8-sig').splitlines()));s1=CSV.stat();assert (s0.st_size,s0.st_mtime_ns)==(s1.st_size,s1.st_mtime_ns)
required=['Name','Type','Model ID','Task ID','Stream ID','Start Time(us)','Duration(us)'];assert rows and all(k in rows[0] for k in required)
task_by_key=defaultdict(list);task_by_available_key=defaultdict(list)
for t in tasks:task_by_key[(t['streamId'],t['taskId'],t['modelId'],t['startNs'])].append(t);task_by_available_key[(t['streamId'],t['taskId'],t['startNs'])].append(t)
csv_types=defaultdict(list);matches=[];unmatched=[];ambiguous=[];type_pairs=Counter();dur_errors=Counter();missing_fields=Counter();missing_model_db_values=Counter()
def integer(v):
 try:return int(v.strip())
 except (ValueError,AttributeError):return None
for i,r in enumerate(rows):
 csv_types[r['Type']].append(i)
 stream,task,model=integer(r['Stream ID']),integer(r['Task ID']),integer(r['Model ID'])
 start=int(Decimal(r['Start Time(us)'].strip())*1000)
 for field,value in [('Stream ID',stream),('Task ID',task),('Model ID',model)]:
  if value is None:missing_fields[field]+=1
 strength='FULL_FOUR_FIELDS' if model is not None else 'EXACT_AVAILABLE_THREE_FIELDS_MODEL_MISSING'
 candidates=(task_by_key.get((stream,task,model,start),[]) if model is not None else task_by_available_key.get((stream,task,start),[])) if stream is not None and task is not None else []
 if len(candidates)==1:
  t=candidates[0];type_pairs[(r['Type'],t['op_type'])]+=1
  dur_error=int(Decimal(r['Duration(us)'].strip())*1000)-(t['endNs']-t['startNs']);dur_errors[dur_error]+=1
  matches.append(dict(csv_row=i,task_record_rowid=t['task_record_rowid'],duration_delta_ns=dur_error,identity_strength=strength))
  if model is None:missing_model_db_values[t['modelId']]+=1
 elif candidates:ambiguous.append(dict(csv_row=i,task_record_rowids=[t['task_record_rowid'] for t in candidates]))
 else:unmatched.append(i)
seen={m['task_record_rowid'] for m in matches}
csv_type_summary=[]
match_index={m['csv_row']:m for m in matches}
for typ,indices in sorted(csv_types.items()):
 csv_type_summary.append(dict(type=typ,rows=len(indices),exact_unique_compute_matches=sum(i in match_index for i in indices),unmatched=sum(i not in match_index for i in indices)))
repeat_groups=defaultdict(list)
for t in tasks:repeat_groups[t['globalTaskId']].append(t)
repeat_examples=[]
for gid,ts in sorted(repeat_groups.items()):
 if len(ts)>1 and len(repeat_examples)<4:
  repeat_examples.append(dict(globalTaskId=gid,execution_records=len(ts),models=sorted({t['modelId'] for t in ts}),
    connections=sorted({t['connectionId'] for t in ts}),starts=[t['startNs'] for t in ts],api_counts=[len(t['api_rowids']) for t in ts],op_type=ts[0]['op_type']))
spot=[]
for typ in ['IndexFill','rejection_greedy_sample_triton','ArgMaxV2','_cache_update_kernel','SparseFlashMlaMetadata']:
 selected=[t for t in tasks if t['op_type']==typ and t['api_rowids']]
 if selected:
  t=selected[0];spot.append(dict(type=typ,task={k:v for k,v in t.items() if k!='api_rowids'},
       API=[dict(api=api[i],project_stack=project_frames(api[i]['callchainId'])) for i in t['api_rowids']]))
summary=dict(created_utc=datetime.now(timezone.utc).isoformat(),source_result_sha256=hashlib.sha256(raw).hexdigest(),source_stats_unchanged=True,
 csv_path=str(CSV),csv_sha256=csv_sha,csv_stats_unchanged=True,csv_rows=len(rows),csv_types=len(csv_types),
 exact_unique_csv_compute_matches=len(matches),identity_strength_counts=dict(Counter(m['identity_strength'] for m in matches)),missing_csv_identity_fields=dict(missing_fields),matched_missing_model_db_values=dict(missing_model_db_values),ambiguous_matches=len(ambiguous),unmatched_csv_rows=len(unmatched),
 unrepresented_compute_records=len(tasks)-len(seen),duration_delta_ns_hist=dict(dur_errors),
 exact_type_pair_counts=[dict(csv_type=a,db_type=b,records=n) for (a,b),n in sorted(type_pairs.items())],
 csv_unmatched_type_counts=dict(Counter(rows[i]['Type'] for i in unmatched)),
 no_target_source_read=True,no_device_or_inference_work=True,whole_modules_closed=0,overall_complete=False,
 limits=['Full four-field identity matches and exact available stream/task/start matches with missing CSV model are counted separately; no N/A sentinel is guessed, no nearest-time or duration-only repair.',
         'Source API leaf is the first project frame in ascending stackDepth; selected full stacks retained to validate direction.',
         'One-rank diagnostic capture and later official capture remain separate; differences are not automatically aliases.',
         'Repeated task definitions without linked PYTORCH_API remain unassigned; no dummy/formal phase invented.',
         'CSV count conservation and source assignment are not all-condition semantic closure.'])
out=A/'ep3-origin-source-and-csv-audit-v9-v2.json';assert not out.exists()
full=dict(summary=summary,project_leaf_locations=dict(locations),project_file_api_counts={short_file(k):v for k,v in d['project_file_api_counts'].items()},csv_type_summary=csv_type_summary,
          csv_compute_matches=matches,unmatched_csv_row_indices=unmatched,ambiguous_matches=ambiguous,repeated_definition_examples=repeat_examples,selected_type_source_stacks=spot)
payload=json.dumps(full,ensure_ascii=False,indent=2)+'\n';out.write_text(payload)
public=dict(summary={k:v for k,v in summary.items() if k!='exact_type_pair_counts'},full_result=str(out),result_sha256=hashlib.sha256(payload.encode()).hexdigest(),
            project_leaf_locations=dict(locations),selected_type_source_stacks=[x for x in spot if x['type']=='IndexFill'],repeated_definition_examples=repeat_examples[:1])
assert len(json.dumps(public,ensure_ascii=False).encode())<=32000
print(json.dumps(public,ensure_ascii=False))
