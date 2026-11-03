from pathlib import Path
from collections import Counter,defaultdict
from datetime import datetime,timezone
import json,hashlib,csv,re,socket
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
A=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/profile-readonly-20261031')
def pinned(p,sha):
 x=p.stat();b=p.read_bytes();y=p.stat();assert hashlib.sha256(b).hexdigest()==sha;assert (x.st_size,x.st_mtime_ns)==(y.st_size,y.st_mtime_ns);return b
d=json.loads(pinned(A/'ep3-all-compute-origins-v9-v2.json','fdce699c6cf2fdf15b9ba85c5e9e5a5cd7893d2b34a0f69b0321e719a9ef6b9d'))
g=json.loads(pinned(A/'ep3-vector-overlay-and-graph-blocks-v10-v3.json','587b81f07b1b8c3057b9b7486bec6d58ce7aa946909d9385d8d4915bd6012c84'))
phase=json.loads(pinned(A/'ep3-graph-schedule-and-block-proxies-v10.json','747f6d1b71ae3c957d328419b2b137c8ad26a2c4608fe9bc2a5b565a3b8065b5'))
ROOT=Path(d['summary']['database']).parents[2]
rank_re=re.compile(r'^dp([01])_pp\d+_tp\d+_dcp\d+_ep(\d+)_rank\d+_(\d+)_\d+_ascend_pt$')
ranks=[];pins=[];rank_rows=[]
for p in ROOT.iterdir():
 m=rank_re.match(p.name)
 if m:ranks.append((int(m[2]),p))
assert len(ranks)==16 and {ep for ep,p in ranks}==set(range(16))
for ep,p in sorted(ranks):
 f=p/'ASCEND_PROFILER_OUTPUT/kernel_details.csv';before=f.stat();assert 0<before.st_size<=50000000
 b=f.read_bytes();after=f.stat();assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
 sha=hashlib.sha256(b).hexdigest()
 if ep==3:assert sha=='46d994119151ad1ccd8c5355262e817ed038cd0fc70cfc8ef0cf8e2b6f834197'
 pins.append(dict(ep=ep,path=str(f),bytes=len(b),sha256=sha))
 rows=list(csv.DictReader(b.decode('utf-8-sig').splitlines()));hist=Counter();selected=[]
 for i,r in enumerate(rows):
  if r['Type']!='DispatchFFNCombine':continue
  shapes=[tuple(int(v.strip()) for v in part.split(',')) if part.strip() else () for part in r['Input Shapes'].strip().strip('"').split(';')]
  assert len(shapes)>=4 and len(shapes[0])==2 and len(shapes[1])==3 and len(shapes[3])==2
  model=int(r['Model ID']);M=shapes[0][0];E=shapes[1][0];K=shapes[3][1]
  key=(model,M,E,K,r['Input Data Types'],r['Output Data Types'],r['Accelerator Core'],r['Block Num'],r['Mix Block Num']);hist[key]+=1
  if model!=4294967295:selected.append(dict(csv_row=i,model_id=model,M=M,local_experts=E,K=K,task_id=r['Task ID'],stream_id=r['Stream ID'],start_us=r['Start Time(us)'],duration_us=r['Duration(us)']))
 regimes=[dict(model_id=k[0],physical_M=k[1],local_experts=k[2],K=k[3],input_dtypes=k[4],output_dtypes=k[5],core=k[6],block_num=k[7],mix_block_num=k[8],records=n) for k,n in sorted(hist.items())]
 rank_rows.append(dict(ep=ep,csv_rows=len(rows),all_MC2_rows=sum(hist.values()),regimes=regimes,retained_graph_MC2_rows=len(selected),graph_MC2_records=selected))
tm={t['task_record_rowid']:t for t in d['tasks']};root_rows=[];phase_map={x['ordinal']:x['status'] for x in phase['graph_schedule_bindings']}
for e in g['graph_executions']:
 assigned={i for block in e['layer_blocks'] for i in block['task_record_rowids']};ts=[tm[i] for i in e['task_record_rowids'] if i not in assigned]
 lo=e['layer_blocks'][0]['startNs'];hi=e['layer_blocks'][-1]['endNs'];groups=defaultdict(list)
 for t in ts:
  region='BEFORE_FIRST_MARKER' if t['endNs']<=lo else 'AFTER_LAST_MARKER' if t['startNs']>=hi else 'BETWEEN_OR_CROSSING_MARKER_BLOCKS'
  key=json.dumps([region,t['op_type'],t['input_shapes'],t['output_shapes'],t['input_dtypes'],t['output_dtypes'],t['streamId']])
  groups[key].append(t)
 rs=[]
 for key,vs in groups.items():
  region,typ,ish,osh,idt,odt,stream=json.loads(key)
  rs.append(dict(region=region,type=typ,input_shapes=ish,output_shapes=osh,input_dtypes=idt,output_dtypes=odt,stream_id=stream,records=len(vs),
   definition_ids=sorted(t['globalTaskId'] for t in vs),task_record_rowids=[t['task_record_rowid'] for t in vs]))
 root_rows.append(dict(execution=e['ordinal'],schedule_status=phase_map[e['ordinal']],outside_records=len(ts),region_counts=dict(Counter('BEFORE_FIRST_MARKER' if t['endNs']<=lo else 'AFTER_LAST_MARKER' if t['startNs']>=hi else 'BETWEEN_OR_CROSSING_MARKER_BLOCKS' for t in ts)),signature_groups=rs))
graph_rows=[r for rank in rank_rows for r in rank['graph_MC2_records']]
allgraph=bool(graph_rows) and all(r['M']==3 and r['local_experts']==24 and r['K']==6 for r in graph_rows)
summary=dict(created_utc=datetime.now(timezone.utc).isoformat(),rank_count=len(ranks),all_source_stats_unchanged=True,
 all_MC2_rows=sum(r['all_MC2_rows'] for r in rank_rows),retained_graph_MC2_rows=len(graph_rows),retained_graph_MC2_rows_per_rank=[r['retained_graph_MC2_rows'] for r in rank_rows],
 all_retained_graph_MC2_are_M3_E24_K6=allgraph,physical_M_regime_counts=dict(Counter(str(reg['physical_M']) for r in rank_rows for reg in r['regimes'] for _ in range(reg['records']))),
 current_EP3_graph_schedule_class='10bound cached-decode,1unbound. No equivalent per-call schedule attestation for other15ranks in this job.',
 outside_block_records_per_execution=[r['outside_records'] for r in root_rows],first_outside_region_counts=root_rows[0]['region_counts'],first_outside_signature_groups=len(root_rows[0]['signature_groups']),
 no_target_source_read=True,no_device_or_inference_work=True,whole_modules_closed=0,overall_complete=False,
 limits=['All MC2 CSV records in one16rank diagnostic capture; other captures and observer snapshots remain separate.',
 'PhysicalM3 signatures do not attest actual masks, receiver counts, same-call EPcohorts or all-rank valid-token totals.',
 'Current EP3 schedule binds400target-like marker blocks; the11graph ordinals on other ranks are not assumed same-call or time-aligned.',
 'No Model ID sentinel normalization is applied to other operator types; known242Triton discrepancies remain scoped to EP3 overlay.',
 'Outside-marker shape/type groups are an inventory, not a semantic consumer/module proof or evidence of idle/removable time.'])
out=A/'allrank-moe-regimes-and-ep3-graph-roots-v10.json';assert not out.exists()
payload=json.dumps(dict(summary=summary,source_manifest=pins,rank_rows=rank_rows,EP3_outside_marker_rows=root_rows),ensure_ascii=False,indent=2)+'\n';out.write_text(payload)
public=dict(summary=summary,full_result=str(out),result_sha256=hashlib.sha256(payload.encode()).hexdigest(),
 rank_regimes=[{k:v for k,v in r.items() if k!='graph_MC2_records'} for r in rank_rows],first_outside_groups=root_rows[0]['signature_groups'])
assert len(json.dumps(public,ensure_ascii=False).encode())<=32000
print(json.dumps(public,ensure_ascii=False))
