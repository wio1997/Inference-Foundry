from pathlib import Path
from collections import Counter,defaultdict
from decimal import Decimal
from datetime import datetime,timezone
import json,hashlib,socket,time
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
A=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/profile-readonly-20261031')
def pinned(p,sha):
 x=p.stat();b=p.read_bytes();y=p.stat();assert hashlib.sha256(b).hexdigest()==sha;assert (x.st_size,x.st_mtime_ns)==(y.st_size,y.st_mtime_ns);return b
s=json.loads(pinned(A/'ep3-outer-segments-and-memory-tasks-v11.json','bce26ccaa1c16eb6d5c979d8d33feaf7f1efd3381a3317cb7211c0b3483bda2f'))
g=json.loads(pinned(A/'ep3-vector-overlay-and-graph-blocks-v10-v3.json','587b81f07b1b8c3057b9b7486bec6d58ce7aa946909d9385d8d4915bd6012c84'))
copies=[t for t in s['memory_TASK_rows'] if t['same_model_control_pair_candidates']]
assert len(copies)==561 and all(len(t['same_model_control_pair_candidates'])==1 for t in copies)
by_time=defaultdict(list)
for t in copies:by_time[(t['startNs'],t['endNs'])].append(t)
trace=Path(g['summary']['database_sha256']) if False else Path('/data/tiankuan/wio/migration-166-20261030/source169-project/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029/runtime/diagnostic-traces/dp0_pp0_tp3_dcp0_ep3_rank3_2128_20261029023525650_ascend_pt/ASCEND_PROFILER_OUTPUT/trace_view.json')
before=trace.stat();assert 0<before.st_size<=2000000000
h=hashlib.sha256()
with trace.open('rb') as f:
 for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
assert h.hexdigest()=='c1eb4df8708a10d2c0be9af84ef8ed2a18f29c19f6297eb1f62aa6743ecaaa85'
def events(path):
 dec=json.JSONDecoder(parse_float=Decimal)
 with path.open(encoding='utf-8') as f:
  buf=f.read(1024*1024);prefix=buf.lstrip()
  if prefix.startswith('['):pos=buf.index('[')+1
  else:
   k=buf.index('"traceEvents"');pos=buf.index('[',k)+1
  while True:
   while pos<len(buf) and buf[pos] in ' \r\n\t,':pos+=1
   if pos<len(buf) and buf[pos]==']':return
   try:e,end=dec.raw_decode(buf,pos)
   except json.JSONDecodeError:
    tail=buf[pos:];assert len(tail)<4*1024*1024,'singleevent budget';more=f.read(1024*1024)
    assert more,'unexpected trace EOF';buf=tail+more;pos=0;continue
   pos=end;yield e
   if pos==len(buf):
    buf=f.read(1024*1024);pos=0;assert buf,'missingtrace array end'
matched=defaultdict(list);scanned=0;copy_names=Counter();t0=time.monotonic()
for e in events(trace):
 scanned+=1
 if scanned%50000==0:assert time.monotonic()-t0<120,'trace scan CPUtime budget'
 if e.get('ph')!='X':continue
 if any(k in str(e.get('name','')).lower() for k in ['memcpy','memset']):copy_names[e.get('name')]+=1
 if 'ts' not in e or 'dur' not in e:continue
 start=int(Decimal(str(e['ts']))*1000);end=start+int(Decimal(str(e['dur']))*1000)
 for t in by_time.get((start,end),[]):
  matched[t['task_record_rowid']].append(dict(name=e.get('name'),cat=e.get('cat'),pid=e.get('pid'),tid=e.get('tid'),ts=str(e['ts']),dur=str(e['dur']),args=e.get('args',{})))
assert scanned<=5000000
after=trace.stat();assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
argkeys=Counter(k for rows in matched.values() for e in rows for k in e['args'])
copy_results=[]
for t in copies:
 es=matched[t['task_record_rowid']]
 copy_results.append(dict(task=t,exact_interval_trace_events=es))
first_segments=[x for x in s['outer_segments'] if x['execution']==0]
engram_pattern_checks=[]
for x in first_segments:
 if x['location'] not in ['GAP_0_1','GAP_13_14']:continue
 for eager in s['eager_source_bound_windows']:
  engram_pattern_checks.append(dict(graph_gap=x['location'],eager_label=eager['label'],graph_records=x['records'],eager_records=eager['records'],
    eager_first_last_types=[eager['type_sequence'][0],eager['type_sequence'][-1]],graph_equals_eager_without_boundary_copies=x['type_sequence']==eager['type_sequence'][1:-1]))
copy_locations=[]
for t in copies:
 e=g['graph_executions'][t['same_model_control_pair_candidates'][0]];blocks=e['layer_blocks'];labels=[]
 if t['endNs']<=blocks[0]['startNs']:labels=['BEFORE_FIRST_MARKER']
 elif t['startNs']>=blocks[-1]['endNs']:labels=['AFTER_LAST_MARKER']
 else:
  for b in blocks:
   if b['startNs']<=t['startNs'] and t['endNs']<=b['endNs']:labels.append('INSIDE_BLOCK_'+str(b['observed_layer_block_ordinal']))
  for left,right in zip(blocks,blocks[1:]):
   if left['endNs']<=t['startNs'] and t['endNs']<=right['startNs']:labels.append('GAP_'+str(left['observed_layer_block_ordinal'])+'_'+str(right['observed_layer_block_ordinal']))
 copy_locations.append(dict(task_record_rowid=t['task_record_rowid'],globalTaskId=t['globalTaskId'],graph=e['ordinal'],locations=labels or ['CROSSING_BOUNDARIES'],streamId=t['streamId'],taskId=t['taskId'],trace_event_count=len(matched[t['task_record_rowid']])) )
summary=dict(created_utc=datetime.now(timezone.utc).isoformat(),trace_path=str(trace),trace_sha256=h.hexdigest(),trace_stats_unchanged=True,trace_bytes=before.st_size,trace_events_scanned=scanned,
 graph_MEMCPY_tasks=len(copies),exact_interval_trace_match_count_hist=dict(Counter(len(matched[t['task_record_rowid']]) for t in copies)),
 trace_copy_name_counts=dict(copy_names),matched_event_argument_key_counts=dict(argkeys),
 first_graph_memory_locations=dict(Counter(label for x in copy_locations if x['graph']==0 for label in x['locations'])),
 graph_eager_Engram_pattern_checks=engram_pattern_checks,no_target_source_read=True,no_device_or_inference_work=True,whole_modules_closed=0,overall_complete=False,
 limits=['Exact start/end trace correspondence is retained with all candidate eventargs; no nearest-time match or inferred bytes/pointers.',
 'A copy temporal location and identical type sequence do not prove producer/consumer or storage identity.',
 'Compute38vs40difference is recorded. Whether boundary copies became MEMCPY requires independent event/call attribution; absent TensorMove is not no-copy.'])
out=A/'ep3-copy-trace-and-outer-patterns-v11.json';assert not out.exists()
payload=json.dumps(dict(summary=summary,copy_trace_correspondences=copy_results,copy_temporal_locations=copy_locations),ensure_ascii=False,indent=2,default=str)+'\n';out.write_text(payload)
examples=[]
for r in copy_results[:4]:
 examples.append(dict(task=r['task'],events=[dict(name=e['name'],cat=e['cat'],pid=e['pid'],tid=e['tid'],args=e['args']) for e in r['exact_interval_trace_events']]))
public=dict(summary=summary,full_result=str(out),result_sha256=hashlib.sha256(payload.encode()).hexdigest(),first_copy_events=examples)
if len(json.dumps(public,ensure_ascii=False,default=str).encode())>30000:public['first_copy_events']=[];public['display_truncated']=True
print(json.dumps(public,ensure_ascii=False,default=str))
