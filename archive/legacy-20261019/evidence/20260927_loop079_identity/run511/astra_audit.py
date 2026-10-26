#!/usr/bin/env python3
"""Independent CPU-only Run508 evidence review. Writes only Run511."""
import ast, collections, hashlib, importlib.util, json, re, sys
from pathlib import Path
ROOT=Path('/data/wio/Inference_Foundry')
RUN=ROOT/'evidence/20260927_loop079_identity/run508'
B=RUN/'b_candidate'
OUT=ROOT/'evidence/20260927_loop079_identity/run511'
OUT.mkdir(exist_ok=True)
inputs={}
def read(p):
 p=Path(p); b=p.read_bytes(); inputs[str(p)]={'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b)}; return b
def j(p):return json.loads(read(p))
def sha(b):return hashlib.sha256(b).hexdigest()
def load(name,p):
 read(p); s=importlib.util.spec_from_file_location(name,p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m);return m
sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT/'scripts'))
for p in sorted(RUN.rglob('*')):
 if p.is_file() and '__pycache__' not in p.parts:read(p)
for name in ['AGENTS.md','MISSION.md','PROJECT_STATE.md','PERFORMANCE_MAP.md','ACHIEVABLE_BOUND.md','RESULTS.md']: read(ROOT/name)
for p in (RUN/'preflight/reviewed_sources').glob('*'):
 assert read(p)==read(ROOT/'scripts'/p.name),p
v=load('review_validator',ROOT/'scripts/loop079_target_frontier_validate_run507.py')
p=load('review_patcher',ROOT/'scripts/loop079_target_frontier_patch_run507.py')
assert v.validate_root(B)==j(B/'validation.json')
assert v.final_admit(B)==j(B/'final_admission.json')
manifest=j(B/'patch_state/manifest.json')
for r in manifest:
 old=read(B/'patch_state'/f"{r['key']}.orig")
 assert sha(old)==r['original']==sha(read(r['source']))
 assert sha(p.patch(r['key'],old.decode()).encode())==r['patched']
assert read(B/'source_before.sha256')==read(B/'source_after.sha256')
mla=Path('/data/wio/vllm_ascend_26/framework/vllm-ascend/vllm_ascend/attention/mla_v1.py')
source=read(mla); assert sha(source)=='67b4479cca16fd164b33b91b2fa4f48e95f2474720e484abdf3b3897a8ea1545'
(OUT/'mla_v1.py').write_bytes(source)
cls=next(x for x in ast.parse(source).body if isinstance(x,ast.ClassDef) and x.name=='AscendMLAImpl')
fn=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='update_graph_params')
loops=[x for x in ast.walk(fn) if isinstance(x,ast.For)]
assert len(loops)==1 and isinstance(loops[0].iter,ast.Call) and ast.unparse(loops[0].iter.func)=='zip'
assert not any(isinstance(x,(ast.Continue,ast.Break)) for x in ast.walk(loops[0]))
assert [ast.unparse(x.func) for x in ast.walk(loops[0]) if isinstance(x,ast.Call) and ast.unparse(x.func)=='event.record']==['event.record']
rows={(r,c):j(B/'capture'/f'rank{r}_cohort{c}.json') for r in range(8) for c in range(1,6)}
runtime={(r,c):j(B/'runtime'/f'rank{r}_cohort{c}.json') for r,c in rows}
uuid=read(B/'run_uuid.txt').decode().strip(); unique_requests=set(); graphstats=[]
for r in range(8):
 base=rows[r,1]; bs=base['replay']; acts=[]
 meta=j(B/'graph_dump'/f'rank{r}_cohort5_acl_graph.meta.json')
 backend=meta['graph_update_backend']
 for c in range(1,6):
  d=rows[r,c]; a=d['actual_update']; rep=d['replay']; rt=runtime[r,c]
  assert d['run_id']==uuid and d['rank']==r and d['cohort']==c and d['cycle_index']==64
  assert d['pid']==base['pid']
  for k in ['entry_id','graph_id','capture_generation','arguments','output_owner','capture_scope','batch_descriptor']:assert rep[k]==bs[k],(r,c,k)
  assert rep['capture_generation']==1 and rep['selected_observation_ordinal']==c
  assert a['attention_key_count']==170 and a['draft_metadata_argument_is_none'] is True
  assert a['list_counts']=={'attn_params':0,'handles':0,'events':0}
  assert a['mla_params_capture']==a['mla_params_selected']==base['actual_update']['mla_params_capture']
  assert all(x=={'count':0,'object_ids':[]} for x in a['mla_params_selected']['lists'].values())
  assert a['zip_iteration_count']==min(a['attention_key_count'],*a['list_counts'].values())==0
  assert a['source_inferred_successful_event_records']==0 and a['return_count']==1
  assert a['native_event_id']==a['device_event_completion']=='unobserved'
  assert a['update_callable']=={k:backend['update_graph_params'][k] for k in ['callable_id','owner_id']}
  assert a['backend_id']==backend['attn_backend']['object_id'] and a['impl_id']==backend['impl']['object_id']
  assert a['replay_capture_generation']==rep['capture_generation'] and a['replay_entry_id']==rep['entry_id'] and a['replay_graph_id']==rep['graph_id']
  assert a['replay_selected_ordinal']==c
  assert d['setup']['graph_update_bound']['backend']==backend['attn_backend']
  assert d['setup']['graph_update_bound']['get_impl_cls']==backend['get_impl_cls']
  assert a['configured_update_stream']=={'stream_id':102,'device_index':r,'device_type':20}
  assert d['graph_output']==d['pre_gather']==rep['output_owner']
  assert rt['pass'] and rt['target_graph_mode']=='FULL' and rt['host_mirror_exact']
  assert rt['cycles']==d['cycles'] and rt['generated_output_counts']==[1024]*12
  assert rt['staged_output_counts']==[sum(row[i] for row in d['accepted_counts']) for i in range(12)]
  assert d['accepted_counts']==rows[0,c]['accepted_counts'] and rt['req_ids']==runtime[0,c]['req_ids']
  acts.append(a)
 assert backend['selected_actual_update']==acts[-1]
 assert backend['update_graph_params']['source']['sha256']==sha(source)
 raw=read(B/'graph_dump'/f'rank{r}_cohort5_acl_graph.json');nodes=json.loads(raw)
 assert sha(raw)==meta['sha256'] and len(raw)==meta['bytes'] and len(nodes)==meta['node_count']==5412
 assert all(n['pid']==f"{base['pid']} aclGraph" for n in nodes)
 cnt=collections.Counter(n['args']['Stream Id'] for n in nodes); kinds=collections.Counter(n['args']['Task Type'] for n in nodes)
 assert cnt=={0:1040,1:3985,99:387}
 byid={(n['args']['Stream Id'],n['args']['Task Id']):n for n in nodes};assert len(byid)==len(nodes)
 edges={k:set() for k in byid};rev={k:set() for k in byid}
 for s,n in cnt.items():
  assert {t for ss,t in byid if ss==s}==set(range(n))
  for t in range(n-1):edges[s,t].add((s,t+1))
 events={kind:{} for kind in ['EVENT_RECORD','EVENT_WAIT','EVENT_RESET']}
 for k,n in byid.items():
  kind=n['args']['Task Type']
  if kind in events:
   suffix=int(n['name'].rsplit('_',1)[1]); assert suffix not in events[kind];events[kind][suffix]=k
 assert set(events['EVENT_RECORD'])==set(events['EVENT_RESET']) and set(events['EVENT_WAIT'])<=set(events['EVENT_RECORD'])
 for e,k in events['EVENT_WAIT'].items():edges[events['EVENT_RECORD'][e]].add(k)
 for k,vs in edges.items():
  for dest in vs:rev[dest].add(k)
 indeg={k:len(vs) for k,vs in rev.items()};queue=collections.deque(k for k,n in indeg.items() if n==0); topo=[]
 while queue:
  k=queue.popleft();topo.append(k)
  for dest in edges[k]:
   indeg[dest]-=1
   if indeg[dest]==0:queue.append(dest)
 assert len(topo)==5412
 terminals=[k for k,n in byid.items() if n['args']['Task Type']=='NOTIFY_RECORD']; assert terminals==[(1,3984)]
 reach=set(terminals);queue=list(terminals)
 while queue:
  for prev in rev[queue.pop()]:
   if prev not in reach:reach.add(prev);queue.append(prev)
 assert len(reach)==5412
 graphstats.append({'rank':r,'tasks':len(nodes),'streams':dict(cnt),'kinds':dict(kinds),'conditional_dag_acyclic':True,'all_tasks_reach_internal_notify':True})
for c in range(1,6):
 ids=runtime[0,c]['req_ids'];assert not unique_requests.intersection(ids);unique_requests.update(ids)
assert len(unique_requests)==60
clients=[j(B/f) for f in ['warmup48.json','bench.json']]
requests=[]
for count,client in zip([48,12],clients):
 assert len(client['requests'])==count and all(x['error'] is None and x['output_tokens']==1024 for x in client['requests'])
 requests+=client['requests']
assert max(x['end'] for x in clients[0]['requests'])<=min(x['start'] for x in clients[1]['requests'])
active=peak=0
for _,delta in sorted([(x['start'],1) for x in requests]+[(x['end'],-1) for x in requests]):active+=delta;peak=max(peak,active)
assert active==0 and peak==12
logpath=Path(read(B/'server_log_path.txt').decode().strip());log=read(logpath).decode(errors='replace'); lines=log.splitlines()
posts=[x for x in lines if 'POST /v1/chat/completions' in x];assert len(posts)==60 and all(re.search(r'HTTP/[^" ]+"\s+200\b',x) for x in posts)
errs=[i for i,x in enumerate(lines) if ' ERROR ' in x];assert len(errs)==8 and min(errs)>next(i for i,x in enumerate(lines) if 'Parent process exited' in x)
(OUT/'shutdown_log_excerpt.txt').write_text('\n'.join(lines[-20:])+'\n')
status=dict(x.split('=',1) for x in read(B/'cleanup_status.txt').decode().splitlines());assert len(status)==8 and set(status.values())=={'0'}
hbm=[int(x) for x in re.findall(r'(\d+)\s*/\s*65536',read(B/'stop_npu_probe.txt').decode())];assert len(hbm)==8 and max(hbm)<6144
assert read(B/'stop_process_probe.txt').decode().strip()=='[]' and read(B/'stop_orphan_bench.txt').decode().strip()=='[]'
# Hash the reviewed comparison and strict-invalid predecessor records; never admit Run507.
read(ROOT/'evidence/20260927_loop079_identity/run503/astra_run502_graph_review.md')
invalid=j(RUN/'preflight/Run507_final_remains_invalid.json')
results={'verdict':'PASS_SCOPED','run_id':uuid,'rows':40,'ranks':8,'cohorts':5,'cycles':[runtime[0,c]['cycles'] for c in range(1,6)],'attention_key_counts':[170],'parameter_handle_event_counts':[0,0,0],'source_inferred_zip_and_event_record_counts':[0],'source_inference_not_native_tracing':True,'exact_1024_outputs':60,'max_client_concurrency':peak,'runtime_unique_request_ids':60,'server_error_lines_in_shutdown_context':len(errs),'cleanup':status,'stop_hbm_mb':hbm,'graph_stats':graphstats,'source_restore_and_patch_reconstruction':True,'finite_bound_endpoints':None,'formal_current_tok_s':571.681}
(OUT/'independent_results.json').write_text(json.dumps(results,indent=2)+'\n')
# Verify input immutability after all analysis, then publish the manifest.
for path,entry in inputs.items():assert sha(Path(path).read_bytes())==entry['sha256'],path
(OUT/'input_sha256_manifest.json').write_text(json.dumps({'inputs':inputs,'count':len(inputs),'rechecked_at_review_end':True},indent=2)+'\n')
print(json.dumps({k:v for k,v in results.items() if k!='graph_stats'},indent=2));print('input_count',len(inputs))
