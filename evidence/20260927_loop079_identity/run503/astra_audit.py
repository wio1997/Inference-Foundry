import json,hashlib,re,collections,sys,subprocess,ast
from pathlib import Path
root=Path('/data/wio/Inference_Foundry');b=root/'evidence/20260927_loop079_identity/run502/b_candidate';out=root/'evidence/20260927_loop079_identity/run503';out.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(root/'scripts'));import loop079_target_frontier_validate_run494 as v
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
result={'root_validation':v.validate_root(b),'final_validation':v.final_admit(b)}
rows=[json.loads(p.read_text()) for p in sorted((b/'capture').glob('*.json'))]
by={(d['rank'],d['cohort']):d for d in rows}
result['cohort_cycles']=[by[0,c]['cycles'] for c in range(1,6)]
result['per_rank_stable_owner']=[]
for r in range(8):
 rs=[by[r,c] for c in range(1,6)]
 keys=['entry_id','graph_id','capture_generation','batch_descriptor','output_owner']
 stable=all(all(row['replay'][k]==rs[0]['replay'][k] for row in rs) for k in keys)
 assert stable and [x['replay']['selected_observation_ordinal'] for x in rs]==list(range(1,6))
 result['per_rank_stable_owner'].append(r)
# Rerun actual server and client gates independently of saved counts.
log=Path((b/'server_log_path.txt').read_text().strip());lines=log.read_text(errors='replace').splitlines()
posts=[l for l in lines if 'POST /v1/chat/completions' in l];ok=[l for l in posts if re.search(r'POST /v1/chat/completions HTTP/[^" ]+"\s+200\b',l)]
assert len(posts)==len(ok)==60
errors=[l for l in lines if re.search(r'\b(?:ERROR|Traceback|OUT_OF_SCOPE|RuntimeError|AssertionError)\b',l)]
result['server']={'posts':len(posts),'http200':len(ok),'error_lines':len(errors)}
clients=[json.loads((b/name).read_text()) for name in ['warmup48.json','bench.json']]
allreq=[q for d in clients for q in d['requests']]
assert len(allreq)==60 and all(q['error'] is None and q['output_tokens']==1024 for q in allreq)
assert max(q['end'] for q in clients[0]['requests'])<=min(q['start'] for q in clients[1]['requests'])
sweep=sorted([(q['start'],1) for q in allreq]+[(q['end'],-1) for q in allreq]);active=peak=0
for _,d in sweep:active+=d;peak=max(peak,active)
assert active==0 and peak==12
result['clients']={'requests':60,'exact1024':True,'combined_peak':peak,'warmup_before_diagnostic':True}
for row in v.SOURCE_MANIFEST:
 p=Path(row['source']);backup=b/'patch_state'/f"{row['key']}.orig"
 assert sha(p)==sha(backup)==row['original']
result['six_live_sources_and_backup_pins']=True
backend=Path('/data/wio/vllm_ascend_26/framework/vllm-ascend/vllm_ascend/attention/mla_v1.py')
for r in range(8):
 m=json.loads((b/f'graph_dump/rank{r}_cohort5_acl_graph.meta.json').read_text())
 assert all(z['source']['sha256']==sha(backend) for k,z in m['graph_update_backend'].items() if isinstance(z,dict) and 'source' in z)
(out/'mla_v1.py').write_bytes(backend.read_bytes());result['backend_source_sha256']=sha(backend)
# Archive corresponding installed kernel metadata and implementation source using read-only container cat.
base='/usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe'
paths={
 'rms_kernel.json':base+'/kernel/ascend910b/ops_nn/rms_norm/RmsNorm_normal_all_bf16_high_performance.json',
 'mean_kernel.json':base+'/kernel/ascend910b/ops_legacy/reduce_mean/ReduceMean_9a2479d4dc5148645cb3f4b9e765c637_high_precision.json',
 'rms_norm.cpp':base+'/impl/ops_nn/ascendc/rms_norm/rms_norm.cpp',
 'reduce_mean.py':base+'/impl/ops_legacy/dynamic/reduce_mean.py',
}
installed=[]
for name,path in paths.items():
 data=subprocess.check_output(['docker','exec','vllm-ascend26-dsv4f-w4a8','cat',path]);(out/name).write_bytes(data)
 record={'file':name,'installed_path':path,'sha256':hashlib.sha256(data).hexdigest()}
 if name.endswith('.json'):
  j=json.loads(data);binpath=str(Path(path).with_suffix('.o'));binary=subprocess.check_output(['docker','exec','vllm-ascend26-dsv4f-w4a8','cat',binpath])
  record.update(kernel_name=j['kernelName'],binary_path=binpath,binary_sha256=hashlib.sha256(binary).hexdigest(),metadata_binary_sha=j.get('sha256'),inputs=j['supportInfo']['inputs'],outputs=j['supportInfo']['outputs'])
 installed.append(record)
result['installed_kernel_candidates']=installed

def leaves(t):return [t['value']] if t['kind']=='tensor' else [x for c in t['children'] for x in leaves(c)]
graphs=[]
for rank in range(8):
 p=b/f'graph_dump/rank{rank}_cohort5_acl_graph.json';n=json.loads(p.read_text());r=by[rank,5];m=json.loads(p.with_suffix('.meta.json').read_text());assert sha(p)==m['sha256']
 streams=collections.defaultdict(list);events={k:{} for k in ('RECORD','WAIT','RESET')};edges=collections.defaultdict(set)
 for i,x in enumerate(n):
  a=x['args'];streams[a['Stream Id']].append((a['Task Id'],i))
  if a['Task Type'].startswith('EVENT_'):
   match=re.fullmatch(a['Task Type']+r'_(\d+)',x['name']);assert match
   key=int(match[1]);assert key not in events[a['Task Type'][6:]];events[a['Task Type'][6:]][key]=i
 for stream,pairs in streams.items():
  pairs.sort();assert [t for t,i in pairs]==list(range(len(pairs)))
  for (_,i),(_,j) in zip(pairs,pairs[1:]):edges[i].add(j)
 rec,wait,reset=[events[k] for k in ('RECORD','WAIT','RESET')]
 assert set(wait)<=set(rec) and set(reset)==set(rec)
 for ev,i in wait.items():edges[rec[ev]].add(i)
 indeg=[0]*len(n)
 for js in edges.values():
  for j in js:indeg[j]+=1
 q=collections.deque(i for i,vv in enumerate(indeg) if vv==0);order=[]
 while q:
  i=q.popleft();order.append(i)
  for j in edges[i]:
   indeg[j]-=1
   if indeg[j]==0:q.append(j)
 assert len(order)==len(n)
 reachable=[0]*len(n)
 for i in reversed(order):
  for j in edges[i]:reachable[i]|=(1<<j)|reachable[j]
 notify=[i for i,x in enumerate(n) if x['args']['Task Type']=='NOTIFY_RECORD'];assert len(notify)==1
 assert all(i==notify[0] or reachable[i]&(1<<notify[0]) for i in range(len(n)))
 def ref(i):return {'stream':n[i]['args']['Stream Id'],'task':n[i]['args']['Task Id'],'name':n[i]['name']}
 lr=[]
 for li,leaf in enumerate(leaves(r['graph_output'])):
  hits=[];ranges=[]
  for i,x in enumerate(n):
   vals=[int(h,16) for h in re.findall(r'0x[0-9a-fA-F]+',x['args'].get('Kernel Args',''))]
   positions=[j for j,h in enumerate(vals) if h==leaf['data']]
   if positions:hits.append({'node_index':i,**ref(i),'raw_hex_word_positions':positions})
   inside=[j for j,h in enumerate(vals) if leaf['storage']<=h<leaf['storage']+leaf['storage_bytes']]
   if inside:ranges.append(i)
  maximal=[h for h in hits if not any((reachable[h['node_index']]&(1<<z['node_index'])) for z in hits if z is not h)]
  range_maximal=[i for i in ranges if not any((reachable[i]&(1<<j)) for j in ranges if j!=i)]
  lr.append({'leaf':li,'tensor':leaf,'data_hex':hex(leaf['data']),'exact_occurrences':len(hits),'hit_streams':dict(collections.Counter(h['stream'] for h in hits)),
   'maximal_occurrences_in_candidate_dag':maximal,'range_maximal_occurrences':[ref(i) for i in range_maximal],
   'candidate_kernel_args':[n[h['node_index']]['args']['Kernel Args'] for h in maximal],
   'candidate_reaches_notify':all(reachable[h['node_index']]&(1<<notify[0]) for h in maximal),
   'candidate_reaches_stream0_last':all(reachable[h['node_index']]&(1<<streams[0][-1][1]) for h in maximal),
   'all_hits':hits})
 graphs.append({'rank':rank,'graph_meta':v.native_tasks(n),'task_types':dict(collections.Counter(x['args']['Task Type'] for x in n)),
 'record_count':len(rec),'wait_count':len(wait),'reset_count':len(reset),'record_without_wait':len(set(rec)-set(wait)),
 'record_wait_stream_pairs':dict(collections.Counter(str((n[rec[e]]['args']['Stream Id'],n[i]['args']['Stream Id'])) for e,i in wait.items())),
 'conditional_dag_acyclic':True,'all_tasks_reach_notify':True,'notify':ref(notify[0]),'last_each_stream':{s:ref(pairs[-1][1]) for s,pairs in streams.items()},'leaves':lr})
result['graphs']=graphs
# Hash all retained acquisition inputs, plus actual source/log references. No reduction input is left unpinned.
files=sorted(p for p in b.rglob('*') if p.is_file());files += [log,backend]
files+=sorted(root/line.split()[1] for line in (root/'evidence/20260927_loop079_identity/run501/reviewed_scripts.sha256').read_text().splitlines())
files=sorted(set(files));manifest=[{'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)} for p in files]
(out/'input_hashes.json').write_text(json.dumps(manifest,indent=2)+'\n');result['hashed_inputs']=len(manifest);result['candidate_files']=sum(p.is_file() for p in b.rglob('*'))
(out/'independent_results.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ['graphs','installed_kernel_candidates']},indent=2))
print(json.dumps({'rank0_leaves':[{k:v for k,v in x.items() if k not in ['all_hits','candidate_kernel_args']} for x in graphs[0]['leaves']]},indent=2))
print(json.dumps(installed,indent=2))
