from pathlib import Path
from collections import Counter,defaultdict
from datetime import datetime,timezone
import json,hashlib,socket

assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
P=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/hostprof-20261030T212812Z-360e0c72/analysis')
E=P/'native-execution-identity-v14';C=P/'native-communications-v15';M=P/'native-PMU-complete-v18'
OUT=M/'H07-external-embedding-copy-and-exact-replay-chains-v26-v2.json';assert not OUT.exists()
def pin(p,sha):
 b=p.read_bytes();assert hashlib.sha256(b).hexdigest()==sha;return json.loads(b)
prep=pin(P/'prepare-field-origins-v12.json','c23b75f3fbd3ae3d3d7152f6882524e899bade53001ea2f1befa09ce943f091f')
gr=pin(E/'graph-blocks-PMU-and-MC2-v14.json','a035c4de08191656141ad0727e8305a6444cc77e6184d4951f86136021af0991')
cr=pin(C/'communication-and-queue-review-v15.json','c196d5083904a75a794c81fadfaf9d528d11461fdd5d81a6e941cad1e9693d6c')
cf=pin(C/'communication-fullchains-v15.json','2781a2a848883e6dee25ce4af368017dbc09627f4c2adc57ada9b37305360800')
fs=pin(E/'native-compute-fullchains-v14-v2.json','6c7bf4bd3a7dcf557327aeadc4a37ec4b9b23cb1ba675353bbd6e5d56a4e1ca0')
ix=pin(E/'index.json','a3aa47cdf9170785432ecbc18117a489f3fba37d89959f110f4fcbd8890c7dc4')
roots={r['root_key']:r for r in prep['prepare_roots']};bydb=defaultdict(list)
for r in roots.values():bydb[r['database_index']].append(r)
markers=defaultdict(list)
for b in prep['bindings']:markers[b['database_index']].append(b['marker'])
def frame_text(row):
 fs=row.get('project_frames',[])
 return '\n'.join(f.get('frame','')if isinstance(f,dict)else f for f in fs)
def root_candidates(di,a):
 rr=[r['root_key']for r in bydb[di]if r['root']['globalTid']==a['globalTid']and r['root']['startNs']<=a['startNs']and a['endNs']<=r['root']['endNs']]
 mm=[m for m in markers[di]if m['globalTid']==a['globalTid']and m['startNs']<=a['startNs']and a['endNs']<=m['endNs']]
 er=sorted({r['root_key']for m in mm for r in bydb[di]if r['root']['globalTid']==m['globalTid']and m['startNs']<=r['root']['startNs']and r['root']['endNs']<=m['endNs']})
 return dict(direct_prepare_roots=rr,same_execute_marker_prepare_roots=er)
opmap={(r['database_index'],r['COMM_OP_rowid']):r for r in cr['communication_operations']}
external=[];unbound_external=[]
chain='3fdf709ac228baf24319b0ec781c7e5ab0642ecf633471391315087fbebbabaa'
for db in cf['databases']:
 di=db['database_index']
 for op in db['records']:
  candidates=[a for g in op['source_groups']for a in g['candidates']if a.get('chain_key')==chain]
  if not candidates:continue
  links=[dict(api=a,**root_candidates(di,a))for a in candidates]
  rr=sorted({k for a in links for k in a['same_execute_marker_prepare_roots']})
  row=dict(database_index=di,communication=opmap[(di,op['COMM_OP_rowid'])],source_candidates=links,prepare_root_candidates=rr)
  if len(rr)!=1:unbound_external.append(row)
  external.append(row)
assert len(external)==88
copyrows=[r for r in prep['selected_API_records']if r['name']=='aten::copy_'and'gpu_model_runner.py(3582)'in frame_text(r)]
copy_by_root=defaultdict(list)
for r in copyrows:copy_by_root[r['root_key']].append(r)
native_apis=[];native_chains={};clamps=[]
for db in fs['databases']:
 di=db['database_index']
 for a in db['API_records']:
  text=frame_text(a)
  if 'vocab_parallel_embedding.py'in text and'gpu_model_runner.py(3575)'in text:
   native_apis.append(dict(database_index=di,API=a,root_links=root_candidates(di,a)))
   if a.get('chain_key'):native_chains[a['chain_key']]=fs['canonical_fullchains'][a['chain_key']]
  if 'gpu_model_runner.py(3532)'in text:clamps.append(dict(database_index=di,API=a,root_links=root_candidates(di,a)))
api_keys={(r['database_index'],r['API']['api_rowid'])for r in native_apis}
copy_keys={(r['database_index'],r['api_rowid'])for r in copyrows}
taskrows=[];copytasks=[];graphs=[];link_field_states=Counter()
bygraph=defaultdict(list)
for g in gr['graph_executions']:bygraph[g['database_index']].append(g)
for ref in ix['databases']:
 di=ref['database_index'];d=pin(Path(ref['path']),ref['sha256']);defs={c['globalTaskId']:c for c in d['COMPUTE_definitions']}
 for t in d['TASK_records']:
  link_field_states['PRESENT'if'native_linked_API_rowids'in t else'ABSENT_UNLINKED']+=1
  kk={(di,k)for k in t.get('native_linked_API_rowids',[])}
  if kk&api_keys:taskrows.append(dict(database_index=di,TASK=t,definition=defs.get(t['globalTaskId']),linked_selected_APIs=sorted(k[1]for k in kk&api_keys)))
  if kk&copy_keys:copytasks.append(dict(database_index=di,TASK=t,definition=defs.get(t['globalTaskId']),linked_copy_APIs=sorted(k[1]for k in kk&copy_keys)))
 for g in bygraph[di]:
  start=g['blocks'][0]['startNs'];prefix=sorted([r for r in g['outside_marker_tasks']if r['task']['endNs']<=start],key=lambda r:r['task']['startNs'])
  comm=[r for r in cr['communication_operations']if r['database_index']==di and len(r['graph_candidates'])==1 and r['graph_candidates'][0]['graph_ordinal']==g['graph_ordinal']and r['endNs']<=start]
  graphs.append(dict(database_index=di,graph_ordinal=g['graph_ordinal'],CANN_API=g['CANN_API'],prepare_root_keys=g['prepare_root_keys'],phase_records=g['phase_records'],prefix_compute=prefix,prefix_communication=comm))
rootgraphs=defaultdict(list)
for g in graphs:
 if len(g['prepare_root_keys'])==1:rootgraphs[g['prepare_root_keys'][0]].append(g)
external_byroot=defaultdict(list)
for e in external:
 if len(e['prepare_root_candidates'])==1:external_byroot[e['prepare_root_candidates'][0]].append(e)
rows=[];regimes=Counter();cohorts=defaultdict(list)
for key,r in roots.items():
 meta=r.get('metadata')or{};gs=rootgraphs[key];es=external_byroot[key];copies=copy_by_root[key]
 row=dict(root_key=key,database_index=r['database_index'],metadata=meta,prepare_root=r['root'],graphs=gs,external_embedding=es,persistent_embedding_copy_APIs=copies)
 scheduled=meta.get('total_scheduled_tokens');regimes[(scheduled,len(gs),len(es),len(copies))]+=1;rows.append(row)
 for g in gs:
  if len(g['phase_records'])==1 and g['phase_records'][0].get('schedule_position_cohort_key'):cohorts[g['phase_records'][0]['schedule_position_cohort_key']].append(dict(root_key=key,database_index=g['database_index'],graph_ordinal=g['graph_ordinal'],ep_rank=g['phase_records'][0]['ep_rank'],external_embeddings=len(es),copies=len(copies),external_comm_groups=[e['communication']['group_name']for e in es],scheduled_rows=scheduled))
summary=dict(created_utc=datetime.now(timezone.utc).isoformat(),TASK_native_link_field_states=dict(link_field_states),external_target_embedding_COMM_OPs=len(external),external_with_ambiguous_or_missing_root=len(unbound_external),persistent_embedding_copy_APIs=len(copyrows),external_embedding_native_API_records=len(native_apis),native_embedding_chain_count=len(native_chains),native_embedding_TASK_records=len(taskrows),copy_native_TASK_records=len(copytasks),input_clamp_native_API_records=len(clamps),graph_executions=len(graphs),graphs_bound_to_one_root=sum(len(g['prepare_root_keys'])==1 for g in graphs),prepare_roots=len(rows),root_regimes=[dict(scheduled_rows=k[0],graphs=k[1],external_embeddings=k[2],copies=k[3],roots=v)for k,v in regimes.items()],graph_prefix_compute_type_counts=dict(Counter(r['definition']['opType_text']for g in graphs for r in g['prefix_compute'])),graph_prefix_communication_regimes=[dict(family=k[0],dtype=k[1],count=k[2],rankSize=k[3],operations=v)for k,v in Counter((r['family'],r['dtype'],r['count'],r['rankSize'])for g in graphs for r in g['prefix_communication']).items()],exact_cohorts=len(cohorts),cohort_rank_sets=[sorted(v['ep_rank']for v in vv)for vv in cohorts.values()],no_new_source_DB_model_device_or_service_work=True,overall_complete=False,
 limits=['TASKrowswithoutthesavednative-APIlinkfieldstayunlinked;missingfielddoesnotmeanabsentwork. Nativeflow andsame-thread execute-marker containment are retained;no nearest-time orordinalpairing.','Graph-onlyprefix roles are structural/source-consistent,notfabricatednativePythonstacks,tensorpointers orbinaryidentity.','Counts,logicalrowdomains andlocalorder are notcausal latencygain.','ExactobservedTPcohorts donotprovefutureentry-hit predicates aregroup-uniform.','Raw-ID/inputs_embeds consumerandpurity questions requireseparatesourcecontract review.'])
result=dict(summary=summary,external_embedding=external,unbound_external=unbound_external,native_embedding_APIs=native_apis,native_embedding_canonical_chains=native_chains,native_embedding_tasks=taskrows,persistent_embedding_copy_APIs=copyrows,copy_native_tasks=copytasks,input_clamp_APIs=clamps,graph_prefixes=graphs,prepare_root_embedding_replay_rows=rows,exact_TP_cohorts=[dict(key=k,members=v)for k,v in cohorts.items()])
b=(json.dumps(result,ensure_ascii=False,separators=(',',':'))+'\n').encode();OUT.write_bytes(b)
public=dict(summary=summary,native_chain_headers=[dict(chain_key=k,project_frames=[r['frame']for r in fs if '/vllm'in r['frame']])for k,fs in native_chains.items()],first_root_with_graph=next((r for r in rows if r['graphs']),None),first_graph_prefix=graphs[:1],full_result=str(OUT),result_sha256=hashlib.sha256(b).hexdigest())
print(json.dumps(public,ensure_ascii=False))
