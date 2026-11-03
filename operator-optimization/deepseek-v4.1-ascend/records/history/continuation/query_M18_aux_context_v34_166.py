from pathlib import Path
from collections import Counter,defaultdict
from datetime import datetime,timezone
import hashlib,json,socket
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
P=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/hostprof-20261030T212812Z-360e0c72/analysis');E=P/'native-execution-identity-v14';C=P/'native-communications-v15';M=P/'native-PMU-complete-v18';OUT=M/'M18-allrank-aux-projection-context-KV-and-graph-checkpoint-scope-v34.json';assert not OUT.exists()
def pin(p,sha):
 b=p.read_bytes();assert hashlib.sha256(b).hexdigest()==sha,str(p);return json.loads(b)
fs=pin(E/'native-compute-fullchains-v14-v2.json','6c7bf4bd3a7dcf557327aeadc4a37ec4b9b23cb1ba675353bbd6e5d56a4e1ca0');ix=pin(E/'index.json','a3aa47cdf9170785432ecbc18117a489f3fba37d89959f110f4fcbd8890c7dc4');gr=pin(E/'graph-blocks-PMU-and-MC2-v14.json','a035c4de08191656141ad0727e8305a6444cc77e6184d4951f86136021af0991');cf=pin(C/'communication-fullchains-v15.json','2781a2a848883e6dee25ce4af368017dbc09627f4c2adc57ada9b37305360800');cr=pin(C/'communication-and-queue-review-v15.json','c196d5083904a75a794c81fadfaf9d528d11461fdd5d81a6e941cad1e9693d6c');prep=pin(P/'prepare-field-origins-v12.json','c23b75f3fbd3ae3d3d7152f6882524e899bade53001ea2f1befa09ce943f091f')
roots={r['root_key']:r for r in prep['prepare_roots']};bydb=defaultdict(list);markers=defaultdict(list)
for r in roots.values():bydb[r['database_index']].append(r)
for b in prep['bindings']:markers[b['database_index']].append(b['marker'])
def links(di,a):
 mm=[m for m in markers[di]if m['globalTid']==a['globalTid']and m['startNs']<=a['startNs']and a['endNs']<=m['endNs']]
 return sorted({r['root_key']for m in mm for r in bydb[di]if r['root']['globalTid']==m['globalTid']and m['startNs']<=r['root']['startNs']and r['root']['endNs']<=m['endNs']})
def role(frames):
 text='\n'.join(frames)
 if any(x in text for x in['/model.py(1561)','/model.py(1563)']):return'TARGET_AUX_MEAN_GATHER'
 if any(x in text for x in['/model_runner_v1.py(2069)','/model_runner_v1.py(2089)']):return'AUX_ROWSELECT_CONCAT'
 if ': combine_hidden_states'in text:return'F27_COMBINE'
 if ': _store_standard_swa_kv'in text and'/deepseek_v41/dspark.py'in text:return'F29_CONTEXT_STORE'
 if ': _project_shared_kv'in text and'/deepseek_v41/dspark.py'in text:return'F29_CONTEXT_PROJECT'
 return None
def phase(frames):
 text='\n'.join(frames)
 if any(x in text for x in['_dummy_run',': dummy_run',': profile_run']):return'DUMMY_ANCESTRY'
 if any(x in text for x in['/llm_base_proposer.py','/dspark_proposer.py','propose_draft_token_ids']):return'PROPOSAL_ANCESTRY'
 return'TARGET_OR_OTHER_NATIVE_ANCESTRY'
def shape(c):return dict(op=c['opType_text'],input=c.get('inputShapes_text'),output=c.get('outputShapes_text'),input_types=c.get('inputDataTypes_text'),output_types=c.get('outputDataTypes_text'))
def census(cs):return[dict(signature=json.loads(k),TASKs=v)for k,v in Counter(json.dumps(shape(c),sort_keys=True)for c in cs).items()]
selected={};apis=[];keys=set();bychain=defaultdict(list)
for db in fs['databases']:
 di=db['database_index'];selected[di]={}
 for a in db['API_records']:
  rr=role(a['project_frames'])
  if rr:
   z=dict(database_index=di,API=a,role=rr,phase=phase(a['project_frames']),prepare_root_candidates=links(di,a));apis.append(z);selected[di][a['api_rowid']]=z;keys.add(a['chain_key'])
native=[]
for ref in ix['databases']:
 di=ref['database_index'];d=pin(Path(ref['path']),ref['sha256']);cc={c['globalTaskId']:c for c in d['COMPUTE_definitions']}
 for t in d['TASK_records']:
  hits=sorted(set(t.get('native_linked_API_rowids',[]))&selected[di].keys())
  if not hits:continue
  aa=[selected[di][key]for key in hits];z=dict(database_index=di,TASK=t,definition=cc.get(t['globalTaskId']),roles=sorted({a['role']for a in aa}),phases=sorted({a['phase']for a in aa}),prepare_root_candidates=sorted({key for a in aa for key in a['prepare_root_candidates']}),source_APIs=[a['API']for a in aa]);native.append(z)
  for key in{a['API']['chain_key']for a in aa}:bychain[key].append(z)
opmap={(r['database_index'],r['COMM_OP_rowid']):r for r in cr['communication_operations']};comms=[]
for db in cf['databases']:
 for r in db['records']:
  hits=[dict(kind=g['kind'],API=a,role=role(a['project_frames']),phase=phase(a['project_frames']),prepare_root_candidates=links(db['database_index'],a))for g in r['source_groups']for a in g['candidates']if role(a['project_frames'])]
  if hits:comms.append(dict(database_index=db['database_index'],operation=opmap[(db['database_index'],r['COMM_OP_rowid'])],source_candidates=hits,roles=sorted({x['role']for x in hits}),phases=sorted({x['phase']for x in hits}),prepare_root_candidates=sorted({key for x in hits for key in x['prepare_root_candidates']})))
segments=[]
for g in gr['graph_executions']:
 for before in[37,38,39]:
  left=next(b for b in g['blocks']if b['block_ordinal']==before-1);right=next(b for b in g['blocks']if b['block_ordinal']==before)
  cc=sorted([r for r in g['outside_marker_tasks']if left['endNs']<=r['task']['startNs']and r['task']['endNs']<=right['startNs']],key=lambda r:(r['task']['startNs'],r['task']['task_record_rowid']))
  co=[r for r in cr['communication_operations']if r['database_index']==g['database_index']and left['endNs']<=r['startNs']and r['endNs']<=right['startNs']and any(x['graph_ordinal']==g['graph_ordinal']for x in r['graph_candidates'])]
  segments.append(dict(database_index=g['database_index'],graph_ordinal=g['graph_ordinal'],before_layer=before,prepare_root_keys=g['prepare_root_keys'],phase_records=g['phase_records'],compute=cc,COMM_candidates=co,grade='EXACT_MARKER_GAP;AUX_ROLE_SOURCE_CONSISTENT_NOT_NATIVE_STACK_ALIAS_PROOF'))
headers=[dict(chain_key=key,APIs=sum(a['API']['chain_key']==key for a in apis),TASKs=len(bychain[key]),roles=sorted({rr for r in bychain[key]for rr in r['roles']}),phases=sorted({rr for r in bychain[key]for rr in r['phases']}),typed_shapes=census([r['definition']for r in bychain[key]if r['definition']]),project_frames=[f['frame']for f in fs['canonical_fullchains'][key]if'/vllm'in f['frame']])for key in sorted(keys)]
summary=dict(created_utc=datetime.now(timezone.utc).isoformat(),native_APIs=len(apis),native_TASKs=len(native),native_chains=len(headers),native_role_phase_types=[dict(role=k[0],phase=k[1],op=k[2],TASKs=v)for k,v in Counter(('|'.join(r['roles']),'|'.join(r['phases']),r['definition']['opType_text']if r['definition']else'NO_COMPUTE_DEFINITION')for r in native).items()],source_bound_COMM=len(comms),collective_regimes=[dict(role=k[0],phase=k[1],dtype=k[2],count=k[3],rankSize=k[4],operations=v)for k,v in Counter(('|'.join(r['roles']),'|'.join(r['phases']),r['operation']['dtype'],r['operation']['count'],r['operation']['rankSize'])for r in comms).items()],graph_aux_segments=len(segments),graph_aux_compute_hist=dict(Counter(len(s['compute'])for s in segments)),graph_aux_types=dict(Counter(r['definition']['opType_text']for s in segments for r in s['compute'])),graph_aux_COMM_hist=dict(Counter(len(s['COMM_candidates'])for s in segments)),native_root_ambiguity_hist=dict(Counter(len(r['prepare_root_candidates'])for r in native)),whole_module_closed=False,overall_complete=False)
result=dict(summary=summary,native_APIs=apis,native_TASKs=native,native_canonical_fullchains={key:fs['canonical_fullchains'][key]for key in keys},collectives=comms,graph_aux_segments=segments,limits=['Selectorscoverauxpreparation/F27combine/F29contextKVnotallDSparkdecoder/headoperations;actualsourcecallerskept.','GraphgaprolesareSOURCE_CONSISTENT,notfabricatednativePythonstack/pointerproof.','COMM/ComputeInfo/APIhierarchyseparate;counts/logicalbytesnotcausalrequestgain.','Noorder/count/shapeisactualalias/weight/cache-request-generationproof;draft/dummyrootsnotnearest-timepaired.','Thisancestryscopeisnotautomaticreplacementforv20M18functionalPMUbucket.','Nonewsource/model/torch/NPU/service/healthwork.'])
b=(json.dumps(result,ensure_ascii=False,separators=(',',':'))+'\n').encode();OUT.write_bytes(b)
print(json.dumps(dict(summary=summary,native_chain_headers=headers,first_graph_segment=dict(before_layer=segments[0]['before_layer'],typed_shapes=census([r['definition']for r in segments[0]['compute']])),full_result=str(OUT),result_sha256=hashlib.sha256(b).hexdigest()),ensure_ascii=False))
