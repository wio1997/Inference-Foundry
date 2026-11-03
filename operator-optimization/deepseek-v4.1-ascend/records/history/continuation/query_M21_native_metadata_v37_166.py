from pathlib import Path
from collections import Counter,defaultdict
from datetime import datetime,timezone
import hashlib,json,socket
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
P=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/hostprof-20261030T212812Z-360e0c72/analysis');E=P/'native-execution-identity-v14';C=P/'native-communications-v15';M=P/'native-PMU-complete-v18';OUT=M/'M21-allrank-input-and-attention-metadata-native-scope-v37.json';assert not OUT.exists()
def pin(p,sha):
 b=p.read_bytes();assert hashlib.sha256(b).hexdigest()==sha,str(p);return json.loads(b)
fs=pin(E/'native-compute-fullchains-v14-v2.json','6c7bf4bd3a7dcf557327aeadc4a37ec4b9b23cb1ba675353bbd6e5d56a4e1ca0');ix=pin(E/'index.json','a3aa47cdf9170785432ecbc18117a489f3fba37d89959f110f4fcbd8890c7dc4');cf=pin(C/'communication-fullchains-v15.json','2781a2a848883e6dee25ce4af368017dbc09627f4c2adc57ada9b37305360800');cr=pin(C/'communication-and-queue-review-v15.json','c196d5083904a75a794c81fadfaf9d528d11461fdd5d81a6e941cad1e9693d6c');prep=pin(P/'prepare-field-origins-v12.json','c23b75f3fbd3ae3d3d7152f6882524e899bade53001ea2f1befa09ce943f091f')
roots={r['root_key']:r for r in prep['prepare_roots']};bydb=defaultdict(list);markers=defaultdict(list)
for r in roots.values():bydb[r['database_index']].append(r)
for b in prep['bindings']:markers[b['database_index']].append(b['marker'])
def root_links(di,a):
 mm=[m for m in markers[di]if m['globalTid']==a['globalTid']and m['startNs']<=a['startNs']and a['endNs']<=m['endNs']]
 return sorted({r['root_key']for m in mm for r in bydb[di]if r['root']['globalTid']==m['globalTid']and m['startNs']<=r['root']['startNs']and r['root']['endNs']<=m['endNs']})
def relevant(frames):
 for frame in frames:
  function=frame.rsplit(': ',1)[-1]
  if 'metadata'in frame.lower():return True
  if '/spec_decode/'in frame and('prepare'in function or function=='set_inputs_first_pass'):return True
  if '/worker/model_runner_v1.py('in frame and any(q in function for q in['_prepare_inputs','_preprocess','_sync_metadata','_build_attention','_prepare_discard']):return True
  if '/attention/'in frame and function in['build','build_for_cudagraph_capture','build_for_drafting','build_for_proposing','build_for_step']:return True
 return False

def context(frames):
 text='\n'.join(frames)
 if '_get_prompt_logprobs_dict'in text:return'PROMPT_LOGPROBS'
 if any(x in text for x in['/model_runner_v1.py(2635)','/model_runner_v1.py(2646)']):return'TARGET_PRIMARY'
 if '_dummy_run'in text or ': dummy_run'in text:return'DUMMY_ANCESTRY'
 if any(x in text for x in['/llm_base_proposer.py','/dspark_proposer.py']):return'DRAFT_PROPOSAL_ANCESTRY'
 return'OTHER_RETAINED'
def shape(c):return dict(op=c['opType_text'],input=c.get('inputShapes_text'),output=c.get('outputShapes_text'),input_types=c.get('inputDataTypes_text'),output_types=c.get('outputDataTypes_text'))
def census(cs):return[dict(signature=json.loads(k),TASKs=v)for k,v in Counter(json.dumps(shape(c),sort_keys=True)for c in cs).items()]
selected={};apirows=[];chains=set();bychain=defaultdict(list)
for db in fs['databases']:
 di=db['database_index'];selected[di]={}
 for a in db['API_records']:
  if relevant(a['project_frames']):
   x=dict(database_index=di,API=a,context=context(a['project_frames']),prepare_root_candidates=root_links(di,a));apirows.append(x);selected[di][a['api_rowid']]=x;chains.add(a['chain_key'])
native=[];typed_only=[];allapi={db['database_index']:{a['api_rowid']:a for a in db['API_records']}for db in fs['databases']}
for ref in ix['databases']:
 di=ref['database_index'];d=pin(Path(ref['path']),ref['sha256']);cc={c['globalTaskId']:c for c in d['COMPUTE_definitions']}
 for t in d['TASK_records']:
  ids=sorted(set(t.get('native_linked_API_rowids',[]))&selected[di].keys())
  definition=cc.get(t['globalTaskId']);is_meta=definition is not None and 'metadata'in definition['opType_text'].lower()
  if not ids and is_meta:
   ids=sorted(set(t.get('native_linked_API_rowids',[]))&allapi[di].keys())
   for key in ids:
    a=allapi[di][key]
    if key not in selected[di]:
     x=dict(database_index=di,API=a,context=context(a['project_frames']),prepare_root_candidates=root_links(di,a),added_by='TYPED_METADATA_POSITIVE');selected[di][key]=x;apirows.append(x);chains.add(a['chain_key'])
   if not ids:typed_only.append(dict(database_index=di,TASK=t,definition=definition,grade='TYPED_METADATA_WITHOUT_NATIVE_FULLCALLER;ORIGIN_NOT_PROMOTED'))
  if not ids:continue
  contexts=sorted({selected[di][key]['context']for key in ids});x=dict(database_index=di,TASK=t,definition=cc.get(t['globalTaskId']),contexts=contexts,source_APIs=[{k:selected[di][key]['API'][k]for k in ['api_rowid','startNs','endNs','globalTid','chain_key']}for key in ids],prepare_root_candidates=sorted({r for key in ids for r in selected[di][key]['prepare_root_candidates']}));native.append(x)
  for key in{a['chain_key']for a in x['source_APIs']}:bychain[key].append(x)
opmap={(r['database_index'],r['COMM_OP_rowid']):r for r in cr['communication_operations']};comms=[]
for db in cf['databases']:
 for r in db['records']:
  hits=[dict(kind=g['kind'],API=a,context=context(a['project_frames']),prepare_root_candidates=root_links(db['database_index'],a))for g in r['source_groups']for a in g['candidates']if relevant(a['project_frames'])]
  if hits:comms.append(dict(database_index=db['database_index'],operation=opmap[(db['database_index'],r['COMM_OP_rowid'])],source_candidates=hits,contexts=sorted({x['context']for x in hits}),prepare_root_candidates=sorted({key for x in hits for key in x['prepare_root_candidates']})))
headers=[]
for key in sorted(chains):
 rr=bychain[key];headers.append(dict(chain_key=key,APIs=sum(r['API']['chain_key']==key for r in apirows),TASKs=len(rr),contexts=sorted({c for r in rr for c in r['contexts']}),typed_shapes=census([r['definition']for r in rr if r['definition']]),project_frames=[f['frame']for f in fs['canonical_fullchains'][key]if'/vllm'in f['frame']]))
summary=dict(created_utc=datetime.now(timezone.utc).isoformat(),native_APIs=len(apirows),native_TASKs=len(native),native_chains=len(headers),native_API_context_counts=dict(Counter(r['context']for r in apirows)),native_TASK_context_and_type=[dict(context=k[0],op=k[1],TASKs=v)for k,v in Counter(('|'.join(r['contexts']),r['definition']['opType_text']if r['definition']else'NO_COMPUTE_DEFINITION')for r in native).items()],native_shapes_by_context={ctx:census([r['definition']for r in native if r['definition']and r['contexts']==[ctx]])for ctx in sorted({r['context']for r in apirows})},source_bound_collectives=len(comms),collective_regimes=[dict(context=k[0],family=k[1],dtype=k[2],count=k[3],rankSize=k[4],operations=v)for k,v in Counter(('|'.join(r['contexts']),r['operation']['family'],r['operation']['dtype'],r['operation']['count'],r['operation']['rankSize'])for r in comms).items()],native_root_ambiguity_hist=dict(Counter(len(r['prepare_root_candidates'])for r in native)),collective_root_ambiguity_hist=dict(Counter(len(r['prepare_root_candidates'])for r in comms)),whole_module_closed=False,overall_complete=False)
summary['typed_metadata_without_native_fullcaller_TASKs']=len(typed_only);summary['metadata_type_counts']=dict(Counter(r['definition']['opType_text']for r in native+typed_only if r['definition']and 'metadata'in r['definition']['opType_text'].lower()));result=dict(summary=summary,typed_metadata_without_native_fullcaller=typed_only,native_APIs=apirows,native_TASKs=native,native_canonical_fullchains={k:fs['canonical_fullchains'][k]for k in chains},collectives=comms,limits=['Metadata/input-preparation ancestry selectors and positive typedMetadata fallback. Hierarchical overlap retained;notM21functionalPMUbucket. Function-name/type does not prove field semantics.','CompleteCOMM/sourcegroupsseparatefromComputeInfohcom/APIcounts;noadditivelatencyorwiretrafficclaim.','Onlyexactnativeflow/executecontainmentrootlinks;draft/dummyunboundrootsremainunassigned.','Noactualvocabpartition/padding/tie/NaN/alias/consumer/generationprooffromshapes/counts.','TargetgraphsuffixendsatnorminM15v31;thisnativeheadcensusdoesnotfabricategraph-onlyPythonstacks.','Nonewmodel/torch/NPU/service/healthwork.'])
b=(json.dumps(result,ensure_ascii=False,separators=(',',':'))+'\n').encode();OUT.write_bytes(b)
print(json.dumps(dict(summary=summary,native_chain_headers=headers,full_result=str(OUT),result_sha256=hashlib.sha256(b).hexdigest()),ensure_ascii=False))
