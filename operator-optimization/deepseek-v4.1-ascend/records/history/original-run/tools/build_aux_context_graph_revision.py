#!/usr/bin/env python3
"""Offline source-conditional graph repair. Create-only; preserves old artifacts.

Builds a fallback-only variant (not a universal runtime graph) and repairs
selected G5 boundaries, then rebuilds all affected structural references.
"""
import collections, copy, hashlib, itertools, json, os
from pathlib import Path
R=Path(__file__).resolve().parents[1]
B=R/'graph-revision-aux-context-v2'
expected={
 'graphs/G1-conditional-materialized-v1.json':'b6ddd6dcc580fa4133dfa1ff937f531e20611c66c6b8567dfffaec9c952e4c44',
 'graphs/source-templates-G5.json':'a3a4e82afb8fed8894300148791a22ed1ca954288a0ed8fd79ef9a3879c56731',
 'evidence/model-config.json':'43d3b2bc55626893d59275b32e04068c0fc916f5c382eb91de73bdbade9c88f7',
 'agents/bounds/metadata-path-snapshot/ascend_runner.py':'568c3049205b9caff5c656a7f5dd4c6e79fb358abe526f89fb9fb3f70a60a43c',
 'agents/bounds/metadata-path-snapshot/vllm_runner.py':'87c29d08c0bbf66993d8b984811e7325e35e6436242a773feb55ec88f16b2c51',
 'agents/fusion/f27-aux-setter-evidence.json':'3dd283959f281dfc3c4c25f43f0546ab610f9950d12d9c90792bff0b29af71f4',
 'sources/vllm-workspace/vllm-ascend/vllm_ascend/models/deepseek_v41/model.py':'101f96b38e9fc47e7be147ddfe56976aee896867c73a33a7d95c3e3c833a6e91',
 'sources/vllm-workspace/vllm-ascend/vllm_ascend/models/deepseek_v41/dspark.py':'fcee256905928917154d8dac94d6611bb83d02da441e8aa154a525ef51e6a601',
 'sources/vllm-workspace/vllm-ascend/vllm_ascend/spec_decode/dspark_proposer.py':'70d216de21875e679d510153ebdb0f0ccf130875a0e644f0673d60b67b6b633c',
 'sources/vllm-workspace/vllm-ascend/vllm_ascend/spec_decode/dflash_proposer.py':'f3b736f44a59ec0831aa7f2849e2eae6d709951e955992b8016f49e772fce3ff',
 'agents/fusion/frozen-source/vllm_ascend/ops/triton/spec_decode/utils.py':'f3c121111d07f08d0de0498faa64529700dd6089ff7e46303eecf4947e516a95',
}
def sha(b): return hashlib.sha256(b).hexdigest()
def enc(x): return (json.dumps(x,ensure_ascii=False,indent=2)+'\n').encode()
def read(p): return json.loads((R/p).read_bytes())
for p,h in expected.items():
 if sha((R/p).read_bytes())!=h: raise ValueError('input hash mismatch '+p)
oldg=read('graphs/G1-conditional-materialized-v1.json'); oldidx=read('candidates/cross-module-G1-conditional-v1.json')
g5=read('graphs/source-templates-G5.json'); oldsi=read('candidates/dspark_boundaries-G5-structural-index.json'); records=read('records.json')
assert oldidx['derived_graph_sha256']==expected['graphs/G1-conditional-materialized-v1.json']
assert read('evidence/model-config.json')['text_config']['dspark_target_layer_ids']==[37,38,39]
assert len({n['id'] for n in oldg['nodes']})==len(oldg['nodes'])

def ad(nodes,edges):
 a=collections.defaultdict(set)
 for e in edges:
  if e.get('dependency_kind') in ('NEXT_INVOCATION_RAW','SHAPE_DEVICE_METADATA'): continue
  a[e['producer']].add(e['consumer']);a[e['consumer']].add(e['producer'])
 return a
def enum_g1(nodes,edges):
 a=ad(nodes,edges);front={frozenset([n]) for n in nodes};sets=[];sizes={}
 for size in (2,3,4):
  front={s|{v} for s in front for n in s for v in a[n] if v not in s}
  assert len(front)<150000
  sel={s for s in front if len({nodes[x].get('module') for x in s})>1}
  sizes[str(size)]={'all_connected_sets':len(front),'cross_module_sets':len(sel)}
  sets.extend(sorted(sel,key=lambda s:tuple(sorted(s))))
 return sets,sizes
def refs(s,edges):
 result={'internal':[],'external_inputs':[],'external_outputs':[]}
 for i,e in enumerate(edges):
  if e.get('dependency_kind')=='NEXT_INVOCATION_RAW':continue
  a=e['producer'] in s;b=e['consumer'] in s
  if a and b:result['internal'].append(i)
  elif b:result['external_inputs'].append(i)
  elif a:result['external_outputs'].append(i)
 return result
oldnodes={n['id']:n for n in oldg['nodes']};oldsets,_=enum_g1(oldnodes,oldg['edges']); olditems={frozenset(x['nodes']):x for x in oldidx['items']}
assert len(olditems)==len(oldidx['items']) and set(olditems)==set(oldsets)
# Existing v1 refs verified before accepting an index as input.
for s,item in olditems.items(): assert item['edge_indices_bound_to_derived_graph']==refs(s,oldg['edges'])
ng=copy.deepcopy(oldg); ng.update(id='G1_DSPARK_FALLBACK_AUX37_39_V2',parent_graph_sha256=expected['graphs/G1-conditional-materialized-v1.json'],complete=False)
ng['abstraction']='Conditional source graph: DSpark fallback aux [37,38,39]; not a runtime graph or all configuration paths'
ng['known_gaps'].extend(['Only DSpark fallback draft IDs [37,38,39], parent returned no override, setter active and all selected layers executed. Other branches remain unknown.', 'G5 fragments are not expanded into this graph.'])
removed='target.layers.36.aux'; new='target.layers.39.aux'; assert removed in oldnodes and new not in oldnodes
ng['nodes']=[n for n in ng['nodes'] if n['id']!=removed]
ng['nodes'].append({'id':new,'kind':'HC_MEAN_AUX','module':'target.layers.39','source':'models/deepseek_v41/model.py:1555-1564','condition':'DSpark fallback [37,38,39], no parent override; actual layers and SP not runtime-bound'})
mapold=[];newedges=[];deleted=[]
for i,e in enumerate(oldg['edges']):
 if removed in (e['producer'],e['consumer']):mapold.append(None);deleted.append(i)
 else:mapold.append(len(newedges));newedges.append(copy.deepcopy(e))
assert len(deleted)==2
newedges += [{'producer':'target.layers.38.hc_ffn_post','consumer':new,'value':'residual entering zero-based layer 39 under DSpark fallback','dependency_kind':'RAW','condition':'draft IDs [37,38,39], parent no override, layer executes'}, {'producer':new,'consumer':'dspark.main','value':'aux hidden from zero-based layer 39','dependency_kind':'RAW','condition':'draft IDs [37,38,39], parent no override, selected layers execute'}]
ng['edges']=newedges;nn={n['id']:n for n in ng['nodes']};assert len(nn)==len(ng['nodes'])
for e in newedges:assert e['producer'] in nn and e['consumer'] in nn
for i,j in enumerate(mapold):
 if j is not None:assert oldg['edges'][i]==newedges[j]
newsets,sizes=enum_g1(nn,newedges); newitems=[]
roles=('CONDITIONAL_CANDIDATE_BLOCK_IDS','SHAPE_ATTRIBUTE','INITIAL_PRE_MIX_VALUE')
for s in newsets:
 rr=refs(s,newedges);p=olditems.get(s)
 if p:
  for cat,indices in p['edge_indices_bound_to_derived_graph'].items():
   assert {mapold[i] for i in indices if mapold[i] is not None}<=set(rr[cat])
 classes={cat:{role:[i for i in indices if newedges[i].get('semantic_class')==role] for role in roles} for cat,indices in rr.items()}
 fp=sha(json.dumps(sorted(s)).encode())[:16]
 newitems.append({'id':'G1_FALLBACK_AUX37_39_'+fp,'nodes':sorted(s),'modules':sorted({str(nn[n].get('module')) for n in s}),'old_v1_item_id':None if p is None else p['id'],'edge_indices_bound_to_derived_graph':rr,'special_edge_indices_by_semantic_class':classes,'status':'STRUCTURAL_ONLY','semantic_equivalence_or_savings_proven':False})
# Rebuild G5 fragments, preserve every prior edge and classify added fields.
new5=copy.deepcopy(g5);new5.update(id='SOURCE_TEMPLATES_G5_AUX_CONTEXT_V2',parent_sha256=expected['graphs/source-templates-G5.json'],whole_model_complete=False)
t={x['id']:x for x in new5['templates']};d=t['DSPARK_BOUNDARIES'];k=t['DSPARK_SHARED_KV']
def edge(a,b,v,source,condition):return {'producer':a,'consumer':b,'value':v,'dependency_kind':'RAW','semantic_class':'CONDITIONAL_POSITION_OR_SLOT_METADATA','source':source,'condition':condition}
def add(tpl,node,edges):
 ids={n['id'] for n in tpl['nodes']};assert node['id'] not in ids;tpl['nodes'].append(node)
 for e in edges:
  assert not any(z['producer']==e['producer'] and z['consumer']==e['consumer'] for z in tpl['edges'])
  tpl['edges'].append(e)
add(d,{'id':'context_positions','kind':'PRODUCED_CONTEXT_POSITION_METADATA','resource':'NPU','shape':'[num_context]','dtype':'int32','condition':'input_expand executes for draft group; empty context separate'},[
 edge('input_expand','context_positions','copied target context positions','spec_decode/dspark_proposer.py:274-285; ops/triton/spec_decode/utils.py:111-120','draft group kernel executes'),
 edge('context_positions','kv_precompute','context positions sliced to num_context','spec_decode/dflash_proposer.py:301-319; models/deepseek_v41/dspark.py:295-309','nonempty context and slots supplied')])
d['edges'].append(edge('slot_outputs','context_slots','group context slot subdomain mapped to layers','spec_decode/dspark_proposer.py:280-310; spec_decode/dflash_proposer.py:301-319','draft group kernel executes; per-layer group mapping'))
add(k,{'id':'context_positions','kind':'EXTERNAL_CONTEXT_POSITION_METADATA','resource':'NPU','shape':'[num_context]','dtype':'int32'},[
 edge('context_positions','rope','context position for RoPE tables','models/deepseek_v41/dspark.py:55-68,284-309','nonempty context and slots supplied')])
for tpl in new5['templates']:
 ids={n['id'] for n in tpl['nodes']};assert len(ids)==len(tpl['nodes'])
 for e in tpl['edges']: assert e['producer'] in ids and e['consumer'] in ids
# Check all old G5 identities/edges survived exactly.
for old in g5['templates']:
 z=t[old['id']];assert z['nodes'][:len(old['nodes'])]==old['nodes']; assert z['edges'][:len(old['edges'])]==old['edges']

def structural(tpl):
 nodes={x['id']:x for x in tpl['nodes']}; eligible={k for k,v in nodes.items() if not v['kind'].startswith('EXTERNAL') and k not in {'confidence','dynamic_scheduler'}};adj=collections.defaultdict(set)
 for e in tpl['edges']:
  a,b=e['producer'],e['consumer']
  if a in eligible and b in eligible:adj[a].add(b);adj[b].add(a)
 sets={frozenset([k]) for k in eligible};out=[]
 for size in (2,3,4):
  sets={s|{x} for s in sets for x in set().union(*(adj[n] for n in s))-s}
  for s in sorted(sets,key=lambda v:tuple(sorted(v))):
   out.append({'nodes':sorted(s),'category':{2:'PAIR',3:'TRIPLE',4:'MULTI_OPERATOR'}[size], 'internal_edges':[z for z in tpl['edges'] if z['producer'] in s and z['consumer'] in s], 'external_inputs':[z for z in tpl['edges'] if z['consumer'] in s and z['producer'] not in s], 'external_outputs':[z for z in tpl['edges'] if z['producer'] in s and z['consumer'] not in s], 'status':'STRUCTURAL_ONLY','feasibility_and_semantic_equivalence':'UNKNOWN'})
 for src in nodes:
  users=sorted({z['consumer'] for z in tpl['edges'] if z['producer']==src and z['consumer'] in eligible})
  for a,b in itertools.combinations(users,2):out.append({'nodes':[a,b],'shared_input':src,'category':'BRANCH_MULTICONSUMER','status':'STRUCTURAL_ONLY','feasibility_and_semantic_equivalence':'UNKNOWN'})
 return out
oldtpl=next(x for x in g5['templates'] if x['id']=='DSPARK_BOUNDARIES')
assert oldsi['items']==structural(oldtpl) and oldsi['count']==len(oldsi['items'])
struct=structural(d);kstruct=structural(k)
recs={x['id']:x for x in records['fusion_candidates']};view={}
for fid,tid in [('F27','DSPARK_BOUNDARIES'),('F28','DSPARK_BOUNDARIES'),('F29','DSPARK_SHARED_KV'),('F30','DSPARK_BOUNDARIES')]:
 rec=recs[fid];oldpath=rec['candidate_graph_path'];assert sha((R/oldpath).read_bytes())==rec['candidate_graph_sha256']
 oldview=read(oldpath); selected={n['id'] for n in oldview['nodes']};assert selected==set(rec['original_node_ids'])
 tpl=t[tid];assert selected<={n['id'] for n in tpl['nodes']};es=tpl['edges']
 view[fid]={'id':fid+'_G5_AUX_CONTEXT_V2','template':tid,'old_candidate_path':oldpath,'old_candidate_sha256':rec['candidate_graph_sha256'],'nodes':[n for n in tpl['nodes'] if n['id'] in selected], 'internal_edges':[e for e in es if e['producer'] in selected and e['consumer'] in selected], 'external_inputs':[e for e in es if e['consumer'] in selected and e['producer'] not in selected], 'external_outputs':[e for e in es if e['producer'] in selected and e['consumer'] not in selected], 'status':'STRUCTURAL_ONLY','all_consumers_and_aliases':'UNKNOWN'}
 if fid!='F29':
  for name in ('internal_edges','external_inputs','external_outputs'): assert all(e in view[fid][name] for e in oldview[name])
assert any(e['producer']=='context_positions' and e['consumer']=='rope' for e in view['F29']['external_inputs'])
# Create all output bytes before touching destination.
gb=enc(ng);tb=enc(new5);gh=sha(gb);th=sha(tb)
oldnames=set(olditems);newnames={frozenset(x['nodes']) for x in newitems}
index={'id':'G1_FALLBACK_AUX37_39_STRUCTURAL_INDEX_V2','graph_sha256':gh,'parent_index_sha256':sha((R/'candidates/cross-module-G1-conditional-v1.json').read_bytes()),'scope':'Connected 2-4 node sets crossing module labels; NEXT_INVOCATION_RAW and SHAPE_DEVICE_METADATA excluded from adjacency, shape metadata retained as external/internal refs.','size_coverage':sizes,'count':len(newitems),'items':newitems,'limitations':['Source conditional fallback only; no runtime binding, feasible fusion or performance guarantee.','G5 not stitched into G1.']}
mp={'old_graph_sha256':expected['graphs/G1-conditional-materialized-v1.json'],'new_graph_sha256':gh,'old_edge_to_new_edge_index':mapold,'deleted_old_edge_indices':deleted,'deleted_node':removed,'added_node':new,'old_v1_sets_absent':len(oldnames-newnames),'new_v2_sets':len(newnames-oldnames)}
si={'id':'DSPARK_BOUNDARIES_STRUCTURAL_V2','template_graph_sha256':th,'template':'DSPARK_BOUNDARIES','count':len(struct),'items':struct,'limitations':['Metadata/control nodes may be enumerated as structure; not proof of fuseable arithmetic or same-branch reachability.']}
ki={'id':'DSPARK_SHARED_KV_STRUCTURAL_V2','template_graph_sha256':th,'template':'DSPARK_SHARED_KV','count':len(kstruct),'items':kstruct,'limitations':si['limitations']}
for x in view.values():x['template_graph_sha256']=th
out={'G1.json':gb,'G5.json':tb,'G1-index.json':enc(index),'G1-edge-map.json':enc(mp),'G5-dspark-index.json':enc(si),'G5-shared-kv-index.json':enc(ki)}
for fid,x in view.items():out[f'{fid}-boundary.json']=enc(x)
manifest={'id':'GRAPH_REVISION_AUX_CONTEXT_V2','old_inputs':{p:sha((R/p).read_bytes()) for p in sorted(set(expected)|{'candidates/cross-module-G1-conditional-v1.json','candidates/dspark_boundaries-G5-structural-index.json','records.json'}|{recs[f]['candidate_graph_path'] for f in view})},'outputs':{p:sha(b) for p,b in out.items()},'checks':{'v1_items_verified':len(olditems),'v2_items':len(newitems),'removed_edges':deleted,'old_sets_absent':len(oldnames-newnames),'new_sets':len(newnames-oldnames),'old_g5_items_verified':len(oldsi['items']),'new_g5_d_bounds_items':len(struct),'new_g5_shared_kv_items':len(kstruct),'old_inputs_unchanged':True},'template_links':[{'producer_template':'DSPARK_BOUNDARIES','producer_node':'context_positions','consumer_template':'DSPARK_SHARED_KV','consumer_node':'context_positions','condition':'same context generation and precompute, input ready, valid context and per-group slots'}, {'producer_template':'DSPARK_BOUNDARIES','producer_node':'context_slots','consumer_template':'DSPARK_SHARED_KV','consumer_node':'slots','condition':'same layer and group mapping, valid slots'}], 'limitations':['Fallback-only not universal resolved runtime.','Source fragments not complete; template_links are conditional references, not materialized monolithic graph.','Structural combinations not fusion, timing, reachability, numerical equivalence or complete consumers.','Old records and indices remain unchanged; adoption requires explicit consumer selection.']}
out['manifest.json']=enc(manifest)
B.mkdir(parents=True,exist_ok=True)
for p,b in out.items():
 f=B/p
 if f.exists() and f.read_bytes()!=b: raise ValueError('refusing overwrite '+str(f))
for p,b in out.items():
 f=B/p
 if not f.exists():
  with f.open('xb') as h:h.write(b);h.flush();os.fsync(h.fileno())
for p,h in manifest['old_inputs'].items():assert sha((R/p).read_bytes())==h
for p,h in manifest['outputs'].items():assert sha((B/p).read_bytes())==h
print(json.dumps({'bundle':str(B),'manifest_sha256':sha(out['manifest.json']),'checks':manifest['checks']},indent=2))
