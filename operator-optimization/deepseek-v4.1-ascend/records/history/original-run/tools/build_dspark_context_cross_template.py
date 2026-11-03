#!/usr/bin/env python3
"""Create a source-conditional, per-layer DSpark fragment join; no device access."""
import collections,copy,hashlib,json,os
from pathlib import Path
R=Path(__file__).resolve().parents[1];P=R/'graph-revision-aux-context-v2';O=R/'graph-revision-cross-template-v1'
h=lambda b:hashlib.sha256(b).hexdigest()
e=lambda x:(json.dumps(x,ensure_ascii=False,indent=2)+'\n').encode()
assert h((P/'manifest.json').read_bytes())=='b0064060152d35dec3013e6ff8f4da74ad6e48a9d42a160eafed4b6fada035b5'
m=json.loads((P/'manifest.json').read_bytes()); assert h((P/'G5.json').read_bytes())==m['outputs']['G5.json']
for path,sha in m['old_inputs'].items():assert h((R/path).read_bytes())==sha,path
for path,sha in m['outputs'].items():assert h((P/path).read_bytes())==sha,path
src=json.loads((P/'G5.json').read_bytes());ts={x['id']:x for x in src['templates']}
assert 'DSPARK_BOUNDARIES' in ts and 'DSPARK_SHARED_KV' in ts
aliases={'DSPARK_BOUNDARIES':'B','DSPARK_SHARED_KV':'K'};nodes=[];edges=[]
for tid,ab in aliases.items():
 t=ts[tid]
 for n in t['nodes']:
  z=copy.deepcopy(n);z['id']=ab+'::'+n['id'];z['template']=tid;z['original_node_id']=n['id'];nodes.append(z)
 for i,x in enumerate(t['edges']):
  z=copy.deepcopy(x);z['producer']=ab+'::'+x['producer'];z['consumer']=ab+'::'+x['consumer'];z['original_template']=tid;z['original_edge_index']=i;z['edge_role']='TEMPLATE_SOURCE_EDGE';edges.append(z)
assert len({n['id'] for n in nodes})==len(nodes)
condition='Same invocation, K>0, draft group exists, selected layer belongs to the per-layer group mapping, num_context>0 and valid supplied slot list; tensor/stream/event identities not historically bound.'
bindings=[
 ('B::context_buffer','K::context','context states slice [:num_context]','spec_decode/dspark_proposer.py:257-259; spec_decode/dflash_proposer.py:301-319; models/deepseek_v41/dspark.py:295-308','CONTEXT_HIDDEN_VIEW'),
 ('B::context_positions','K::context_positions','context position slice [:num_context]','spec_decode/dspark_proposer.py:274-285; spec_decode/dflash_proposer.py:301-318; models/deepseek_v41/dspark.py:55-68,284-308','CONTEXT_POSITION_VIEW'),
 ('B::context_slots','K::slots','group slot buffer selected for the same layer and sliced [:num_context]','spec_decode/dspark_proposer.py:280-310; spec_decode/dflash_proposer.py:301-319; models/deepseek_v41/dspark.py:295-309','PER_LAYER_GROUP_SLOT_VIEW')]
for a,b,v,s,k in bindings:
 edges.append({'producer':a,'consumer':b,'value':v,'source':s,'condition':condition,'dependency_kind':'CONDITIONAL_VIEW_OR_VALUE_BINDING','edge_role':'CROSS_TEMPLATE_SOURCE_BINDING','semantic_class':k,'is_new_operator':False,'runtime_storage_or_event_identity':'UNKNOWN'})
n={z['id']:z for z in nodes};assert len(n)==len(nodes)
for z in edges:assert z['producer'] in n and z['consumer'] in n
cross=[i for i,z in enumerate(edges) if z['edge_role']=='CROSS_TEMPLATE_SOURCE_BINDING'];assert len(cross)==3
# kv_precompute is the enclosing composite of K fragment, not an extra operator.
excluded={'B::kv_precompute'}
ad=collections.defaultdict(set)
for z in edges:
 a,b=z['producer'],z['consumer']
 if a in excluded or b in excluded:continue
 if z.get('dependency_kind') in ('NEXT_INVOCATION_RAW','SHAPE_DEVICE_METADATA'):continue
 ad[a].add(b);ad[b].add(a)
front={frozenset([x]) for x in n if x not in excluded};selected=[];stats={}
for size in (2,3,4):
 front={s|{v} for s in front for x in s for v in ad[x] if v not in s}
 yes={s for s in front if any(edges[j]['producer'] in s and edges[j]['consumer'] in s for j in cross)}
 stats[str(size)]={'connected_all':len(front),'containing_cross_binding':len(yes)}
 selected.extend(sorted(yes,key=lambda s:tuple(sorted(s))))
items=[]
for s in selected:
 rr={'internal':[],'external_inputs':[],'external_outputs':[]}
 for i,z in enumerate(edges):
  a=z['producer'] in s;b=z['consumer'] in s
  if a and b:rr['internal'].append(i)
  elif b:rr['external_inputs'].append(i)
  elif a:rr['external_outputs'].append(i)
 ids=sorted(s);c=[j for j in cross if j in rr['internal']]
 items.append({'id':'CROSS_STRUCTURE_'+h(json.dumps(ids).encode())[:16],'nodes':ids,'edge_indices':rr,'cross_binding_indices':c,'metadata_or_view_nodes':[x for x in ids if ('METADATA' in n[x]['kind'] or 'EXTERNAL' in n[x]['kind'])], 'classification':'STRUCTURAL_ONLY_NOT_A_FUSION','feasibility_and_performance':'UNKNOWN'})
# Propagate exact cross-fragment inputs/outputs of previously frozen candidate selections.
views={}
for fid in ('F27','F28','F29','F30'):
 v=json.loads((P/(fid+'-boundary.json')).read_bytes());ab=aliases[v['template']];s={ab+'::'+x['id'] for x in v['nodes']}
 assert s<=n.keys()
 rr={'internal':[],'external_inputs':[],'external_outputs':[]}
 for i,z in enumerate(edges):
  a=z['producer'] in s;b=z['consumer'] in s
  if a and b:rr['internal'].append(i)
  elif b:rr['external_inputs'].append(i)
  elif a:rr['external_outputs'].append(i)
 views[fid]={'selected_nodes':sorted(s),'source_boundary_path':str((P/(fid+'-boundary.json')).relative_to(R)),'source_boundary_sha256':m['outputs'][fid+'-boundary.json'],'edge_indices':rr,'cross_binding_external_indices':[i for i in rr['external_inputs']+rr['external_outputs'] if i in cross],'status':'PARTIAL_VIEW_NOT_COMPLETE_CONSUMERS'}
assert views['F29']['cross_binding_external_indices']==[cross[2]]
assert not views['F27']['cross_binding_external_indices'] and not views['F28']['cross_binding_external_indices']
graph={'id':'DSPARK_CONTEXT_TWO_TEMPLATE_CONDITIONAL_JOIN_V1','source_G5_sha256':m['outputs']['G5.json'],'nodes':nodes,'edges':edges,'cross_bindings':cross,'hierarchy_exclusion':['B::kv_precompute'],'scope':'One symbolic draft layer in one invocation; source-level correspondence is conditional, not runtime tensor alias or a whole-model graph.','limitations':['B::kv_precompute is an enclosing aggregate of the K fragment; excluded from combination connectivity to avoid hierarchical double counting, but retained as an external reference.','All slot mappings refer to the same layer-group selection, not aggregate equality across groups or three layer caches.','View edges are not operators, not established zero-copy, and not proof of device stream readiness.','Source branches and tensors may not match historical runtime.']}
gb=e(graph);gh=h(gb)
index={'id':'DSPARK_CROSS_TEMPLATE_CONNECTED_SETS_V1','graph_sha256':gh,'scope':'All connected 2-4 node sets with at least one explicit cross-template view/value binding and excluding the enclosing kv_precompute node; includes metadata and state. Not eligible fusion count.','size_coverage':stats,'count':len(items),'items':items,'limitations':graph['limitations']}
view={'id':'DSPARK_CROSS_TEMPLATE_BOUNDARY_VIEWS_V1','graph_sha256':gh,'views':views,'lineages':[{'description':'context states','nodes':['B::main_norm','B::context_buffer','K::context','K::proj']},{'description':'context positions','nodes':['B::input_expand','B::context_positions','K::context_positions','K::rope']},{'description':'layer-group slots','nodes':['B::input_expand','B::slot_outputs','B::context_slots','K::slots','K::scatter']}], 'identity_limit':'Lineages are source-level conditional values and named slice/group mappings; physical identity and stream/event visibility unknown.'}
out={'graph.json':gb,'index.json':e(index),'views.json':e(view)}
report={'id':'DSPARK_CROSS_TEMPLATE_BINDING_V1','inputs':{'parent_manifest_sha256':h((P/'manifest.json').read_bytes()),'parent_g5_sha256':m['outputs']['G5.json'],'parent_f27_f28_f29_f30_sha256':{fid:m['outputs'][fid+'-boundary.json'] for fid in views}},'outputs':{x:h(v) for x,v in out.items()},'checks':{'template_edges_preserved':len(edges)-3,'cross_bindings':3,'structural_set_count':len(items),'size_coverage':stats,'candidate_external_cross_bindings':{fid:v['cross_binding_external_indices'] for fid,v in views.items()}},'limitations':graph['limitations']}
out['manifest.json']=e(report)
O.mkdir(parents=True,exist_ok=True)
for x,b in out.items():
 if (O/x).exists() and (O/x).read_bytes()!=b:raise ValueError('refuse overwrite '+x)
for x,b in out.items():
 if not (O/x).exists():
  with (O/x).open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
for x,s in m['outputs'].items():assert h((P/x).read_bytes())==s
print(json.dumps({'output':str(O),'manifest_sha256':h(out['manifest.json']),'checks':report['checks']},indent=2))
