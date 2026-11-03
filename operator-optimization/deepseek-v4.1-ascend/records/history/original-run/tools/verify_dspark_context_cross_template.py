#!/usr/bin/env python3
"""Independent read-only checks of DSpark per-layer context fragment join."""
import hashlib,json
from collections import defaultdict
from pathlib import Path
r=Path(__file__).resolve().parents[1];p=r/'graph-revision-aux-context-v2';o=r/'graph-revision-cross-template-v1'
h=lambda f:hashlib.sha256(f.read_bytes()).hexdigest();load=lambda f:json.loads(f.read_text())
m=load(o/'manifest.json');parent=load(p/'manifest.json');g=load(o/'graph.json');idx=load(o/'index.json');views=load(o/'views.json')
assert h(p/'manifest.json')==m['inputs']['parent_manifest_sha256']
assert h(p/'G5.json')==m['inputs']['parent_g5_sha256']==parent['outputs']['G5.json']
for f,v in m['outputs'].items():assert h(o/f)==v
for f,v in parent['outputs'].items():assert h(p/f)==v
for f,v in parent['old_inputs'].items():assert h(r/f)==v
assert h(o/'graph.json')==idx['graph_sha256']==views['graph_sha256']
src=load(p/'G5.json');t={x['id']:x for x in src['templates']};ab={'DSPARK_BOUNDARIES':'B','DSPARK_SHARED_KV':'K'}
ns={n['id']:n for n in g['nodes']};assert len(ns)==len(g['nodes'])
for name,tag in ab.items():
 for n in t[name]['nodes']:
  z=ns[tag+'::'+n['id']];assert z['template']==name and z['original_node_id']==n['id']
  for k,v in n.items():
   if k!='id':assert z[k]==v
 for i,e in enumerate(t[name]['edges']):
  zs=[x for x in g['edges'] if x.get('original_template')==name and x.get('original_edge_index')==i];assert len(zs)==1;z=zs[0]
  for k,v in e.items():assert z[k]==(tag+'::'+v if k in ('producer','consumer') else v)
for e in g['edges']:assert e['producer'] in ns and e['consumer'] in ns
pairs=[('B::context_buffer','K::context'),('B::context_positions','K::context_positions'),('B::context_slots','K::slots')]
cross=g['cross_bindings'];assert len(cross)==3
assert [(g['edges'][i]['producer'],g['edges'][i]['consumer']) for i in cross]==pairs
for i in cross:
 z=g['edges'][i];assert z['is_new_operator'] is False and z['runtime_storage_or_event_identity']=='UNKNOWN' and 'Same invocation' in z['condition']
# Re-enumerate connected sets independently, excluding parent composite in connectivity only.
adj=defaultdict(set);incident=defaultdict(set)
for i,e in enumerate(g['edges']):
 a,b=e['producer'],e['consumer'];incident[a].add(i);incident[b].add(i)
 if 'B::kv_precompute' not in (a,b) and e.get('dependency_kind') not in ('NEXT_INVOCATION_RAW','SHAPE_DEVICE_METADATA'):
  adj[a].add(b);adj[b].add(a)
front={frozenset([n]) for n in ns if n!='B::kv_precompute'};want=set()
for size in (2,3,4):
 front={s|{v} for s in front for n in s for v in adj[n] if v not in s}
 want.update(s for s in front if any(g['edges'][i]['producer'] in s and g['edges'][i]['consumer'] in s for i in cross))
assert len(want)==idx['count']==len(idx['items'])
assert {frozenset(x['nodes']) for x in idx['items']}==want
for z in idx['items']:
 s=set(z['nodes']);rr={'internal':[],'external_inputs':[],'external_outputs':[]}
 for i in sorted(set().union(*(incident[n] for n in s))):
  e=g['edges'][i];a=e['producer'] in s;b=e['consumer'] in s
  rr['internal' if a and b else 'external_outputs' if a else 'external_inputs'].append(i)
 assert z['edge_indices']==rr and z['cross_binding_indices']==[i for i in cross if i in rr['internal']]
 assert z['classification']=='STRUCTURAL_ONLY_NOT_A_FUSION'
for fid,z in views['views'].items():
 source=load(p/(fid+'-boundary.json'));assert h(p/(fid+'-boundary.json'))==z['source_boundary_sha256'];tag=ab[source['template']]
 s={tag+'::'+x['id'] for x in source['nodes']};assert s==set(z['selected_nodes'])
 for key,fn in {'internal':lambda a,b:a and b,'external_inputs':lambda a,b:not a and b,'external_outputs':lambda a,b:a and not b}.items():
  assert z['edge_indices'][key]==[i for i,e in enumerate(g['edges']) if fn(e['producer'] in s,e['consumer'] in s)]
 assert z['cross_binding_external_indices']==[i for i in z['edge_indices']['external_inputs']+z['edge_indices']['external_outputs'] if i in cross]
assert views['views']['F29']['cross_binding_external_indices']==[cross[2]]
for line in views['lineages']:
 for a,b in zip(line['nodes'],line['nodes'][1:]):assert any(e['producer']==a and e['consumer']==b for e in g['edges'])
print(json.dumps({'status':'PASS','source_template_edges_checked':sum(len(t[x]['edges']) for x in ab),'cross_bindings':len(cross),'structural_sets':len(want),'scope':'reference/source-conditional graph only; physical identities, reachability, feasibility and performance UNKNOWN'},indent=2))
