#!/usr/bin/env python3
"""Independent read-only validation of the conditional aux/context graph bundle."""
import hashlib,itertools,json
from collections import defaultdict
from pathlib import Path
r=Path(__file__).resolve().parents[1];p=r/'graph-revision-aux-context-v2'
get=lambda path:json.loads(path.read_text())
h=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
m=get(p/'manifest.json')
for x,s in m['old_inputs'].items():assert h(r/x)==s,('old hash',x)
for x,s in m['outputs'].items():assert h(p/x)==s,('new hash',x)
old=get(r/'graphs/G1-conditional-materialized-v1.json');g=get(p/'G1.json');mapping=get(p/'G1-edge-map.json');idx=get(p/'G1-index.json')
assert idx['graph_sha256']==mapping['new_graph_sha256']==h(p/'G1.json')
assert mapping['old_graph_sha256']==h(r/'graphs/G1-conditional-materialized-v1.json')
assert [i for i,j in enumerate(mapping['old_edge_to_new_edge_index']) if j is None]==mapping['deleted_old_edge_indices']
for i,j in enumerate(mapping['old_edge_to_new_edge_index']):
 if j is not None:assert old['edges'][i]==g['edges'][j]
ns={n['id']:n for n in g['nodes']};assert len(ns)==len(g['nodes'])
for e in g['edges']:assert e['producer'] in ns and e['consumer'] in ns
aux={n for n in ns if n.endswith('.aux')};assert aux=={f'target.layers.{i}.aux' for i in (37,38,39)}
for i in (37,38,39):
 assert any(e['producer']==f'target.layers.{i-1}.hc_ffn_post' and e['consumer']==f'target.layers.{i}.aux' for e in g['edges'])
 assert any(e['producer']==f'target.layers.{i}.aux' and e['consumer']=='dspark.main' for e in g['edges'])
# Construct indexes independently via an edge-to-incident map.
adj=defaultdict(set); incident=defaultdict(set)
for j,e in enumerate(g['edges']):
 a,b=e['producer'],e['consumer']
 if e['dependency_kind']=='NEXT_INVOCATION_RAW':continue
 incident[a].add(j);incident[b].add(j)
 if e['dependency_kind']!='SHAPE_DEVICE_METADATA':adj[a].add(b);adj[b].add(a)
front={frozenset([n]) for n in ns};want=set()
for size in (2,3,4):
 front={z|{v} for z in front for n in z for v in adj[n] if v not in z}
 want.update(z for z in front if len({ns[n].get('module') for n in z})>1)
assert len(idx['items'])==idx['count']==len(want)
assert {frozenset(z['nodes']) for z in idx['items']}==want
for z in idx['items']:
 s=set(z['nodes']);rr={'internal':[],'external_inputs':[],'external_outputs':[]}
 for j in sorted(set().union(*(incident[n] for n in s))):
  e=g['edges'][j];a=e['producer'] in s;b=e['consumer'] in s
  rr['internal' if a and b else 'external_outputs' if a else 'external_inputs'].append(j)
 assert rr==z['edge_indices_bound_to_derived_graph']
 for cat,roles in z['special_edge_indices_by_semantic_class'].items():
  for name,indices in roles.items():assert indices==[j for j in rr[cat] if g['edges'][j].get('semantic_class')==name]
new5=get(p/'G5.json');old5=get(r/'graphs/source-templates-G5.json');ts={t['id']:t for t in new5['templates']}
for t in old5['templates']:
 n=ts[t['id']];assert n['nodes'][:len(t['nodes'])]==t['nodes'];assert n['edges'][:len(t['edges'])]==t['edges']
for t in new5['templates']:
 ids={n['id'] for n in t['nodes']};assert len(ids)==len(t['nodes'])
 for e in t['edges']:assert {e['producer'],e['consumer']}<=ids
for tid,pairs in [('DSPARK_BOUNDARIES',[('input_expand','context_positions'),('context_positions','kv_precompute'),('slot_outputs','context_slots')]),('DSPARK_SHARED_KV',[('context_positions','rope')])]:
 for a,b in pairs:assert sum(e['producer']==a and e['consumer']==b for e in ts[tid]['edges'])==1
# Validate each exact excerpt, including the older F29/G2 reference.
recs={z['id']:z for z in get(r/'records.json')['fusion_candidates']}
for fid in ('F27','F28','F29','F30'):
 v=get(p/(fid+'-boundary.json'));assert v['template_graph_sha256']==h(p/'G5.json')
 s={n['id'] for n in v['nodes']};t=ts[v['template']]
 assert s==set(recs[fid]['original_node_ids'])
 for key,fn in {'internal_edges':lambda a,b:a and b,'external_inputs':lambda a,b:not a and b,'external_outputs':lambda a,b:a and not b}.items():
  assert v[key]==[e for e in t['edges'] if fn(e['producer'] in s,e['consumer'] in s)]
  assert all(e in v[key] for e in get(r/recs[fid]['candidate_graph_path'])[key])
# Independently check every structural item edge description, branch identity and set coverage.
for fname,tid in [('G5-dspark-index.json','DSPARK_BOUNDARIES'),('G5-shared-kv-index.json','DSPARK_SHARED_KV')]:
 doc=get(p/fname);t=ts[tid];nodes={n['id']:n for n in t['nodes']};eligible={n for n,v in nodes.items() if not v['kind'].startswith('EXTERNAL') and n not in {'confidence','dynamic_scheduler'}};ad=defaultdict(set)
 for e in t['edges']:
  if e['producer'] in eligible and e['consumer'] in eligible:ad[e['producer']].add(e['consumer']);ad[e['consumer']].add(e['producer'])
 f={frozenset([n]) for n in eligible};expected=[]
 for size in (2,3,4):
  f={s|{v} for s in f for u in s for v in ad[u] if v not in s}
  expected += [(str(size),tuple(sorted(s))) for s in f]
 for source in nodes:
  users=sorted({e['consumer'] for e in t['edges'] if e['producer']==source and e['consumer'] in eligible})
  expected += [('branch',tuple(pair),source) for pair in itertools.combinations(users,2)]
 seen=[]
 for z in doc['items']:
  s=set(z['nodes'])
  if z['category']=='BRANCH_MULTICONSUMER':seen.append(('branch',tuple(z['nodes']),z['shared_input']))
  else:
   seen.append((str(len(s)),tuple(z['nodes'])))
   for key,fn in {'internal_edges':lambda a,b:a and b,'external_inputs':lambda a,b:not a and b,'external_outputs':lambda a,b:a and not b}.items():
    assert z[key]==[e for e in t['edges'] if fn(e['producer'] in s,e['consumer'] in s)]
 assert len(seen)==len(expected)==doc['count'] and set(seen)==set(expected)
print(json.dumps({'status':'PASS','g1_node_sets':len(want),'g5_d_bounds':get(p/'G5-dspark-index.json')['count'],'g5_shared_kv':get(p/'G5-shared-kv-index.json')['count'],'f29_old_edges_all_preserved':True,'scope':'reference and structural checks only; not runtime identity, equivalence, exhaustive fusions or performance'},indent=2))
