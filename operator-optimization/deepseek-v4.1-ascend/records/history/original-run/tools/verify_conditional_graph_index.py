#!/usr/bin/env python3
"""Independently reconstruct every new graph reference; no writes or NPU use."""
import hashlib
import json
from pathlib import Path
p=Path(__file__).resolve().parents[1]
def load(name):return json.loads((p/name).read_text())
g=load('graphs/G1-conditional-materialized-v1.json')
idx=load('candidates/cross-module-G1-conditional-v1.json')
m=load('candidates/G1-conditional-old-new-edge-map-v1.json')
old=load('graphs/G1.json')
sha=hashlib.sha256((p/'graphs/G1-conditional-materialized-v1.json').read_bytes()).hexdigest()
assert idx['derived_graph_sha256']==m['new_graph_sha256']==sha
assert m['old_graph_sha256']==hashlib.sha256((p/'graphs/G1.json').read_bytes()).hexdigest()
assert m['old_edge_index_to_new_edge_index'][644] is None
for i,j in enumerate(m['old_edge_index_to_new_edge_index']):
 if j is not None: assert old['edges'][i]==g['edges'][j]
for x in idx['items']:
 s=set(x['nodes']);exp={'internal':[],'external_inputs':[],'external_outputs':[]}
 for i,e in enumerate(g['edges']):
  if e.get('dependency_kind')=='NEXT_INVOCATION_RAW':continue
  a=e['producer'] in s;b=e['consumer'] in s
  if a and b:exp['internal'].append(i)
  elif b:exp['external_inputs'].append(i)
  elif a:exp['external_outputs'].append(i)
 assert x['edge_indices_bound_to_derived_graph']==exp
 for cat,roles in x['special_edge_indices_by_semantic_class'].items():
  for role,ids in roles.items():assert ids==[i for i in exp[cat] if g['edges'][i].get('semantic_class')==role]
print(json.dumps({'status':'PASS','new_index_items':len(idx['items']),'new_graph_sha256':sha,'limitations':'Only reference consistency, not feasibility or performance.'}))
