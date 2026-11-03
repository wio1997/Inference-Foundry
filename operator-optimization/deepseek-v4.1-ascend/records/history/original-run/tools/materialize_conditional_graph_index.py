#!/usr/bin/env python3
"""Rebuild a versioned graph and edge-bound structural index entirely offline.

Inputs are immutable frozen artifacts. Outputs are create-only/idempotent and
never update old graph/index/STATE. This is not an optimization or proof of
feasibility. No NPU, remote service, or inference request is used.
"""
import collections
import copy
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INPUTS = {
 'graphs/G1.json': '9bece982c89be6320c4f3ef986ec33030a2e71327f0f0a96ae1159a19bb187cc',
 'graphs/G1-source-conditional-overlay.json': 'ac4b3138e8461443c5dda153553b9013c255d03cccc56bbd3b773115a4749e5f',
 'graphs/source-templates-G5.json': 'a3a4e82afb8fed8894300148791a22ed1ca954288a0ed8fd79ef9a3879c56731',
 'evidence/model-config.json': '43d3b2bc55626893d59275b32e04068c0fc916f5c382eb91de73bdbade9c88f7',
 'candidates/cross-module-G1.json': 'f375d77bc12836c4666c1123c5c432dd1f20211204a2a9db599c21546dc462fc',
 'sources/vllm-workspace/vllm-ascend/vllm_ascend/models/deepseek_v41/model.py': '101f96b38e9fc47e7be147ddfe56976aee896867c73a33a7d95c3e3c833a6e91',
 'sources/vllm-workspace/vllm-ascend/vllm_ascend/models/deepseek_v41/indexer.py': 'd5d6a438bbced4e63d729dac94c08869df30dab4a71ceacecdea211386b3db33',
 'sources/vllm-workspace/vllm-ascend/vllm_ascend/attention/dsa_v41.py': '3505769857f83fa7e67f652984f50bd0478158f1d24f185cab091203e70392fd',
}
def digest(b): return hashlib.sha256(b).hexdigest()
def raw(path):
 b = (ROOT/path).read_bytes()
 if digest(b) != INPUTS[path]: raise ValueError('input SHA mismatch: '+path)
 return b
def load(path): return json.loads(raw(path))
for key in INPUTS: raw(key)
g1 = load('graphs/G1.json'); overlay = load('graphs/G1-source-conditional-overlay.json'); old_index = load('candidates/cross-module-G1.json')
if overlay['parents']['G1.json'] != INPUTS['graphs/G1.json'] or old_index['G1_sha256'] != INPUTS['graphs/G1.json']: raise ValueError('parent mismatch')
if overlay['parents']['source-templates-G5.json'] != INPUTS['graphs/source-templates-G5.json'] or overlay['parents']['evidence/model-config.json'] != INPUTS['evidence/model-config.json']: raise ValueError('source/config parent mismatch')
if len(overlay['remove_g1_edges']) != 1 or g1['edges'].count(overlay['remove_g1_edges'][0]) != 1: raise ValueError('ambiguous removal')
removed = g1['edges'].index(overlay['remove_g1_edges'][0])
if removed != 644: raise ValueError('unexpected removal')
graph = copy.deepcopy(g1)
graph['id'] = 'G1_CONDITIONAL_MATERIALIZED_V1'
graph['parent_graph_id'] = g1['id']
graph['parent_graph_sha256'] = INPUTS['graphs/G1.json']
graph['overlay_sha256'] = INPUTS['graphs/G1-source-conditional-overlay.json']
graph['abstraction'] = 'Source conditional composite graph; edges are not runtime bindings or schedules'
graph['complete'] = False
graph['known_gaps'].extend(['G5 templates are not expanded or stitched into this materialized G1 graph.', 'shape metadata is external: source may use inputs_embeds and optional SP shard; runtime row count is unbound.', 'candidate publishing, row readiness and binary/runtime identity are not established by this graph.'])
if {n['id'] for n in graph['nodes']} & {n['id'] for n in overlay['add_g1_nodes']}: raise ValueError('node collision')
graph['nodes'].extend(copy.deepcopy(overlay['add_g1_nodes']))
new_edges=[]; mapping=[]
for i,e in enumerate(g1['edges']):
 if i == removed: mapping.append(None); continue
 mapping.append(len(new_edges)); new_edges.append(copy.deepcopy(e))
for edge in overlay['add_g1_edges']:
 e=copy.deepcopy(edge)
 if e['dependency_kind']=='SHAPE_DEVICE_METADATA':
  e['semantic_class']='SHAPE_ATTRIBUTE'
  e['source_abstraction_condition']='Actual source is hidden after optional SP shard; inputs_embeds may bypass embedding.'
 elif e.get('source_detail','').startswith('ATTN_C1_SOURCE.qli'):
  e['semantic_class']='CONDITIONAL_CANDIDATE_BLOCK_IDS'
  e['condition']='Same forward and matching active rows; nonempty QLI and candidate-filter branches. Empty query/cache and packed backend require separate branch contracts.'
  e['whole_producer_barrier_required']=None
  e['publication_granularity']='UNKNOWN; relevant row dependence is logical only; device release and barrier contracts not proved.'
 else: e['semantic_class']='INITIAL_PRE_MIX_VALUE'
 new_edges.append(e)
graph['edges']=new_edges
ns={n['id']:n for n in graph['nodes']}
if len(ns)!=len(graph['nodes']) or any(e['producer'] not in ns or e['consumer'] not in ns for e in new_edges): raise ValueError('invalid graph endpoints')
# The old positions are bound to old graph SHA, and each surviving mapping is exact.
for i,j in enumerate(mapping):
 if j is not None and g1['edges'][i]!=new_edges[j]: raise ValueError('mapping edge mismatch')
if sum(x is None for x in mapping)!=1: raise ValueError('unexpected deleted edges')

def adjacency(nodes, edges, include_shape=False):
 adj=collections.defaultdict(set)
 for e in edges:
  if e.get('dependency_kind')=='NEXT_INVOCATION_RAW': continue
  if not include_shape and e.get('dependency_kind')=='SHAPE_DEVICE_METADATA': continue
  a,b=e['producer'],e['consumer'];adj[a].add(b);adj[b].add(a)
 return adj

def enumerate_cross(nodes,adj):
 front={frozenset((n,)) for n in nodes};ret=[];counts={}
 for size in (2,3,4):
  front={s|{v} for s in front for n in s for v in adj[n] if v not in s}
  if len(front)>150000: raise ValueError('bounded enumeration cap exceeded')
  selected={s for s in front if len({nodes[x].get('module') for x in s})>1}
  counts[str(size)]={'all_connected_sets':len(front),'cross_module_sets':len(selected)}
  ret.extend(sorted(selected,key=lambda s:tuple(sorted(s))))
 return ret,counts
old_nodes={n['id']:n for n in g1['nodes']}
old_sets,old_sizes=enumerate_cross(old_nodes,adjacency(old_nodes,g1['edges']))
old_items={frozenset(x['nodes']):x for x in old_index['items']}
if len(old_items)!=22802 or set(old_sets)!=set(old_items): raise ValueError('old enumeration mismatch')

def edge_index(edges):
 inc=collections.defaultdict(set);out=collections.defaultdict(set)
 for i,e in enumerate(edges):
  if e.get('dependency_kind')=='NEXT_INVOCATION_RAW':continue
  out[e['producer']].add(i);inc[e['consumer']].add(i)
 return inc,out

def refs(s,edges,inc,out):
 incoming=set().union(*(inc[n] for n in s));outgoing=set().union(*(out[n] for n in s));return {'internal': sorted(incoming & outgoing),'external_inputs':sorted(incoming-outgoing),'external_outputs':sorted(outgoing-incoming)}
old_in,old_out=edge_index(g1['edges'])
fields={'internal':'internal_edges_G1_edge_indices','external_inputs':'external_inputs_G1_edge_indices','external_outputs':'external_outputs_G1_edge_indices'}
for s,item in old_items.items():
 rr=refs(s,g1['edges'],old_in,old_out)
 for k,v in rr.items():
  if sorted(item[fields[k]])!=v: raise ValueError('old edge references mismatch: '+item['id'])
new_sets,new_sizes=enumerate_cross(ns,adjacency(ns,new_edges))
if not set(old_sets).issubset(new_sets): raise ValueError('old structural connectivity not preserved')
new_in,new_out=edge_index(new_edges)
records=[]; edge_ref_counts=collections.Counter()
for s in new_sets:
 rr=refs(s,new_edges,new_in,new_out); old=old_items.get(s)
 roles={}
 for cat,indices in rr.items():
  roles[cat]={name:[i for i in indices if new_edges[i].get('semantic_class')==name] for name in ('CONDITIONAL_CANDIDATE_BLOCK_IDS','SHAPE_ATTRIBUTE','INITIAL_PRE_MIX_VALUE')}
  edge_ref_counts.update({name:len(vals) for name,vals in roles[cat].items()})
 # Prove each old in-scope reference maps to the same new structural relationship.
 if old:
  for category,key in fields.items():
   mapped={mapping[i] for i in old[key] if mapping[i] is not None}
   if not mapped.issubset(rr[category]): raise ValueError('edge relationship rebinding failed: '+old['id'])
 nodes_sorted=sorted(s); fingerprint=digest(json.dumps(nodes_sorted).encode())[:16]
 records.append({'id':'G1_CONDITIONAL_CROSS_'+fingerprint,'nodes':nodes_sorted,'modules':sorted({str(ns[n].get('module')) for n in s}), 'legacy_G1_item_id':None if old is None else old['id'],'edge_indices_bound_to_derived_graph':rr,'special_edge_indices_by_semantic_class':roles,'status':'STRUCTURAL_ONLY','semantic_equivalence_or_savings_proven':False})
# Exact special-edge validation: candidate and KV remain distinct despite same endpoints.
candidates=[i for i,e in enumerate(new_edges) if e.get('semantic_class')=='CONDITIONAL_CANDIDATE_BLOCK_IDS']
shape=[i for i,e in enumerate(new_edges) if e.get('semantic_class')=='SHAPE_ATTRIBUTE']
if len(candidates)!=4 or len(shape)!=1: raise ValueError('special edge count')
for i in candidates:
 e=new_edges[i];matches=[j for j,z in enumerate(new_edges) if (z['producer'],z['consumer'])==(e['producer'],e['consumer']) and z['value']=='shared kv cache/state']
 if len(matches)!=1 or matches[0]==i: raise ValueError('candidate/KV separation failed')
# Serialize deterministic artifacts with SHA-based bindings. Never overwrite different bytes.
def encode(x):return (json.dumps(x,ensure_ascii=False,indent=2)+'\n').encode()
def emit(relative,payload):
 path=ROOT/relative;path.parent.mkdir(parents=True,exist_ok=True)
 try:
  with path.open('xb') as f: f.write(payload);f.flush();os.fsync(f.fileno())
 except FileExistsError:
  if path.read_bytes()!=payload: raise ValueError('refusing to overwrite changed artifact: '+relative)
 return digest(payload)
graph_bytes=encode(graph);graph_hash=digest(graph_bytes)
index={'id':'G1_CONDITIONAL_COMPOSITE_STRUCTURAL_INDEX_V1','derived_graph_sha256':graph_hash,'old_index_sha256':INPUTS['candidates/cross-module-G1.json'],'scope':'Connected 2–4 node sets crossing module labels; next-invocation and shape-attribute edges excluded from connectivity, but shape edges retained as references.','size_coverage':new_sizes,'count':len(records),'items':records,'limitations':['Structural combinations are not feasible fusions or performance predictions.','Source graph, branches, aliases, runtime identity, scheduling and tile readiness remain incomplete.','G5 source templates are not instance-expanded; candidate suboperator semantics require separately scoped review.']}
map_artifact={'old_graph_sha256':INPUTS['graphs/G1.json'],'new_graph_sha256':graph_hash,'old_edge_index_to_new_edge_index':mapping,'deleted_old_edge_indices':[removed],'semantic_classes':{'CONDITIONAL_CANDIDATE_BLOCK_IDS':candidates,'SHAPE_ATTRIBUTE':shape},'next_invocation_excluded_from_index':True}
summary={'id':'G1_CONDITIONAL_INDEX_AUDIT_V1','inputs':INPUTS,'old_graph_nodes':len(g1['nodes']),'old_graph_edges':len(g1['edges']),'new_graph_nodes':len(graph['nodes']),'new_graph_edges':len(new_edges),'old_index_count':len(old_items),'new_index_count':len(records),'new_only_count':len(records)-len(old_items),'old_node_sets_preserved':len(old_items),'old_edge_references_checked':sum(len(item[k]) for item in old_items.values() for k in fields.values()),'old_sizes':old_sizes,'new_sizes':new_sizes,'special_edge_indices':map_artifact['semantic_classes'],'special_reference_counts':dict(edge_ref_counts),'new_graph_sha256':graph_hash,'limits':index['limitations']}
outputs=[('graphs/G1-conditional-materialized-v1.json',graph_bytes),('candidates/cross-module-G1-conditional-v1.json',encode(index)),('candidates/G1-conditional-old-new-edge-map-v1.json',encode(map_artifact)),('evidence/G1-conditional-index-audit-v1.json',encode(summary))]
# Check all conflicts before creating any output.
for name,payload in outputs:
 path=ROOT/name
 if path.exists() and path.read_bytes()!=payload: raise ValueError('refusing to overwrite changed artifact: '+name)
for name,payload in outputs: print(name,emit(name,payload))
print(json.dumps({k:v for k,v in summary.items() if k not in ('inputs','limits')},indent=2))
