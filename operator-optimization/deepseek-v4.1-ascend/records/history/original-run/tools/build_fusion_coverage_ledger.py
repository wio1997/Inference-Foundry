"""Per-node/edge structural coverage of the saved G1/G5 enumerations."""
from pathlib import Path
from collections import Counter,defaultdict
import csv,datetime,hashlib,json
T=Path(__file__).resolve().parents[1]
def read(p):return json.loads((T/p).read_text())
def sha(p):return hashlib.sha256((T/p).read_bytes()).hexdigest()
g1=read('graphs/G1.json');cross=read('candidates/cross-module-G1.json');assert cross['G1_sha256']==sha('graphs/G1.json') and len(cross['items'])==22802
nodes={x['id']:x for x in g1['nodes']};edge_counts=defaultdict(Counter);node_counts=defaultdict(Counter)
for c in cross['items']:
 size=len(c['nodes']);assert size in (2,3,4)
 for n in c['nodes']:assert n in nodes;node_counts[n][size]+=1
 for i in c['internal_edges_G1_edge_indices']:
  e=g1['edges'][i];assert e['producer'] in c['nodes'] and e['consumer'] in c['nodes'];edge_counts[i][size]+=1
nrows=[]
for n in g1['nodes']:
 c=node_counts[n['id']];nrows.append({'graph':'G1','template':'','node_id':n['id'],'module':n.get('module',''),'kind':n.get('kind',''),'pair_count':c[2],'triple_count':c[3],'four_count':c[4],'branch_source_count':0,'status':'STRUCTURALLY_ENUMERATED_CROSS_MODULE' if c else 'UNKNOWN_NOT_INCLUDED_IN_CROSS_MODULE_ENUMERATION','semantic_coverage':'UNKNOWN'})
erows=[]
for i,e in enumerate(g1['edges']):
 c=edge_counts[i];a,b=e['producer'],e['consumer'];crossmod=nodes[a].get('module')!=nodes[b].get('module')
 erows.append({'graph':'G1','template':'','edge_index':i,'producer':a,'consumer':b,'dependency_kind':e.get('dependency_kind','UNKNOWN'),'cross_module':crossmod,'pair_count':c[2],'triple_count':c[3],'four_count':c[4],'status':'STRUCTURALLY_ENUMERATED_CROSS_MODULE' if c else 'UNKNOWN_NOT_INCLUDED_IN_CROSS_MODULE_ENUMERATION','semantic_coverage':'UNKNOWN'})
g2=read('graphs/source-templates-G2.json');g5=read('graphs/source-templates-G5.json');templates2={x['id']:x for x in g2['templates']};templates5={x['id']:x for x in g5['templates']}
updated={'ENGRAM':'candidates/engram-G5-structural-index.json','DSPARK_BOUNDARIES':'candidates/dspark_boundaries-G5-structural-index.json'}
base=read('candidates/structural-index-G2.json');ix=defaultdict(list)
for x in base['items']:ix[x['template_id']].append({'nodes':x['node_ids'],'category':x['category'],'shared_input':x.get('shared_input')})
source_indices={'G1_CROSS':'candidates/cross-module-G1.json','G2_BASE':'candidates/structural-index-G2.json',**updated}
for tid,t in templates5.items():
 if tid in updated:
  items=read(updated[tid])['items']
 else:
  assert t==templates2[tid],tid;items=ix[tid]
 tnodes={n['id']:n for n in t['nodes']};cn=defaultdict(Counter);ce=defaultdict(Counter);branch=Counter()
 for c in items:
  sel=c['nodes'];assert all(n in tnodes for n in sel),(tid,sel)
  cat=c['category'];size=len(sel)
  for n in sel:cn[n][cat]+=1
  if cat=='BRANCH_MULTICONSUMER' and c.get('shared_input'):
   assert c['shared_input'] in tnodes;branch[c['shared_input']]+=1
  if cat in ('PAIR','TRIPLE','MULTI_OPERATOR'):
   for i,e in enumerate(t['edges']):
    if e['producer'] in sel and e['consumer'] in sel:ce[i][size]+=1
 for n in t['nodes']:
  c=cn[n['id']];nrows.append({'graph':'G5','template':tid,'node_id':n['id'],'module':tid,'kind':n['kind'],'pair_count':c['PAIR'],'triple_count':c['TRIPLE'],'four_count':c['MULTI_OPERATOR'],'branch_source_count':branch[n['id']],'status':'STRUCTURALLY_ENUMERATED_IN_TEMPLATE' if c or branch[n['id']] else 'UNKNOWN_NOT_INCLUDED_IN_TEMPLATE_ENUMERATION','semantic_coverage':'UNKNOWN'})
 for i,e in enumerate(t['edges']):
  c=ce[i];erows.append({'graph':'G5','template':tid,'edge_index':i,'producer':e['producer'],'consumer':e['consumer'],'dependency_kind':e.get('dependency_kind','UNKNOWN'),'cross_module':'','pair_count':c[2],'triple_count':c[3],'four_count':c[4],'status':'STRUCTURALLY_INTERNAL_TO_ENUMERATED_SET' if c else 'UNKNOWN_NOT_INTERNAL_TO_ENUMERATED_SET','semantic_coverage':'UNKNOWN'})
o=T/'coverage';o.mkdir(exist_ok=True)
for name,rows in [('node-structural-coverage.csv',nrows),('edge-structural-coverage.csv',erows)]:
 p=o/name;assert not p.exists()
 with p.open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
summary={'timestamp':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'PER_NODE_EDGE_STRUCTURAL_LEDGER_WITH_UNKNOWNS','graph_sha256':{'G1':sha('graphs/G1.json'),'G5':sha('graphs/source-templates-G5.json'),'G2':sha('graphs/source-templates-G2.json')},'index_sha256':{k:sha(v) for k,v in source_indices.items()},'node_rows':len(nrows),'edge_rows':len(erows),'node_status_counts':dict(Counter(r['status'] for r in nrows)),'edge_status_counts':dict(Counter(r['status'] for r in erows)), 'limitations':['G1 counts cover cross-module connected sets up to 4 composite nodes only; zero means not represented in that index, not proof no opportunity.','G5 counts cover individual partial templates; source graphs, aliasing, consumers, dynamic branches, and primitive expansions remain incomplete.','Counts establish structural inclusion only, not equivalence, feasibility, materialization savings, or global exhaustion.','G1 and G5 overlap in semantic scope; do not sum counts as unique opportunities.','Same-template edge with an external producer/consumer may intentionally be excluded from internal candidate sets and is marked UNKNOWN.']}
p=o/'summary.json';assert not p.exists();p.write_text(json.dumps(summary,indent=2)+'\n')
s=read('fusion-scan.json');s['per_node_edge_coverage_ledger']={'summary':'coverage/summary.json','nodes':'coverage/node-structural-coverage.csv','edges':'coverage/edge-structural-coverage.csv','structural_only':True};(T/'fusion-scan.json').write_text(json.dumps(s,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:summary[k] for k in ['status','node_rows','edge_rows','node_status_counts','edge_status_counts']},indent=2))
