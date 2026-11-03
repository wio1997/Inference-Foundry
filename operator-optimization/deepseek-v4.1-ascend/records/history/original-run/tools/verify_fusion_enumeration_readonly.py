"""Rebuild bounded structural sets without modifying the task directory."""
from pathlib import Path
from collections import defaultdict,Counter
import hashlib,itertools,json
T=Path(__file__).resolve().parents[1]
def read(p):return json.loads((T/p).read_text())
def sha(p):return hashlib.sha256((T/p).read_bytes()).hexdigest()
def enumerate_sets(t,omit=()):
 nodes={x['id']:x for x in t['nodes']};eligible={n for n,x in nodes.items() if not x['kind'].startswith('EXTERNAL') and n not in omit};adj=defaultdict(set)
 for e in t['edges']:
  a,b=e['producer'],e['consumer']
  if a in eligible and b in eligible:adj[a].add(b);adj[b].add(a)
 front={frozenset([n]) for n in eligible};out=set()
 for size in (2,3,4):
  front={s|{v} for s in front for v in set().union(*(adj[x] for x in s))-s}
  for s in front:out.add(({2:'PAIR',3:'TRIPLE',4:'MULTI_OPERATOR'}[size],tuple(sorted(s)),None))
 for src in nodes:
  users=sorted({e['consumer'] for e in t['edges'] if e['producer']==src and e['consumer'] in eligible})
  for a,b in itertools.combinations(users,2):out.add(('BRANCH_MULTICONSUMER',(a,b),src))
 return out
g=read('graphs/G1.json');nodes={n['id']:n for n in g['nodes']};adj=defaultdict(set)
for e in g['edges']:
 if e.get('dependency_kind')=='NEXT_INVOCATION_RAW':continue
 a,b=e['producer'],e['consumer'];adj[a].add(b);adj[b].add(a)
front={frozenset([n]) for n in nodes};cross=set()
for size in (2,3,4):
 front={s|{v} for s in front for v in set().union(*(adj[x] for x in s))-s}
 cross.update(s for s in front if len({nodes[x].get('module') for x in s})>1)
saved=read('candidates/cross-module-G1.json');observed=[frozenset(x['nodes']) for x in saved['items']];assert len(observed)==len(set(observed))==len(cross)==22802 and set(observed)==cross
base=read('candidates/structural-index-G2.json');seen=defaultdict(set)
for x in base['items']:seen[x['template_id']].add((x['category'],tuple(x['node_ids']),x.get('shared_input')))
g2=read('graphs/source-templates-G2.json');results={}
for t in g2['templates']:
 got=enumerate_sets(t);assert got==seen[t['id']],(t['id'],len(got),len(seen[t['id']]));results[t['id']]=len(got)
g5=read('graphs/source-templates-G5.json');templates={x['id']:x for x in g5['templates']}
for tid,p in [('ENGRAM','candidates/engram-G5-structural-index.json'),('DSPARK_BOUNDARIES','candidates/dspark_boundaries-G5-structural-index.json')]:
 x=read(p);obs={(z['category'],tuple(z['nodes']),z.get('shared_input')) for z in x['items']};assert len(obs)==len(x['items']);got=enumerate_sets(templates[tid],('confidence','dynamic_scheduler'));assert got==obs,(tid,len(got),len(obs));results[tid+'_G5']=len(got)
out={'status':'PASS_BOUNDED_STRUCTURAL_REPLAY_ONLY','g1_cross_sets':len(cross),'g2_template_counts':results,'g1_sha256':sha('graphs/G1.json'),'g2_sha256':sha('graphs/source-templates-G2.json'),'g5_sha256':sha('graphs/source-templates-G5.json'),'limitation':'Exact bounded enumeration replay only. It does not prove semantic equivalence, complete original graph or global exhaustion.','task_files_modified':False}
print(json.dumps(out,indent=2))
