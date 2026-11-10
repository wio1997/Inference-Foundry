#!/usr/bin/env python3
"""Finite source-bound dependency witnesses, never a native stream or bit test."""
from pathlib import Path
from hashlib import sha256
from collections import defaultdict
import ast,json

ROOT=Path('audit/execution-search')
def topo(nodes,edges):
 adj=defaultdict(list);degree={n:0 for n in nodes}
 for a,b in edges:adj[a].append(b);degree[b]+=1
 ready=sorted(n for n in nodes if degree[n]==0);out=[]
 while ready:
  a=ready.pop(0);out.append(a)
  for b in adj[a]:
   degree[b]-=1
   if not degree[b]:ready.append(b);ready.sort()
 return out if len(out)==len(nodes) else None
def before(nodes,edges,a,b):
 # A valid totalorder with b before a witnesses an absent dependency.
 return topo(nodes,edges|{(b,a)}) is None
def graph(layers,engram,aux,mtp,mode,drop=None):
 nodes={'input_write','seed_premix','graph_end','next_input_overwrite'};edges={('input_write','seed_premix'),('graph_end','next_input_overwrite')};checks=[];last='seed_premix';lastpre='seed_premix'
 for layer in range(layers):
  if layer in aux:
   snap=f'L{layer}.aux_snapshot';nodes.add(snap);edges.add((last,snap));last=snap
  if layer in engram:
   update=f'L{layer}.Engram_update';nodes.add(update);edges.add((last,update));last=update
  for kind in ['attention','ffn']:
   stem=f'L{layer}.{kind}.';x=stem+'input_ready';fork=stem+'fork';cr=stem+'coefficient_reads_x_and_premix';cd=stem+'whole_original_coeff_done';ey=stem+'early_original_collapse';norm=stem+'original_norm';work=stem+'original_downstream';join=stem+'own_Post_join';post=stem+'original_HCPost';reuse=stem+'storage_reuse'
   nodes.update([x,fork,cr,cd,ey,norm,work,join,post,reuse]);edges.add((last,x));edges.add((lastpre,x))
   links=[(x,fork),(fork,cr),(cr,cd),(fork,ey),(ey,norm),(norm,work),(work,join),(cd,join),(join,post),(post,reuse)]
   for a,b in links:
    if drop=='producer_fence' and (a,b)==(fork,cr):continue
    if drop=='late_coeff_join' and (a,b)==(cd,join):continue
    edges.add((a,b))
   checks += [(x,cr),(cd,post),(cr,reuse),(post,reuse),(ey,work)]
   last=post;lastpre=cd
 end='target_MTP_fullstate_copy' if mtp else 'final_original_collapse';nodes.add(end);edges.add((last,end));edges.add((lastpre,end))
 if mtp:
  nodes.add('final_original_collapse');edges.add((end,'final_original_collapse'));end='final_original_collapse'
 edges.add((end,'graph_end'));checks += [(lastpre,end),(last,'graph_end')]
 for n in nodes:
  if n.endswith('whole_original_coeff_done'):checks.append((n,'next_input_overwrite'))
 if drop=='skip_Post_without_cleanup':
  # Separate early-return path after a submitted coefficient branch; no terminal join.
  nodes={'input_write','coefficient_submitted','coeff_done','early_return','next_input_overwrite'};edges={('input_write','coefficient_submitted'),('coefficient_submitted','coeff_done'),('coefficient_submitted','early_return'),('early_return','next_input_overwrite')};checks=[('coeff_done','next_input_overwrite')]
 assert topo(nodes,edges) is not None
 failures=[]
 for a,b in checks:
  if not before(nodes,edges,a,b):
   witness=topo(nodes,edges|{(b,a)});assert witness.index(b)<witness.index(a);failures.append(dict(required_before=a,required_after=b,witness_total_order=witness))
 return dict(layers=layers,engram_before_layers=engram,aux_before_layers=aux,MTP_fullstate=mtp,mode=mode,removed_dependency=drop,nodes=len(nodes),edges=len(edges),required_pairs=len(checks),all_required_ordered=not failures,violations=failures[:2],qualification='Finite invented scheduling model, not runtime stream/capture execution or numerical proof.')
def main():
 dep=ROOT/'hc-coefficient-consumer-dependencies.json';src=json.loads(dep.read_text())['sources'];pins=[];seeds=[]
 for name,cls in [('models/deepseek_v41/model.py','DeepseekV41Model'),('models/deepseek_v41/dspark.py','DeepseekV41DSparkModel')]:
  r=next(r for r in src if r['runtime_path'].endswith('/'+name));p=Path(r['local_path']);b=p.read_bytes();assert sha256(b).hexdigest()==r['runtime_sha256'];tree=ast.parse(b.decode());c=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==cls);fn=next(n for n in c.body if isinstance(n,ast.FunctionDef) and n.name=='forward')
  assignments=[n for n in ast.walk(fn) if isinstance(n,ast.Assign) and any('pre_mix' in ast.unparse(t) for t in n.targets)];seed=next(n for n in assignments if any(ast.unparse(t)=='pre_mix[:, 0]' for t in n.targets));assert isinstance(seed.value,ast.Constant) and seed.value.value==1.0
  zeros=next(n for n in assignments if any(ast.unparse(t)=='pre_mix' for t in n.targets));assert 'new_zeros' in ast.unparse(zeros.value)
  call=next(n for n in ast.walk(fn) if isinstance(n,ast.Call) and ast.unparse(n.func)=='layer');assert ast.unparse(call.args[2])=='pre_mix'
  seeds.append(dict(model=cls,zero_assignment_line=zeros.lineno,seed_assignment_line=seed.lineno,seed=ast.unparse(seed),layer_call_line=call.lineno,premix_argument=ast.unparse(call.args[2]),native_bit_proof=False));pins.append(dict(path=str(p),sha256=sha256(b).hexdigest()))
 fixtures=[]
 for mode in ['NONE','FULL']:
  for layers,engram,aux,mtp in [(40,[1,14],[36,37,38],False),(40,[1,14],[],True),(3,[],[],False)]:
   fixtures.append(graph(layers,engram,aux,mtp,mode))
 for drop in ['producer_fence','late_coeff_join','skip_Post_without_cleanup']:fixtures.append(graph(1,[],[],False,'FULL',drop))
 assert all(f['all_required_ordered'] for f in fixtures[:6]) and all(not f['all_required_ordered'] for f in fixtures[6:])
 result=dict(status='SOURCE_SEED_BINDING_AND_FINITE_FORK_JOIN_ORDER_WITNESSES',seeds=seeds,fixtures=fixtures,normal_fixtures=6,negative_fixtures=3,
  proposed_protocol=['One dedicated local coefficient stream per device/model; original main-stream collective order retained, not borrowing Engram/shared-expert streams or their descriptor external events.','After actual seed/Engram/input writes, auxiliary waits for that exact main prefix. Retained original HCPre reads both rounded residual and oldpre_mix, including its unusedY; hold every input/output/workspace owner until actual completion.','Main computes original-order earlycollapse and originalnormalization/downstream. Before each originalHCPost, join its entire coefficient operation; this dominates later auxsnapshots, Engram updates, nextpremix and final/MTP readers.','In FULL both fork and join must be recorded in samegraph using actual capture-compatible event semantics; graphupdate ExternalEvents gate hostmetadata updates, not coefficient completion. Static addresses/owners and allbucket/error/fallback generations require native proof.','Skipped/failed consumer must join whatever work was actuallysubmitted before reuse or quarantine owners; do not wait on an unrecorded event or rely on Pythonreturn.'],
  limits=['Six finite normal models and three deliberatelybroken recipes, not all legalbranch or topologicalschedule enumeration.','AST proves caller expressions/seeds only, not arbitrary runtime values or bitwise arithmetic.','No physical co-residency, graph capture or actual Event API executed. No substitute for native primitivebit and matched net/request tests.'],inputs=[dict(path=str(dep),sha256=sha256(dep.read_bytes()).hexdigest())]+pins,analyzer_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),new_NPU_work=0,new_profile=0,service_requests=0,remote_writes=[],new_primitive_admissions=0,whole_chain_complete=False)
 (ROOT/'hc-fork-join-cpu.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(dict(seeds=seeds,normal_models_pass=6,broken_models_witnessed=3,required_pairs=sum(f['required_pairs'] for f in fixtures[:6]),native_bit_proof=False),ensure_ascii=False))
if __name__=='__main__':main()
