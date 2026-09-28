import copy,hashlib,importlib.util,itertools,json,random
from pathlib import Path
import scipy
ROOT=Path("/data/wio/Inference_Foundry")
sp=ROOT/"scripts/loop081_run99_prefix_envelope_run595.py"
s=importlib.util.spec_from_file_location("r595",sp);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
out=ROOT/"evidence/20260928_loop081_bound/run595";summary=json.loads((out/"summary.json").read_text())
assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==v for p,v in summary["source_sha256"].items())
maxerr=0;all8=[]
for base,cohorts in [(m.BASE,range(5,17)),(m.A0,range(5,9))]:
 for c in cohorts:
  ref=json.loads((base/f"rank0_cohort{c}.json").read_text());L,T,st=m.window_data(ref)
  for r in range(8):
   x=json.loads((base/f"rank{r}_cohort{c}.json").read_text());assert x["rank"]==r and x["cohort"]==c and x["pass"]
   assert m.window_data(x)==(L,T,st) and x["req_ids"]==ref["req_ids"]
  for name,mean in ref["acceptance_window_means"].items():
   a,b=map(int,name.split("-"));raw=mean*12*(b-a+1);maxerr=max(maxerr,abs(raw-round(raw)))
  all8.append([str(base.relative_to(ROOT)),c])
for row in summary["rows"]:
 for key in ["min_solver","max_solver"]:assert row[key]["mip_gap"]==0 and "Optimal" in row[key]["status"]
# Exact small instance of the production solver: 2 free slots over 3 cycles,
# 10 additional slots have staged=1 and therefore are active for cycle0 only.
# Enumerate ALL possible positive-prefix 1..8 sequences, including within-window parking.
seqs=[v+(0,)*(3-n) for n in range(1,4) for v in itertools.product(range(1,9),repeat=n)]
groups={}
for a in seqs:
 for b in seqs:
  key=(sum(a),sum(b),a[0]+b[0]+10,sum(a[1:])+sum(b[1:]))
  val=sum(x>0 for x in a+b)+10
  if key not in groups:groups[key]=[val,val]
  else:groups[key]=[min(groups[key][0],val),max(groups[key][1],val)]
keys=sorted(groups);rng=random.Random(595);chosen=rng.sample(keys,48)
small=[]
for s1,s2,w0,w1 in chosen:
 lo,lmeta=m.solve([1,2],[w0,w1],[s1,s2]+[1]*10,False)
 hi,hmeta=m.solve([1,2],[w0,w1],[s1,s2]+[1]*10,True)
 assert [lo,hi]==groups[(s1,s2,w0,w1)]
 small.append({"staged2":[s1,s2],"window_totals":[w0,w1],"active_interval":[lo,hi]})
# Check exact A0 trajectory at every window rather than only interval inclusion.
a0=[]
parks=json.loads((ROOT/"evidence/20260928_loop081_bound/run594/summary.json").read_text())["rows"]
for c in range(5,9):
 rt=json.loads((m.A0/f"rank0_cohort{c}.json").read_text());L,T,st=m.window_data(rt)
 trace=json.loads((m.A0.parent/"trace"/f"trace_rank0_cohort{c}.json").read_text())
 pr=next(x for x in parks if x["rank"]==0 and x["cohort"]==c);ends=[p[0] if p else len(trace) for p in pr["park_cycles"]]
 Ys=[];As=[];start=0
 for leng in L:
  As.append([max(0,min(leng,end-start)) for end in ends])
  Ys.append([sum(trace[t]["counts"][slot] for t in range(start,min(start+leng,ends[slot]))) if ends[slot]>start else 0 for slot in range(12)])
  start+=leng
 assert [sum(y) for y in Ys]==T
 assert [sum(y[s] for y in Ys) for s in range(12)]==st
 for w,leng in enumerate(L):
  for slot in range(12):
   a=As[w][slot];y=Ys[w][slot];assert a<=y<=8*a
   if w+1<len(L) and As[w+1][slot]>0:assert a==leng
 assert 8*sum(map(sum,As))==pr["active_target_rows"]
 a0.append({"cohort":c,"active_rows":8*sum(map(sum,As)),"window_assignment_feasible":True})
ref=json.loads((m.BASE/"rank0_cohort5.json").read_text());negative={}
for name,mut in [("noninteger_window",lambda x:x["acceptance_window_means"].__setitem__("0-7",x["acceptance_window_means"]["0-7"]+0.001)),("wrong_staged_sum",lambda x:x["staged_output_counts"].__setitem__(0,x["staged_output_counts"][0]+1)),("wrong_cycles",lambda x:x.__setitem__("cycles",x["cycles"]+1))]:
 x=copy.deepcopy(ref);mut(x)
 try:m.window_data(x);raise AssertionError("negative admitted")
 except ValueError as e:negative[name]=str(e)
# All slots start positive; zero first-window output is infeasible.
try:m.solve([1,2],[0,12],[1]*12,False);raise AssertionError("infeasible admitted")
except ValueError as e:negative["zero_initial_window"]=str(e)
result={"scope":"CPU-only independent audit; no service/NPU", "scipy_version":scipy.__version__,"script_sha256":hashlib.sha256(sp.read_bytes()).hexdigest(),"summary_sha256":hashlib.sha256((out/"summary.json").read_bytes()).hexdigest(),"all128_input_hashes_match":True,"source_file_count":len(summary["source_sha256"]),"all8_runtime_aggregate_request_consensus":all8,"max_integer_recovery_residual":maxerr,"all32_recorded_solver_terminations_optimal_zero_gap":True,"small_exhaustive_sequence_count":len(seqs),"small_exhaustive_trajectory_pairs":len(seqs)**2,"small_exhaustive_aggregate_groups":len(groups),"small_exact_comparisons":small,"a0_exact_window_feasibility":a0,"negative_tests":negative}
(out/"astra_cpu_audit.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps({k:v for k,v in result.items() if k not in ("small_exact_comparisons","all8_runtime_aggregate_request_consensus")},indent=2))
