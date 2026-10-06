import json,hashlib,sys,collections,statistics,re
from pathlib import Path
J=Path(__file__).parent;R=J.parents[1];G=R.parents[4];sys.path.insert(0,str(G/"runtime"))
from phase_runner import atomic_json,utc
job=json.loads(Path(sys.argv[1]).read_text())
assert json.loads((R/"jobs/EXPORT-DEVICE-20261006/export_exit.json").read_text())["exit_code"]==0
clock=json.loads((R/"before_clock.json").read_text());offset=(clock["realtime_ns"]-(clock["monotonic_before_ns"]+clock["monotonic_after_ns"])/2)/1000
clients=[]; starts=[];ends=[]
for f in sorted(R.glob("request*.events.json")):
 v=json.loads(f.read_text());total=0;milestones={};rows=[]
 for e in v["events"]:
  n=sum(len(c.get("token_ids") or []) for c in e.get("payload",{}).get("choices",[]));total+=n
  if n:rows.append(dict(tokens=total,monotonic_ns=e["monotonic_ns"]))
  for k in [8,56]:
   if total>=k and k not in milestones:milestones[k]=e["monotonic_ns"]/1000+offset
 assert total==64 and v["events"][-1].get("done")
 clients.append(dict(path=str(f),total=total,milestones=milestones,token_progress=rows));starts.append(milestones[8]);ends.append(milestones[56])
lo=max(starts);hi=min(ends);assert hi>lo

def merge(xs):
 out=[]
 for a,b in sorted(xs):
  a=max(a,lo);b=min(b,hi)
  if b<=a:continue
  if out and a<=out[-1][1]:out[-1][1]=max(b,out[-1][1])
  else:out.append([a,b])
 return out

def length(xs):return sum(b-a for a,b in merge(xs))
def intersect(xs,ys):
 xs=merge(xs);ys=merge(ys);out=[];a=b=0
 while a<len(xs) and b<len(ys):
  s=max(xs[a][0],ys[b][0]);e=min(xs[a][1],ys[b][1])
  if e>s:out.append([s,e])
  if xs[a][1]<=ys[b][1]:a+=1
  else:b+=1
 return out

def gaps(xs):
 xs=merge(xs);out=[];last=lo
 for a,b in xs:
  if a>last:out.append([last,a])
  last=b
 if hi>last:out.append([last,hi])
 return out

pre=json.loads((R/"preflight.json").read_text());ranks=[];all_intervals={};evidence=[];raw_samples=[]
coretypes={"AI_CORE","AI_VECTOR_CORE","MIX_AIC","MIX_AIV"};copytypes={"SDMA_SQE","PCIE_DMA_SQE"}
for rank in pre["ranks"]:
 cp=rank["container_pid"];dirs=[p for p in (R/"device_trace").glob("PROF*") if re.search(r"_0*"+str(cp)+r"[A-Z]+$",p.name)];assert len(dirs)==1,(cp,dirs)
 fs=list(dirs[0].glob("mindstudio_profiler_output/msprof_*.json"));assert len(fs)==1
 f=fs[0];v=json.loads(f.read_text());meta={e["pid"]:e["args"]["name"] for e in v if e.get("ph")=="M" and e.get("name")=="process_name"}
 hp=[pid for pid,name in meta.items() if name=="Ascend Hardware"];assert len(hp)==1
 hw=[(i,e)for i,e in enumerate(v)if e.get("ph")=="X" and e.get("pid")==hp[0] and float(e["ts"])+float(e["dur"])>lo and float(e["ts"])<hi]
 assert hw
 groups=collections.defaultdict(list);types=collections.Counter();names=collections.Counter();events=[]
 for i,e in hw:
  a=float(e["ts"]);b=a+float(e["dur"]);t=e.get("args",{}).get("Task Type");types[t]+=1;names[e["name"]]+=1
  g="compute" if t in coretypes else "communication" if t=="COMMUNICATION" else "copy" if t in copytypes else "aicpu" if t=="AI_CPU" else "control_wait"
  groups[g].append([a,b]);events.append(dict(index=i,event=e,group=g))
 busy=groups["compute"]+groups["communication"]+groups["copy"]+groups["aicpu"]
 ints={k:merge(x)for k,x in groups.items()};ints["busy"]=merge(busy)
 key=f'PP{rank["pp"]}_TP{rank["tp"]}';all_intervals[key]=ints
 row=dict(rank=rank,key=key,trace=str(f),trace_sha256=hashlib.sha256(f.read_bytes()).hexdigest(),trace_bytes=f.stat().st_size,hardware_pid=hp[0],hardware_event_count=len(hw),task_counts=dict(types),top_names=names.most_common(10),union_us={k:length(x)for k,x in ints.items()},compute_comm_overlap_us=length(intersect(groups["compute"],groups["communication"])),busy_uncovered_us=length(gaps(busy)),largest_busy_uncovered_us=max([b-a for a,b in gaps(busy)] or [0]),largest_busy_uncovered=gaps(busy)[:2])
 ranks.append(row)
 if rank["tp"]==0:
  # Exact events with original timeline array indices, not synthetic causal labels.
  raw_samples.append(dict(rank=key,trace=str(f),sha256=row["trace_sha256"],metadata=[e for e in v if e.get("ph")=="M" and e.get("name")=="process_name"],longest_compute=sorted([e for e in events if e["group"]=="compute"],key=lambda x:float(x["event"]["dur"]),reverse=True)[:5],longest_communication=sorted([e for e in events if e["group"]=="communication"],key=lambda x:float(x["event"]["dur"]),reverse=True)[:5],longest_control_wait=sorted([e for e in events if e["group"]=="control_wait"],key=lambda x:float(x["event"]["dur"]),reverse=True)[:5],first_events=events[:25]))
 del v
assert {(x["rank"]["pp"],x["rank"]["tp"]) for x in ranks}=={(pp,tp)for pp in range(2)for tp in range(8)}
stages={}
for pp in range(2):
 stages[pp]={g:merge([z for k,v in all_intervals.items() if k.startswith("PP"+str(pp)+"_") for z in v[g]])for g in ["compute","communication","copy","busy"]}
width=5000;bins=[]
for i in range(int((hi-lo)/width)+1):
 a=lo+i*width;b=min(a+width,hi)
 if b<=a:continue
 vals={}
 for pp in range(2):
  vals[pp]={g:sum(max(0,min(y,b)-max(x,a))for x,y in stages[pp][g])/(b-a) for g in stages[pp]}
 bins.append(dict(offset_us=a-lo,duration_us=b-a,stage_fraction=vals))
summary=dict(at=utc(),run="GLM-RUN-0244",window=dict(lo_realtime_us=lo,hi_realtime_us=hi,duration_us=hi-lo,definition="All4 requests after8 tokens and before any reaches56; actual SSE progress",clock_bracket_ns=clock["monotonic_after_ns"]-clock["monotonic_before_ns"],mono_to_realtime_offset_us=offset),clients=clients,ranks=ranks,stage_union_us={pp:{g:length(v)for g,v in ints.items()}for pp,ints in stages.items()},stage_compute_overlap_us=length(intersect(stages[0]["compute"],stages[1]["compute"])),stage_busy_overlap_us=length(intersect(stages[0]["busy"],stages[1]["busy"])),combined_busy_union_us=length(stages[0]["busy"]+stages[1]["busy"]),scope="Profiled direct-native4x64; existing code/epochs; no matched E2E or inferred removable time",unknowns=["Runtime graph tasks/collectives may overlap; interval unions are not additive costs","Control/notify waits are dependencies, not device compute or proved removable bubbles","All-rank trace/copy/communication completeness and clock semantics still require Sol direct reading","No same-Run batch IDs or independent half-batch cost counterfactual; no E2E Gain"])
atomic_json(J/"summary.json",summary);atomic_json(J/"stage_bins.json",dict(window=summary["window"],bins=bins));atomic_json(J/"decisive_events.json",dict(window=summary["window"],ranks=raw_samples));atomic_json(J/"intervals.json",all_intervals)
for p in [J/"summary.json",J/"stage_bins.json",J/"decisive_events.json",J/"intervals.json"]:evidence.append(dict(id=p.name,path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size,locator="Actual selected hardware tracks, original array indices and per-rank interval unions"))
atomic_json(Path(job["result"]["path"]),dict(schema_version=1,job_id=job["job_id"],status="completed",summary="Mechanical all16 hardware interval/SSE window reduction complete, no causal/performance verdict.",execution=dict(inner_exit_code=0,acceptance="passed",processes=[]),findings=[dict(kind="fact",text="All16 PP/TP rank timelines and effective4/256 output trace reduced",scope=dict(run="GLM-RUN-0244",profiled=True,per_rank=True),evidence_ids=["summary.json","decisive_events.json"])],evidence=evidence,unknowns=summary["unknowns"],decision_request=None,next_check_at=None))
