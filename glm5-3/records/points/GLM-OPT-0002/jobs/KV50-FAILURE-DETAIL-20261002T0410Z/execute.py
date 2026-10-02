from pathlib import Path
import json,sys,hashlib,re
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc
j=Path(__file__).parent;p=j.parents[1];r=p/"runs/GLM-RUN-0050";audit=p/"jobs/REDUCE-PD50-NATIVE-20261002T0400Z"
def ref(f):
 b=f.read_bytes();return dict(path=str(f),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
assert ref(audit/"reduction.json")["sha256"]=="2cfa09f160b94ddaa2b573ddafbeed19fd276b488748c8a6147b56f42f9bdfd6"
a=json.loads((audit/"reduction.json").read_text());assert a["inference_attempts"]==0 and a["state"]["status"]=="failed" and a["source_pins"]==18
details={}
for role in ["P","D"]:
 for rank in [0,1]:
  f=audit/(role+str(rank)+".native.log");txt=f.read_text();lines=txt.splitlines()
  weights=[l for l in lines if "Loading model weights took"in l];assert len(weights)==16 and all("25.5299 GB"in l for l in weights)
  capacity=[l for l in lines if "GPU KV cache size:"in l or "NPU KV cache size:"in l]
  assert capacity
  expected=91924 if role=="P"else 187089
  assert any(str(expected)in l.replace(",","")for l in capacity)
  free=[float(x)for x in re.findall(r"Initial free memory ([0-9.]+) GiB",txt)];assert len(free)==16
  e=[(i,l)for i,l in enumerate(lines)if "torch.OutOfMemoryError: NPU out of memory."in l]
  d=dict(raw=ref(f),weights_complete_workers=16,weight_GB_per_worker=25.5299,native_KV_capacity_per_DP=expected,capacity_raw=capacity,initial_free_GiB_min=min(free),initial_free_GiB_max=max(free),explicit_KV_skips_profiling=all("skipping memory profiling"in l for l in lines if "Initial free memory"in l))
  if role=="D":
   assert e and not any("Capturing CUDA graph"in l or "Capturing NPU graph"in l or "Graph capturing finished"in l for l in lines)
   i,line=e[0];prefix=line.split(" ERROR ")[0]
   stack=[l for l in lines[max(0,i-100):i+1]if l.startswith(prefix)]
   assert any("_allocate_int8_cache_tensor"in l for l in stack)and any("raw_tensor = torch.zeros"in l for l in stack)
   d.update(first_native_OOM=line,first_worker_stack=stack,OOM_lines=[v for _,v in e],graph_capture_observed=False)
  else:assert not e
  details[role+str(rank)]=d
out=dict(at=utc(),run_id=r.name,classification="native raw int8 KV allocation OOM before D Graph capture",verdict="REJECT",scope="Specific P1GiB/D2GiB/batch4096/Dseq2 dualresident configuration; not hardware capacity bound",terminal_audit=ref(audit/"reduction.json"),state=a["state"],native=details,inference_attempts=0,effective_output_tokens=0,signals=0,new_models=0,new_requests=0,limits=["Allocator/fragmentation/segment/contiguity cause unknown; error free memory is nonzero and no allocator event trace captured","Native planned logical KV capacity differs from physical aligned storage; actual physical allocations are native","Native worker explicit KV branch skips profiling; gmu .48 does not cap explicit KV allocation","Terminal Result inherited Run49 prose label; actual run_id/paths/spec/owners allRun50, frozen raw unchanged","Role HCCL accepted sufficiently to load 32D workers; no global HCCL/transport certificate","No inference/PD transport/Graph/runtime/stable capacity claim"],next_candidate=dict(run_id="GLM-RUN-0051",P_KV_GiB=.9,D_KV_GiB=1.6,batch=1024,D_sequences=1,P_maxlen=81933,D_maxlen=144384,native_operators_unchanged=True,D_K5_FULL_preserved=True,fit="unproven until actual native guards/startup/pilot"))
atomic_json(j/"reduction.json",out)
brief=json.loads((r/"reduction_brief.json").read_text());brief.update(verdict="REJECT",classification=out["classification"],KV_failure_detail=ref(j/"reduction.json"),limits=out["limits"]);atomic_json(r/"reduction_brief.json",brief)
m=json.loads((r/"manifest.json").read_text());m.update(verdict="REJECT",status="failed",valid=False,KV_failure_detail=ref(j/"reduction.json"));atomic_json(r/"manifest.json",m)
(r/"summary.md").write_text("# GLM-RUN-0050\n\nREJECT specific dualresident P1GiB/D2GiB/batch4096/Dseq2 startup. P32workers ready; D32workers load main+draft weights25.5299GB/worker, native planned P91924/D187089 tokens perDP. D fails native raw int8 KV allocation via torch.zeros before Graph capture; no inference, zero outputs. Nonzero free memory in errors does not establish fragmentation/allocator cause. Explicit KV skips profiling and gmu does not cap it. Actual18pin/native9files/owner/HCCLenv terminal audit authentic; frozen Result stale Run49 prose label corrected here. Run51 .9/1.6GiB/batch1024/Dseq1 retains contexts/K5FULL/operators, actual native capacity and fit unproven. CurrentNone.\n")
atomic_json(j/"result.json",dict(schema_version=1,job_id=j.name,status="completed",summary="Run50 REJECT actual native raw KV allocation OOM before D Graph; P32ready/D32weightsloaded/nativecapacity captured,0requests; allocation cause unknown",execution=dict(inner_exit_code=0,acceptance="passed",processes=[]),findings=[],evidence=[dict(id="reduction",locator="frozen 50 native OOM full stack/weights/KV capacity/18pin audit",**ref(j/"reduction.json"))],unknowns=out["limits"],decision_request=None,next_check_at=None))
print(json.dumps(dict(verdict=out["verdict"],reduction=ref(j/"reduction.json"),D_initial_free={k:[v["initial_free_GiB_min"],v["initial_free_GiB_max"]]for k,v in details.items()if k.startswith("D")})))

