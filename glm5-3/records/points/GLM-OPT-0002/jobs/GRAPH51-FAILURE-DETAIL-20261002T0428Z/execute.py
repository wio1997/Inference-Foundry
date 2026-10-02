from pathlib import Path
import json,sys,hashlib
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc
j=Path(__file__).parent;p=j.parents[1];r=p/"runs/GLM-RUN-0051";a=p/"jobs/REDUCE-PD51-NATIVE-20261002T0424Z"
def ref(f):
 b=f.read_bytes();return dict(path=str(f),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
assert ref(a/"reduction.json")["sha256"]=="e24af8bc16475b9dc11c026299f3f89441d952dbe6a93724e4adac47ce7bd979"
v=json.loads((a/"reduction.json").read_text());assert v["state"]["status"]=="cancelled"and v["inference_attempts"]==0 and v["source_pins"]==18
d={}
for rank in[0,1]:
 f=a/("D"+str(rank)+".native.log");lines=f.read_text().splitlines();fatal=[(i,l)for i,l in enumerate(lines)if"ValueError: No valid cudagraph sizes after rounding to multiple of 6"in l];assert fatal
 i,line=fatal[0];prefix=line.split(" ERROR ")[0];stack=[l for l in lines[max(0,i-80):i+1]if l.startswith(prefix)]
 assert any("self.initialize_attn_backend(kv_cache_config)"in l for l in stack)and not any("initialize_kv_cache_tensors"in l for l in stack)
 weights=[l for l in lines if"Loading model weights took"in l];assert len(weights)==16
 d["D"+str(rank)]=dict(raw=ref(f),first_fatal=line,first_stack=stack,weights_complete=16,native_KV_capacity=[l for l in lines if"GPU KV cache size:"in l],OOM_observed=any("OutOfMemoryError"in l for l in lines))
 assert not d["D"+str(rank)]["OOM_observed"]
out=dict(at=utc(),run_id=r.name,verdict="REJECT",classification="native K5 FULL Graph capture defaultgeometry invalid at initialize_attn_backend before KV tensors",controller_terminal_state="cancelled",terminal_audit=ref(a/"reduction.json"),controlled_cancellation=ref(p/"jobs/CANCEL51-NATIVE-FATAL-20261002T0423Z/reduction.json"),native=d,inference_attempts=0,outputs=0,new_signals=0,new_models=0,new_requests=0,limits=["Run51Dplanned150484tokens is logical planning only; rawKVtensors not allocated due earlier Graphgeometry check","P32ready andD32weightsloaded; noactualdualKV/Graph/PDfit","Frozenreadiness failure scanner lacked ValueError; GPTidentifiednativefatal andZcode stopped exactcontroller viaexistinghandler, rawpreserved","No observedOOM does not prove lowerKV candidate solvesRun50memoryfailure"],next_candidate=dict(run_id="GLM-RUN-0052",D_capture_sizes=[6],D_max_capture=6,K5=True,seq=1,contexts=[81933,144384],KV_GiB=[.9,1.6],native_math_changes=0))
atomic_json(j/"reduction.json",out);b=json.loads((r/"reduction_brief.json").read_text());b.update(verdict="REJECT",classification=out["classification"],Graph_geometry_failure_detail=ref(j/"reduction.json"),limits=out["limits"]);atomic_json(r/"reduction_brief.json",b)
m=json.loads((r/"manifest.json").read_text())
for k in list(m):
 if k.endswith("failure_detail"):m.pop(k)
m.update(status="cancelled",valid=False,verdict="REJECT",Graph_geometry_failure_detail=ref(j/"reduction.json"));atomic_json(r/"manifest.json",m)
(r/"summary.md").write_text("# GLM-RUN-0051\n\nREJECT native K5 FULL default capture geometry withseq1: sizes[1,2,4]/max4 roundedtomultiple6 leavesnone. P32ready/nativeplanned81933; D32weightsloaded/nativeplanned150484 butinitialize_attn_backend fails BEFORE KVtensor allocation. NoOOMobserved does not resolveRun50physicalKVfit. Actualcontrollerstatuscancelled afterZcode exactcontrollerSIGTERM viaexistinghandler; readinessscannerValueError omission fixedonlynew52. Terminal18pins/native9files/HCCLenv/currentowners valid,0requests/outputs. New52explicitcapture6/max6 withnativepositive/negativeCPUcheck preservesK5/contexts/operators; actualfitunknown, CurrentNone.\n")
atomic_json(j/"result.json",dict(schema_version=1,job_id=j.name,status="completed",summary="Run51 REJECT nativeK5Graph seq1defaultcapture4 geometry beforeKVtensor allocation; controlledcontrollercancelled/18pins/0inference;52nativecapture6 candidate",execution=dict(inner_exit_code=0,acceptance="passed",processes=[]),findings=[],evidence=[dict(id="reduction",locator="both51nativefirstfatal/sourceorder/terminalaudit/cancelproof",**ref(j/"reduction.json"))],unknowns=out["limits"],decision_request=None,next_check_at=None))
print(json.dumps(dict(verdict="REJECT",reduction=ref(j/"reduction.json"))))

