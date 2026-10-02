from pathlib import Path
import json,sys,hashlib
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc
j=Path(__file__).parent;p=j.parents[1];r=p/"runs/GLM-RUN-0052";a=p/"jobs/REDUCE-PD52-NATIVE-20261002T0445Z"
def ref(f):
 b=f.read_bytes();return dict(path=str(f),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
assert ref(a/"reduction.json")["sha256"]=="4fa8bf90db05912b2727f280edd8580394861887ed6db61fe9d28f66b66b32c7"
v=json.loads((a/"reduction.json").read_text());assert v["state"]["status"]=="failed"and v["inference_attempts"]==0 and v["source_pins"]==18
d={}
for rank in[0,1]:
 f=a/("D"+str(rank)+".native.log");lines=f.read_text().splitlines();fatal=[(i,l)for i,l in enumerate(lines)if "torch.OutOfMemoryError: NPU out of memory."in l]
 stacks=[]
 for i,line in fatal:
  prefix=line.split(" ERROR ")[0];stack=[l for l in lines[max(0,i-80):i+1]if l.startswith(prefix)]
  assert any("self.drafter.initialize_attn_backend(kv_cache_config"in l for l in stack)
  assert any("self.chunked_prefill_workspace = torch.empty("in l for l in stack)
  stacks.append(dict(first_fatal=line,stack=stack))
 weights=[l for l in lines if"Loading model weights took"in l];assert len(weights)==16
 d["D"+str(rank)]=dict(raw=ref(f),OOM_stacks=stacks,weights_complete=16,native_KV_capacity=[l for l in lines if"GPU KV cache size:"in l],error_tail=[l for l in lines if"ERROR"in l][-12:])
assert d["D0"]["OOM_stacks"]
source=Path("/vllm-workspace/vllm-ascend/vllm_ascend/worker/model_runner_v1.py")
# Container source is retrieved read-only by docker; no model/worker.
import subprocess,ast
raw=subprocess.check_output(["docker","exec","glm52-single","cat",str(source)])
tree=ast.parse(raw);cls=next(n for n in tree.body if isinstance(n,ast.ClassDef)and n.name=="NPUModelRunner");method=next(n for n in cls.body if isinstance(n,ast.FunctionDef)and n.name=="initialize_kv_cache")
calls=[n for n in ast.walk(method)if isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)]
kv=next(n for n in calls if n.func.attr=="initialize_kv_cache_tensors")
draft=next(n for n in calls if n.func.attr=="initialize_attn_backend"and isinstance(n.func.value,ast.Attribute)and n.func.value.attr=="drafter")
assert kv.lineno<draft.lineno
out=dict(at=utc(),run_id=r.name,verdict="REJECT",classification="D MTP drafter SFA metadata workspace allocation OOM after main KV tensors, before Graph capture",controller_terminal_state="failed",terminal_audit=ref(a/"reduction.json"),native=d,native_source_order=dict(path=str(source),sha256=hashlib.sha256(raw).hexdigest(),main_KV_initialize_lineno=kv.lineno,drafter_metadata_initialize_lineno=draft.lineno),inference_attempts=0,outputs=0,new_signals=0,new_models=0,new_requests=0,correction="Earlier GPT first-tail interpretation before KV was wrong; full stack places failure at drafter metadata AFTER main KV initialization. Run51 geometry failure really was before KV.",limits=["Actual main KV initialization reached before drafter workspace OOM, not a successful dual service/Graph/runtime fit certificate","D0 observed native 154MiB workspace OOM on TP0 and TP2; D1 raw has no OutOfMemoryError, do not attribute peer failure as a second observed OOM","Nonzero free memory does not prove allocator/physical fragmentation cause or board capacity bound","P32ready/D32weightsloaded/plannedP81933 D150484 per DP; no native inference/output or transport proof","Task-only unused inherited SFA MLA workspace guard is a candidate; source/CPU proof and actual E2E required"],next_candidate=dict(run_id="GLM-RUN-0053",task_only_unused_SFA_workspace=True,D_capture_sizes=[6],D_max_capture=6,KV_GiB=[.9,1.6],native_math_changes=0))
atomic_json(j/"reduction.json",out);b=json.loads((r/"reduction_brief.json").read_text());b.update(verdict="REJECT",classification=out["classification"],SFA_workspace_failure_detail=ref(j/"reduction.json"),limits=out["limits"]);atomic_json(r/"reduction_brief.json",b)
m=json.loads((r/"manifest.json").read_text())
for k in list(m):
 if k.endswith("failure_detail"):m.pop(k)
m.update(status="failed",valid=False,verdict="REJECT",SFA_workspace_failure_detail=ref(j/"reduction.json"));atomic_json(r/"manifest.json",m)
(r/"summary.md").write_text("# GLM-RUN-0052\n\nREJECT D MTP drafter SFA metadata workspace allocation: native154MiB OOM AFTER main KV tensor initialization, before Graph capture. P32ready/planned81933; D32weightsloaded/planned150484. D0TP0/TP2 observed OOM/free609–612MiB; D1peer log does not show its own OOM. Fullstack/sourceorder correct GPT earlier first-tail preKV interpretation. Terminal18pins/native9/HCCLenv/currentowner auditVALID,0requests/outputs. No allocator cause/hardware bound/dualready/Graph/runtime/PD proof. New53task-only unused inheritedSFAworkspace guard candidate; ordinaryMLA nativebuffer protected. CurrentNone.\n")
atomic_json(j/"result.json",dict(schema_version=1,job_id=j.name,status="completed",summary="Run52 REJECT native D0 drafter SFA workspace154MiB OOM AFTER mainKV tensors/beforeGraph; D1 peer has no own OOM,18pins/0inference; task SFA unusedbuffer candidate",execution=dict(inner_exit_code=0,acceptance="passed",processes=[]),findings=[],evidence=[dict(id="reduction",locator="52D0nativeOOM/peerabsence/sourceorder/terminalaudit/correction",**ref(j/"reduction.json"))],unknowns=out["limits"],decision_request=None,next_check_at=None))
print(json.dumps(dict(verdict="REJECT",reduction=ref(j/"reduction.json"))))

