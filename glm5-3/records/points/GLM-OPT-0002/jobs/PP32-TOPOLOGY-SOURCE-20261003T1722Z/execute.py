from pathlib import Path
import json,hashlib,subprocess,sys,ast
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc
j=Path(__file__).parent
paths=["/vllm-workspace/vllm-ascend/docs/source/user_guide/feature_guide/pipeline_parallel.md","/vllm-workspace/vllm/vllm/v1/executor/multiproc_executor.py","/vllm-workspace/vllm/vllm/entrypoints/cli/serve.py","/vllm-workspace/vllm/vllm/engine/arg_utils.py","/vllm-workspace/vllm/vllm/config/parallel.py"]
code="from pathlib import Path;import json;print(json.dumps([dict(path=n,content=Path(n).read_text())for n in "+repr(paths)+"]))"
z=subprocess.run(["docker","exec","glm52-single","python3","-c",code],capture_output=True,timeout=90)
(j/"read.stdout").write_bytes(z.stdout);(j/"read.stderr").write_bytes(z.stderr);z.check_returncode()
refs=[];excerpts=[]
for v in json.loads(z.stdout):
 p=Path(v["path"]);snapshot=j/("source_"+p.name);snapshot.write_text(v["content"]);raw=snapshot.read_bytes()
 refs.append(dict(path=v["path"],snapshot=str(snapshot),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
 if p.suffix==".py":ast.parse(v["content"])
 lines=v["content"].splitlines()
 selected=[]
 for i,l in enumerate(lines):
  if any(k in l for k in["nnodes","headless","node_rank","node-rank","local_world_size","tensor_parallel_size %","world_size //","world_size//","pipeline_parallel_size","master_addr","master_port"]):
   selected.append(dict(line=i+1,text="\n".join(lines[max(0,i-2):min(len(lines),i+4)])))
 excerpts.append(dict(path=v["path"],matches=selected[:90]))
out=dict(at=utc(),kind="source_only_multi_node_MP_PP32_feasibility",valid=True,proposal=dict(TP=8,PP=4,DCP=8,PCP=1,DP=1,nnodes=2,world=32,local_world_each=16,PP_partition="22,20,20,16",K=3,no_EP=True,no_native_KV=True,native_operators_unchanged=True),sources=refs,excerpts=excerpts,process_operations=0,inference=0,models=0,NPU_tensors=0,SDK_communicators=0,limits=["Installed native docs permit multi-node mp PP with node0 frontend and node1headless; actual fullCLI/rank layout/GLM MTP support must be verified, no fit or performance claim","One shared32-rank engine is a different resource category from two independent16-rank replicas; contextcold/decode latency plus admission/capacity/physicalfault/STORE/protocol effects require realE2E if chosen","No service/model/worker operations; no oldqueue replay; preserve authoritative runtime/functional evidence"])
atomic_json(j/"reduction.json",out);b=(j/"reduction.json").read_bytes()
atomic_json(j/"result.json",dict(schema_version=1,job_id=j.name,status="completed",summary="Source-only native MP multi-node PP32 controls/docs and rank-layout identities captured; fullCLI/fit/service-capacity pending",execution=dict(inner_exit_code=0,acceptance="passed",processes=[]),findings=[],evidence=[dict(id="reduction",path=str(j/"reduction.json"),bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),locator="native installed source topology/multi-node/headless/rank layout")],unknowns=out["limits"],decision_request=None,next_check_at=None))
print(json.dumps(dict(valid=True,sources=refs,proposal=out["proposal"])))
