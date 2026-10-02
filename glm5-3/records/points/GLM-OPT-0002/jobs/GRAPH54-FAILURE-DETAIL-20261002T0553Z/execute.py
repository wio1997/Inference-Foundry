from pathlib import Path
import json,sys,hashlib,subprocess,shlex
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc
j=Path(__file__).parent;p=j.parents[1];r=p/"runs/GLM-RUN-0054";a=p/"jobs/REDUCE-PD54-NATIVE-20261002T0547Z"
def ref(f):
 b=f.read_bytes();return dict(path=str(f),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
assert ref(a/"reduction.json")["sha256"]=="2f20483fe6655f6e2bad39434c39e5fbdb55abd006b594f94621d49bf10f5b8d"
v=json.loads((a/"reduction.json").read_text());assert v["state"]["status"]=="failed"and v["inference_attempts"]==0 and v["source_pins"]==20
d={};counts={}
for key in["P0","P1","D0","D1"]:
 f=a/(key+".native.log");ls=f.read_text().splitlines();skips=[json.loads(l.split("GLM_UNUSED_SFA_WORKSPACE_SKIPPED ",1)[1])for l in ls if"GLM_UNUSED_SFA_WORKSPACE_SKIPPED "in l];assert len(skips)==32 and len({x["pid"]for x in skips})==16 and all(x["actual_elements"]==0 for x in skips)
 counts[key]=dict(workers=16,builders=32,zero_workspace_builders=32,nominal_bytes_by_builder=sorted({x["nominal_bytes"]for x in skips}),native_log=ref(f),physical_savings_unknown=True)
 if key.startswith("P"):continue
 fatals=[(i,l)for i,l in enumerate(ls)if "torch.OutOfMemoryError:"in l];stack=[]
 if fatals:
  i,line=fatals[0];prefix=line.split(" ERROR ")[0];stack=[l for l in ls[max(0,i-145):i+17]if l.startswith(prefix)]
  assert any("self._warmup_and_capture("in l for l in stack)and any("self._dummy_run("in l for l in stack)and any("capture_model"in l for l in stack)
 d[key]=dict(raw=ref(f),first_torch_OOM=fatals[0][1]if fatals else None,first_stack=stack,HCCL_or_runtime_errors=[l for l in ls if"Failed to allocate memory requested by"in l][:8],KV_capacity=[l for l in ls if"GPU KV cache size:"in l])
assert d["D0"]["first_torch_OOM"]
owners=json.loads((r/"startup_model_identities.json").read_text());envs={}
for key,o in owners.items():
 code="from pathlib import Path;import json;p=Path('/proc/"+str(o["pid"])+"');out=dict(present=p.exists())\nif p.exists():\n s=(p/'stat').read_text();assert s[s.rfind(')')+2:].split()[19]=="+repr(o["identity"]["start_ticks"])+";e=dict(x.decode().split('=',1)for x in(p/'environ').read_bytes().split(bytes([0]))if b'='in x);out['environment']={k:e.get(k)for k in ['HCCL_BUFFSIZE','HCCL_NPU_SOCKET_PORT_RANGE','HCCL_HOST_SOCKET_PORT_RANGE']}\nprint(json.dumps(out))"
 argv=["python3","-c",code]
 if o["host"]=="167":argv=["ssh","-o","BatchMode=yes","root@172.16.10.167",shlex.join(argv)]
 envs[key]=json.loads(subprocess.check_output(argv,timeout=30))
 if envs[key]["present"]:assert envs[key]["environment"]["HCCL_BUFFSIZE"]=="768"
out=dict(at=utc(),run_id=r.name,verdict="REJECT",classification="Native K5 Graph capture warmup dummy execution fails with asynchronous HCCL/RUNTIME memory allocation OOM after all SFA inheritedworkspaces removed",terminal_audit=ref(a/"reduction.json"),workspace_receipt_counts=counts,D_native=d,actual_API_environment=envs,inference_attempts=0,outputs=0,new_signals=0,new_models=0,new_requests=0,limits=["Native asynchronous error names current HcclAlltoAll but stack points batch_matmul_transpose; exact failing device allocation/size/domain remains unknown, no operator-root-cause assertion","64actualworkers/128builders allzero buffers: conditional source-controlled startup mechanism exercised, not full functional/Graph or isolatedphysicalmemory saving proof","D1 peer no own torch.OutOfMemoryError in frozen audit; no doubleOOM claim","HCCL_BUFFSIZE768MB and ordinary PG per-group fixed200MB must be distinguished; installed utils exempts mc2 pg_options to use env","Communication buffer size reduction is standard native configuration candidate, performance/actualallocation/dualfit/capacity unproved"],next_candidate=dict(run_id="GLM-RUN-0055",HCCL_BUFFSIZE_MB=256,ordinary_native_PG_buffer_options_unchanged=True,KV_GiB=[.9,1.6],same_K5FULLcapture6=True,native_math_changes=0))
atomic_json(j/"reduction.json",out);b=json.loads((r/"reduction_brief.json").read_text());b.update(verdict="REJECT",classification=out["classification"],Graph_warmup_failure_detail=ref(j/"reduction.json"),limits=out["limits"]);atomic_json(r/"reduction_brief.json",b)
m=json.loads((r/"manifest.json").read_text());m.update(status="failed",valid=False,verdict="REJECT",Graph_warmup_failure_detail=ref(j/"reduction.json"));atomic_json(r/"manifest.json",m)
(r/"summary.md").write_text("# GLM-RUN-0054\n\nREJECT nativeD K5FULL6 Graph capture warmup dummy-run HCCL/RUNTIME allocation OOM; firstD0torchOOM05:42:32, asynchronouscurrentopHcclAlltoAll/stackbatch_matmul_transpose doesnot isolateexactallocation/domain. 64workers/128SFA builders actually0inheritedworkspace; P32ready andD32weights/KV/metadata advancedpast52drafterbuffer failure. 20pins/native9/owners/env auditVALID,0inference/output; nofullGraph/PD/hardwarebound. HCCL_BUFFSIZEactual768 vsordinaryPGfixed200, next55nativeenv256 preservesPGoptions/operators/KV/layout, source/sourceCPU/E2E decide. CurrentNone.\n")
atomic_json(j/"result.json",dict(schema_version=1,job_id=j.name,status="completed",summary="Run54 REJECT nativeGraph warmup asyncHCCL/RUNTIME memoryOOM,64worker/128builders zeroSFA workspace;20pins/0inference; nativeenv768 vsfixedPG200 distinguished",execution=dict(inner_exit_code=0,acceptance="passed",processes=[]),findings=[],evidence=[dict(id="reduction",locator="nativefullGraphwarmupstack/128zeroWorkspaceReceipts/actualHCCLenv",**ref(j/"reduction.json"))],unknowns=out["limits"],decision_request=None,next_check_at=None));print(json.dumps(dict(verdict="REJECT",reduction=ref(j/"reduction.json"))))

