from pathlib import Path
import json,hashlib,sys,subprocess,re,ast
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc
j=Path(__file__).parent;p=j.parents[1];r=p/"runs/GLM-RUN-0058"
def ref(f):
 b=f.read_bytes();return dict(path=str(f),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
audit=p/"jobs/REDUCE-PD58-NATIVE-20261002T0747Z/reduction.json";assert ref(audit)["sha256"]=="07c809c1a36909a6582220b0489900c570ed0d3b4babc60331ac1a26e52cc77d"
a=json.loads(audit.read_text());assert a["measurement_valid"] and a["inference_attempts"]==3 and a["completed_native_requests"]==2 and a["effective_public_output_tokens"]==64
registered=json.loads((r/"registered_P_engine_identities.json").read_text());assert all(v["num_blocks"]==46 for v in registered.values())
rows={}
for key in ["P0","P1","D0","D1"]:
 f=audit.parent/(key+".native.log");ls=f.read_text().splitlines();hits=[i for i,l in enumerate(ls)if "torch.OutOfMemoryError:"in l]
 per={}
 for i in hits:
  m=re.search(r"\(Worker_[^ ]+ pid=(\d+)\)",ls[i])
  if m and m[1]not in per:per[m[1]]=dict(line=i+1,error=ls[i],stack=ls[max(0,i-100):i+1])
 rows[key]=dict(raw=ref(f),OOM_workers=len(per),errors=list(per.values()),Graph_finished=[l for l in ls if "Graph capturing finished"in l])
assert rows["P0"]["OOM_workers"] and any("114.00 MiB"in x["error"]for x in rows["P0"]["errors"]);assert not rows["D0"]["OOM_workers"]and not rows["D1"]["OOM_workers"]
native_paths=["/vllm-workspace/vllm-ascend/vllm_ascend/attention/context_parallel/sfa_cp.py","/vllm-workspace/vllm-ascend/vllm_ascend/device/device_op.py","/vllm-workspace/vllm-ascend/csrc/attention/sparse_flash_attention/sparse_flash_attention_torch_adpt.h"]
sources=[]
for n,path in enumerate(native_paths):
 z=subprocess.run(["docker","exec","glm52-single","cat",path],capture_output=True,check=True);f=j/(str(n)+"_"+Path(path).name);f.write_bytes(z.stdout);sources.append(dict(native_path=path,**ref(f)))
header=(j/"2_sparse_flash_attention_torch_adpt.h").read_text();assert "construct_sparse_flash_attention_output_tensor"in header and "EXEC_NPU_CMD("in header
limits=["Run58 firstactualP canonical request advancespast41block admission, then nativeSFA OOM114MiB on recordedworkers; no completedP/PD/nativequality/capacity credit","Python op stack cannot distinguish outputtensor vs internal ACL workspace allocation; C++adapter allocatesoutput and callsnativeACLop, no nativeoperator modification","Reportedfree memory >114MiB is not proof of contiguous/page/domain available allocation; actualfailure domain/fragmentation/peak isolatedcause unknown","P96MB HCCL decrement and chunk1024→256 candidate not isolatedgain or fixedworkspace scaling proof; nativewindow floor408MB observedmaxBs1 only","RetainD56/source/operators and P1GiB46blocks; newP reload/newpilot required; nooldqueue"]
out=dict(at=utc(),run_id=r.name,verdict="REJECT",classification="FirstcanonicalP compute native SFA allocationOOM114MiB, client240s timeout/zero output; native46block pool admission no longer firstfailure",terminal_audit=ref(audit),registered_P=ref(r/"registered_P_engine_identities.json"),logs=rows,installed_sources=sources,inference_attempts=3,completed_native_requests=2,effective_public_outputs=64,job_signals=0,new_requests=0,new_models=0,limits=limits,next_candidate=dict(run_id="GLM-RUN-0059",retain_D56=True,P_KV_bytes=1073741824,P_gpu_memory_utilization=.46,P_HCCL_MB=416,D_HCCL_MB=512,P_batch=256,D_batch=1024,diagnostic_P_timeout_s=600,performance_comparison="new deployment candidate, not isolated gain"))
atomic_json(j/"reduction.json",out)
b=json.loads((r/"reduction_brief.json").read_text());b.update(verdict="REJECT",classification=out["classification"],SFA_runtime_failure_detail=ref(j/"reduction.json"),limits=limits);atomic_json(r/"reduction_brief.json",b)
m=json.loads((r/"manifest.json").read_text());m.update(verdict="REJECT",valid=False,SFA_runtime_failure_detail=ref(j/"reduction.json"));atomic_json(r/"manifest.json",m)
(r/"summary.md").write_text("# GLM-RUN-0058\n\nREJECT firstcanonicalP runtime: resident dual64workers/retained D56Graph, Pnative46blocks/32registeredIDs verified;2Dshort64effective output. Pfirst81932helper executesnativeSFA then114MiBOOM, client240s timeout/zero wire/output; noPD. Allocation output/workspace/domain/fragmentation unknown. Candidate59 keepP1GiB/GMU.46/D56, lowerP HCCL512→416 (observed408 conditionalfloor), batch1024→256; newconfig notisolatedgain. FreshP-only exactcleanup/newpilot, nooldqueue.\n")
atomic_json(j/"result.json",dict(schema_version=1,job_id=j.name,status="completed",summary="Run58 REJECT actualP SFA114MiBOOM after46pool admission; 2D64outputs retained; next P416MB buffer/chunk256 retainedD candidate",execution=dict(inner_exit_code=0,acceptance="passed",processes=[]),findings=[],evidence=[dict(id="reduction",locator="actualnativefirstOOMworkers/fullstack/source32/registered46pool/partialwire/64outputs",**ref(j/"reduction.json"))],unknowns=limits,decision_request=None,next_check_at=None));print(json.dumps(ref(j/"reduction.json")))

