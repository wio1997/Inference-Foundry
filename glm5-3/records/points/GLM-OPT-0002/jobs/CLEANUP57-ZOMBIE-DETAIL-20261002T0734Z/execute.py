from pathlib import Path
import json,sys,hashlib,subprocess,shlex
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc,same_process
j=Path(__file__).parent;p=j.parents[1];r=p/"runs/GLM-RUN-0057"
def ref(f):
 b=f.read_bytes();return dict(path=str(f),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
audit=p/"jobs/REDUCE-PD57-NATIVE-20261002T0722Z/reduction.json"
assert ref(audit)["sha256"]=="161b95b455f7fbe1a37c86a798f2a199b11d907215131611856760fde97bbb22"
a=json.loads(audit.read_text());assert a["state"]["status"]=="failed" and a["state"]["failure_phase"]=="deploy" and a["inference_attempts"]==0
assert not same_process(a["state"]["owner"])
prior=json.loads((r/"prior_model_owners.json").read_text());rows={}
for node in ["166","167"]:
 checker=j/"reconcile.py";snapshot=r/("prior_"+node+".owner_preflight.json")
 if node=="167":
  for f in [checker,snapshot]:subprocess.run(["scp","-q",str(f),"root@172.16.10.167:"+str(j/f.name)],check=True)
 args=["env","GLM_EXPECTED_COHORT="+json.dumps([o for o in prior.values()if o["host"]==node]),"GLM_PRIOR_SNAPSHOT="+str(snapshot if node=="166"else j/snapshot.name),"python3",str(checker)]
 if node=="167":args=["ssh","-o","BatchMode=yes","root@172.16.10.167",shlex.join(args)]
 z=subprocess.run(args,capture_output=True,timeout=100);(j/(node+".stdout")).write_bytes(z.stdout);(j/(node+".stderr")).write_bytes(z.stderr);z.check_returncode();rows[node]=json.loads(z.stdout)
native="/vllm-workspace/vllm-ascend/vllm_ascend/worker/worker.py"
z=subprocess.run(["docker","exec","glm52-single","cat",native],capture_output=True,check=True);(j/"native_worker.py").write_bytes(z.stdout)
s=z.stdout.decode();assert "self.init_snapshot.free_memory < self.requested_memory"in s and "if kv_cache_memory_bytes := self.cache_config.kv_cache_memory_bytes:"in s and "This does not respect the gpu_memory_utilization config."in s
assert not any(e["event"]=="native_role_started"for e in json.loads((r/"deployment_events.json").read_text()))
limits=["Run57 sent SIGTERM to exact prior P roots and SIGKILL fallback to exact verified P member identities; failed before complete per-signal journal, no precise fallback count claim","Audit signals=0 describes auditor actions, not prior deploy; original source/raw frozen","640 observed oldP zombies are inactive/released NPU; reaper ownership unresolved and no parent/container/D signal authorized by this job","P1GiB/GMU.46 pending native GPU fit and new E2E; explicitKV bypasses native memory profiling but startup free-memory gate remains","No hardware capacity/allocator-domain/throughput/KEEP claim"]
out=dict(at=utc(),run_id=r.name,verdict="INVALID",classification="GPT cleanup checker required /proc identity disappearance; stopped P zombies preserve identity and caused false live assertion before any new model/inference",terminal_audit=ref(audit),reconciliation={n:dict(raw=ref(j/(n+".stdout")),P_absent=x["P_absent"],P_zombies=x["P_zombies"],P_active=x["P_active"],D_members=len(x["D_states"]),D_every_identity_retained=x["D_every_identity_retained"],NPU_workers=len(x["npu_D_worker_pids"]))for n,x in rows.items()},native_worker=dict(native_path=native,**ref(j/"native_worker.py")),new_models=0,inference_attempts=0,effective_outputs=0,job_signals=0,limits=limits,next_candidate=dict(run_id="GLM-RUN-0058",P_KV_bytes=1073741824,P_gpu_memory_utilization=.46,retain_D56=True,cleanup="verification_only/no signals"))
atomic_json(j/"reduction.json",out)
b=json.loads((r/"reduction_brief.json").read_text());b.update(verdict="INVALID",measurement_valid=False,classification=out["classification"],cleanup_failure_detail=ref(j/"reduction.json"),limits=limits);atomic_json(r/"reduction_brief.json",b)
m=json.loads((r/"manifest.json").read_text());m.update(verdict="INVALID",valid=False,cleanup_failure_detail=ref(j/"reduction.json"));atomic_json(r/"manifest.json",m)
(r/"summary.md").write_text("# GLM-RUN-0057\n\nINVALID cleanup verification: exact oldP stopped by prior task signals, 640 remaining zombies retain /proc identity; no activeP/NPU/PAPI. D56 647 exactmembers/32NPUworkers retained. 29pins/fourCLI passed; zero newmodels/requests. Complete fallback signal journal unavailable; auditor signals0 means audit itself. Next58 verify-only reconcile/no signals, P1GiB/GMU.46 retainedD; explicitKV/nativeinit gate source captured. No GPU candidate failure or capacity claim.\n")
atomic_json(j/"result.json",dict(schema_version=1,job_id=j.name,status="completed",summary="Run57 INVALID zombie identity checker; oldP0active/640Z, D647exactmembers retained; zero newmodels/requests; next58 verifier/no signals P1GiB GMU.46",execution=dict(inner_exit_code=0,acceptance="passed",processes=[]),findings=[],evidence=[dict(id="reduction",locator="exactfrozenP/D members states/NPU/nativeinit explicitKV source/terminalaudit",**ref(j/"reduction.json"))],unknowns=limits,decision_request=None,next_check_at=None))
print(json.dumps(ref(j/"reduction.json")))
