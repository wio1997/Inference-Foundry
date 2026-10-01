import json,hashlib,sys,re
from pathlib import Path
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,same_process,utc
j=Path(__file__).parent;p=j.parents[1];r=p/"runs/GLM-RUN-0036";state=json.loads((r/"state.json").read_text());phase=json.loads((r/"deploy.phase.json").read_text())
assert state["status"]=="failed" and state["completed_stages"]==[] and state["failure_phase"]=="deploy" and not same_process(state["owner"])
assert phase["status"]=="failed" and phase["exit_code"]==1 and not phase["timed_out"]
assert "native TP32 readiness deadline; preserve raw" in (r/"deploy.log").read_text()[-1500:]
spec=json.loads((r/"controller_spec.json").read_text())
for pin in spec["stages"][0]["sources"]:assert hashlib.sha256(Path(pin["path"]).read_bytes()).hexdigest()==pin["sha256"]
assert not list(r.glob("**/load_result.json")) and not list(r.glob("**/*.wire")) and not list(r.glob("**/*.events.jsonl"))
def ref(path):
 raw=path.read_bytes();return {"path":str(path),"bytes":len(raw),"sha256":hashlib.sha256(raw).hexdigest()}
jobs={}
for name in ["TP32-GRAPH-DIAG-20261001T2210Z-v3","TP32-COMPILER-STATE-20261001T2225Z","TP32-DCP16-CONFIG-20261001T2220Z"]:
 d=p/"jobs"/name;bridge=json.loads((d/"bridge.json").read_text());assert bridge["cli_exit_code"]==0 and not bridge["timed_out"]
 result=json.loads((d/"result.json").read_text());assert result["execution"]["inner_exit_code"]==0 and result["execution"]["acceptance"]=="passed"
 jobs[name]={"bridge":ref(d/"bridge.json"),"result":ref(d/"result.json"),"reduction":ref(d/("config_reduction.json" if name.startswith("TP32-DCP16") else "reduction.json"))}
native={}
for node in ["166","167"]:
 log=r.parent/"GLM-RUN-0037"/("prior_"+node+".log")
 assert log.exists()
 text=log.read_text()
 native[node]={"log_at_Run37_before_jointcleanup":ref(log),"warmup_lines":[l for l in text.splitlines()if "Warmup for Triton" in l][-3:],"weight_lines":[l for l in text.splitlines()if "Model loading" in l or "Model weights" in l][-20:],"graph_tail":[l for l in text.splitlines()if "Capturing" in l][-3:],"native_fatal_lines":[l for l in text.splitlines()if "OutOfMemoryError" in l or "Engine core initialization failed" in l][-10:]}
out={"run_id":r.name,"at":utc(),"diagnostic_evidence_valid":True,"measurement_valid":False,"verdict":"INCONCLUSIVE","terminal_state":state,"deploy_phase":phase,"sourcepins_verified":len(spec["stages"][0]["sources"]),"inference_attempts":0,"completed_requests":0,"effective_output_tokens":0,"startup_owners":json.loads((r/"startup_model_identities.json").read_text()),"native":native,"jobs":jobs,"facts":["Both native weights/HCCL/Triton warmup reached first FULL decode graph capture","Actual nonblocking Python stacks on native workers show subprocess.run waiting for fused_sfa_dcp_lse_combine R32 Triton compilation","16 active native bishengir compilers perhost sampled CPU ticks, approximately one CPUcore each, unchanged native IR retained","Native40min readiness window exhausted; no APIready or inference issued","NativeTP32DCP16 explicitly rejected by installed SFA equalDCPandTP assertion; no guard bypass"],"limits":["CPU consumption does not prove forwardprogress, finite eventualcompiletime, device deadlock or hardwarecapacitylimit","Native startup logs captured afterdeadline and before Run37 jointlycleanup; diagnostic raw snapshots fixed at observedtimestamps","Graphdiag rawindex includes mutable CLI stdout/stderr sampled beforeCLIcompletion; do not treat those sampledzero hashes as finalCLIhash","No functional/performance/KEEP; DP4TP8DCP8EP32 nextlayout configaccepted only until actual nativeE2E"]}
atomic_json(j/"reduction.json",out);atomic_json(r/"reduction.json",out)
m=json.loads((r/"manifest.json").read_text());m.update(status="failed",valid=False,verdict="INCONCLUSIVE",diagnostic_evidence_valid=True,completed_at=phase["finished_at"],new_client_attempts=0,new_completed_inference_requests=0,new_effective_output_tokens=0,failure_phase="deploy",reduction=ref(r/"reduction.json"));atomic_json(r/"manifest.json",m)
(r/"summary.md").write_text("# "+r.name+"\n\nNative DP1/TP32/DCP32/EP32 first FULL decode Graph capture did not complete within original40min native readiness window, terminal22:37:54Z; zero inference attempts/completed/output. Weights/native communication/warmup reached capture. Nonblocking native Python stacks and CPU tick samples identify R32 fused SFA Triton compiler subprocesses, 16perhost with active CPU. No operator/source/compiler option edits. INCONCLUSIVE: CPU activity is not forwardprogress or eventual compile proof, and startup deadline is not hardware/service capacity limit. Exact old group ownership and native logs preserved by new unique Run37 controller before task-only jointcleanup. Next acceptedDP4TP8DCP8 layout reduces DCP group and expands schedulers; nativeE2E pending. TP32DCP16 rejected by native SFA equal-size assertion, no bypass. Raw snapshots/IR retained server; sampled mutable CLI index limits explicit in reduction.\n")
job=json.loads((j/"job.json").read_text());ev={"id":"reduction",**ref(j/"reduction.json"),"locator":"zero inference, exact source/state and observed native CPUcompiler evidence"}
atomic_json(job["result"]["path"],{"schema_version":1,"job_id":job["job_id"],"status":"completed","summary":"Run36 original native readiness deadline audited, zero inference, CPUcompiler stack/IR evidence scoped, no hardwarecapacity claim","execution":{"inner_exit_code":0,"acceptance":"passed","processes":[]},"findings":[],"evidence":[ev],"unknowns":out["limits"],"decision_request":None,"next_check_at":None})
print(json.dumps({"diagnostic_evidence_valid":True,"measurement_valid":False,"new_requests":0,"reduction":ref(j/"reduction.json")}))
