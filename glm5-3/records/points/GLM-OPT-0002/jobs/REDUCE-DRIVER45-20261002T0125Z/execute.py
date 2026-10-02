from pathlib import Path
import json,sys,hashlib,subprocess,shlex,re
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,same_process,utc
j=Path(__file__).parent;r=j.parents[1]/"runs/GLM-RUN-0045"
def ref(p):
 raw=p.read_bytes();return {"path":str(p),"bytes":len(raw),"sha256":hashlib.sha256(raw).hexdigest()}
s=json.loads((r/"state.json").read_text());assert s["status"]=="failed"and s["failure_phase"]=="deploy"and not same_process(s["owner"])
spec=json.loads((r/"controller_spec.json").read_text())
for x in spec["stages"][0]["sources"]:assert hashlib.sha256(Path(x["path"]).read_bytes()).hexdigest()==x["sha256"]
events=json.loads((r/"deployment_events.json").read_text());assert not any(x["event"]=="physical_node_started"for x in events);stops=[x for x in events if x["event"]=="exact_audited_failed_Run44_cohort_jointly_stopped"];assert len(stops)==1
cpu=[x for x in events if x.get("event")=="command"and x.get("exit_code")!=0];assert len(cpu)==1 and "full_config_tp32_dcp1_run45.py"in str(cpu[0]["argv"])and "AssertionError"in cpu[0]["stderr"]
current={}
for node in ["166","167"]:
 argv=["docker","top","glm52-single","-eo","pid,ppid,comm,args"]
 if node=="167":argv=["ssh","-o","BatchMode=yes","root@172.16.10.167",shlex.join(argv)]
 raw=subprocess.check_output(argv,timeout=30);path=j/(node+".processes");path.write_bytes(raw);assert not any("/bin/vllm serve "in x or"VLLM::"in x or"bishengir-compile "in x for x in raw.decode().splitlines())
 current[node]=ref(path)
assert not any((r/name).exists()for name in ["tools.phase.json","responses.phase.json","prepare.phase.json"])
out={"run_id":r.name,"diagnostic_evidence_valid":True,"verdict":"INVALID","reason":"GPT rollback CPU probe sets enable_reduce_sample=False but retains assert enable_reduce_sampleTrue; native EngineArgs config itself accepted, helper assertion failed before anynewnative launch/inference","inference_attempts":0,"new_native_starts":0,"old_Run44_cohort_cleanup_confirmed":True,"cleanup_signals":"Exact audited old44 root cohorts and verified descendants only; actualstoprecords retained, not counted aszero signals","current_native_absence":current,"cpu_failure":cpu[0],"spec_sha256":s["spec_sha256"],"source_pins":len(spec["stages"][0]["sources"]),"events":ref(r/"deployment_events.json"),"limits":["DriverpreflightINVALID, no native fit/state/E2E verdict","Old44exactcohortcleaned byauthorized45controller beforeCPUassert failure; no oldqueue replay","No operator/vendor changes/no hardwarelimit"]}
atomic_json(r/"reduction.json",out);m=json.loads((r/"manifest.json").read_text());m.update(status="failed",verdict="INVALID",valid=False,reduction=ref(r/"reduction.json"),results=out);atomic_json(r/"manifest.json",m)
(r/"summary.md").write_text("# "+r.name+"\n\nINVALID driver CPUpreflight: configuration reduce-samplingFalse but leftoverassertTrue. Exactold44nativecohortjointcleanup confirmed; no newnativeentry/inference/Responses/phase ran. Bothhosts native/compilers absent. Frozen23pins/spec/source/raw retained. NewRunrequiredforcorrectedhelper, no nativefit/state/limit verdict.\n")
atomic_json(j/"result.json",{"schema_version":1,"job_id":j.name,"status":"completed","summary":"Run45 driverCPUassert INVALID, old44exactcohortcleanup confirmed,0newnativeentry/inference; bothhosts nativeabsenceverified","execution":{"inner_exit_code":0,"acceptance":"passed","processes":[]},"findings":[],"evidence":[{"id":"reduction",**ref(r/"reduction.json"),"locator":"actualCPUstderr/source23pins/authorizedoldcleanup/currentbothnativeabsence"}],"unknowns":out["limits"],"decision_request":None,"next_check_at":None});print(json.dumps(ref(r/"reduction.json")))
