import json,hashlib,sys,re,subprocess
from pathlib import Path
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,same_process,utc
j=Path(__file__).parent;p=j.parents[1];r=p/"runs/GLM-RUN-0040";state=json.loads((r/"state.json").read_text());phase=json.loads((r/"deploy.phase.json").read_text())
assert state["status"]=="failed" and state["completed_stages"]==[] and state["failure_phase"]=="deploy" and not same_process(state["owner"])
assert phase["status"]=="failed" and phase["exit_code"]==1 and not phase["timed_out"]
spec=json.loads((r/"controller_spec.json").read_text())
for pin in spec["stages"][0]["sources"]:assert hashlib.sha256(Path(pin["path"]).read_bytes()).hexdigest()==pin["sha256"]
assert not list(r.glob("**/load_result.json")) and not list(r.glob("**/*.wire")) and not list(r.glob("**/*.events.jsonl"))
def ref(path):
 raw=path.read_bytes();return {"path":str(path),"bytes":len(raw),"sha256":hashlib.sha256(raw).hexdigest()}
def command(node,argv):
 if node=="167":
  import shlex
  argv=["ssh","-o","BatchMode=yes","root@172.16.10.167",shlex.join(argv)]
 return subprocess.check_output(argv,timeout=30)
native={}
for rank,node in enumerate(["166","167"]):
 raw=command(node,["cat",f"/data/tiankuan/wio/glm52-pd/deploy/logs/TP_run40_{rank}.log"]);log=r/f"native_{node}.log";log.write_bytes(raw);a=raw.decode(errors="replace").splitlines();i=next(i for i,x in enumerate(a)if "WorkerProc failed to start."in x)
 top=command(node,["docker","top","glm52-single","-eo","pid,ppid,comm,args"]);(r/f"terminal_{node}.processes").write_bytes(top)
 assert not any("/bin/vllm serve "in x or "VLLM::"in x or "bishengir-compile "in x for x in top.decode().splitlines())
 sockets=command(node,["ss","-ltnp"]);(r/f"terminal_{node}.sockets").write_bytes(sockets)
 assert not re.search(r":(?:32620|32621|32630|9081|9082|9900|9901|8000|8002)\s",sockets.decode())
 native[node]={"log":ref(log),"first_error_line":i+1,"first_error_context":a[i:i+30],"terminal_processes":ref(r/f"terminal_{node}.processes"),"terminal_sockets":ref(r/f"terminal_{node}.sockets"),"native_workers_remaining":0,"weights_lines":[x for x in a if "Loading model weights took"in x]}
assert any("zmq.error.ZMQError: Address already in use"in x and "55557"in x for x in native["167"]["first_error_context"])
out={"run_id":r.name,"at":utc(),"diagnostic_evidence_valid":True,"measurement_valid":False,"verdict":"INCONCLUSIVE","terminal_state":state,"deploy_phase":phase,"sourcepins_verified":len(spec["stages"][0]["sources"]),"inference_attempts":0,"completed_requests":0,"effective_output_tokens":0,"startup_owners":json.loads((r/"startup_model_identities.json").read_text()),"native":native,"facts":["Earliest observed worker error on167 is ZMQ TCP55557 bind Address already in use in native MessageQueue constructor","Later observed Gloo peer-closed errors are cascading after that bind failure","Both native roots/workers have exited without new audit-issued model signals; only unrelated existing shell/sleep remain","No request or performance/capacity result"],"limits":["Port owner at failure time was not captured; collision between simultaneous native queue creators is a source-supported hypothesis, not proven culprit","CPU engine config acceptance does not prove native startup/fit/functional feasibility","Failure occurred before native KV profiling; no OOM/Graph/runtime operator conclusion"]}
atomic_json(j/"reduction.json",out);atomic_json(r/"reduction.json",out);atomic_json(r/"reduction_brief.json",out)
m=json.loads((r/"manifest.json").read_text());m.update(status="failed",valid=False,verdict="INCONCLUSIVE",diagnostic_evidence_valid=True,completed_at=phase["finished_at"],new_client_attempts=0,new_completed_inference_requests=0,new_effective_output_tokens=0,failure_phase="deploy",reduction=ref(r/"reduction.json"));atomic_json(r/"manifest.json",m)
(r/"summary.md").write_text("# "+r.name+"\n\nNative DP1/TP32/DCP1/EP32 startup failed before any inference. First observed native error on167 is MessageQueue ZMQ TCP55557 bind Address already in use; later Gloo peer-closed errors cascade. Both native groups exited; audit issued no model signals. Source/config accepted onCPU only, no KV profiling/Graph/fit/performance/capacity conclusion. Port owner at failure time unknown. Next investigate atomic bind control-layer fix without changing operators or native files; new Run only.\n")
job=json.loads((j/"job.json").read_text());ev={"id":"reduction",**ref(j/"reduction.json"),"locator":"source/state/full native startup logs/terminal container process and socket inventory"}
atomic_json(job["result"]["path"],{"schema_version":1,"job_id":job["job_id"],"status":"completed","summary":"Run40 zero-request startup failure audited: first ZMQ port bind conflict, cascading peer disconnect, no remaining native groups","execution":{"inner_exit_code":0,"acceptance":"passed","processes":[]},"findings":[],"evidence":[ev],"unknowns":out["limits"],"decision_request":None,"next_check_at":None})
print(json.dumps({"diagnostic_evidence_valid":True,"measurement_valid":False,"requests":0,"reduction":ref(j/"reduction.json")}))
