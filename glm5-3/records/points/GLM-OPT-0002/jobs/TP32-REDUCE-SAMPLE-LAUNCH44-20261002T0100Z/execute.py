import json,sys,subprocess,hashlib
from pathlib import Path
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,same_process
j=Path(__file__).parent;job=json.loads((j/"job.json").read_text());r=Path(job["inputs"][0]["path"]);old=r.parent/"GLM-RUN-0043"
state=json.loads((old/"state.json").read_text());assert state["status"]=="completed" and not same_process(state["owner"])
audit=json.loads((old.parents[1]/"jobs/REDUCE-FULL43-20261002T0057Z-v2/reduction.json").read_text());assert audit["measurement_valid"] and len(audit["gateway_requests"])==4 and audit["native_full_only"]["TP32"]["delta"]["generation_tokens_total"]==245760
probe=json.loads((old.parents[1]/"jobs/REDUCE-SAMPLING-SOURCE-20261002T0042Z/reduction.json").read_text());assert probe["model_calls"]==0 and all(x["enable_reduce_sample"]for x in probe["engine_config"]["full_native_engine_config"])
cpu=j.parents[1]/"jobs/ATOMIC-MQ-CONTRACTS-20261001T2334Z"
assert json.loads((cpu/"bridge.json").read_text())["cli_exit_code"]==0
assert json.loads((cpu/"reduction.json").read_text())["native_queue_payload_roundtrips"]==32
spec=json.loads((r/"controller_spec.json").read_text());assert [x["id"]for x in spec["stages"]]==["deploy","identity","cold","tools","capability","pilot","prefix","finalize"]
for pin in spec["stages"][0]["sources"]:assert hashlib.sha256(Path(pin["path"]).read_bytes()).hexdigest()==pin["sha256"]
p=subprocess.run(["/usr/bin/python3","/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime/controller.py","start",str(r/"controller_spec.json")],capture_output=True,timeout=30)
(j/"launch.stdout").write_bytes(p.stdout);(j/"launch.stderr").write_bytes(p.stderr);p.check_returncode()
state=json.loads((r/"state.json").read_text());assert state["status"]=="running" and same_process(state["owner"]);atomic_json(j/"launch_state.json",state)
m=json.loads((r/"manifest.json").read_text());m.update(status="running",controller_state_ref=str(r/"state.json"));atomic_json(r/"manifest.json",m)
def ref(path,key):
 raw=path.read_bytes();return {"id":key,"path":str(path),"bytes":len(raw),"sha256":hashlib.sha256(raw).hexdigest(),"locator":"immutable newcontroller launch identity"}
atomic_json(job["result"]["path"],{"schema_version":1,"job_id":job["job_id"],"status":"running","summary":"New unique Run44 controller for nativeDP1TP32DCP1EP32 headless rebuild, absent failedRun40 group verified, atomicMQ control installed before native launch","execution":{"inner_exit_code":0,"acceptance":"passed","processes":[{"pid":state["owner"]["pid"],"host":state["owner"]["host"],"readiness":"unready","evidence_ids":["state"],"boot_id":state["owner"]["boot_id"],"start_ticks":state["owner"]["start_ticks"]}]},"findings":[{"kind":"fact","text":"Run40failed/dead zero inference audited, CPUatomicMQ32native wirepairs valid; newsource/specverified. Bothhosts fullDCP1CPUconfigs andbothnativegroups absent guard and native source hashes run indeploy before newstart, nativefit/E2E pending.","scope":{"launch_only":True},"evidence_ids":["state"]}],"evidence":[ref(j/"launch_state.json","state"),ref(j/"launch.stdout","stdout")],"unknowns":["NativeTP32DCP1 graph/memory/cache/functional/dynamic/prefix E2E pending; no hardwarecapacity/KEEP"],"decision_request":None,"next_check_at":None})
print(json.dumps({"run":r.name,"owner":state["owner"],"stage":state["active_stage"]}))
