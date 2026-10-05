from pathlib import Path
import sys,json,subprocess,hashlib,urllib.request
r=Path(__file__).parent;p=r.parents[1];g=p.parents[2];native=r.parent/"GLM-RUN-0228"
sys.path.insert(0,str(native/"runtime_bundle"))
from phase_runner import atomic_json,utc,same_process
from native_engines_service_config import checked_config
from owner_guard import start_watchdog,guard
import controls
start_watchdog()
prev=json.loads((native/"state.json").read_text());assert prev["status"]=="completed"and not same_process(prev["owner"])
audit=p/"jobs/AUDIT-RUN231-20261005TPOSTLATE/reduction.json";assert hashlib.sha256(audit.read_bytes()).hexdigest()=='b5d96084daf0d411d2256518474a479d3d60035994eecaa446d84d963335b2a7'
a=json.loads(audit.read_text());assert a["measurement_valid"]and a["functional_acceptance"]
proof=json.loads((r/"public_service_proof.json").read_text());assert same_process(proof["host"])and [x.decode()for x in Path("/proc/"+str(proof["host"]["pid"])+"/cmdline").read_bytes().split(bytes([0]))if x]==proof["argv"]
assert hashlib.sha256(Path(proof["container"]["config"]["path"]).read_bytes()).hexdigest()==proof["container"]["config"]["sha256"]
config,_=checked_config(r/"service_config.json");assert config["native_domains"]==proof["container"]["native_domains"]and config["placement"]["kind"]=="shape_split_idle_spill"
z=subprocess.run([sys.executable,str(r/"epoch_check.py"),"prepare"],capture_output=True,timeout=240);(r/"prepare_epoch.stdout").write_bytes(z.stdout);(r/"prepare_epoch.stderr").write_bytes(z.stderr);z.check_returncode()
http=urllib.request.build_opener(urllib.request.ProxyHandler({}))
for path in["/healthcheck","/control/replicas"]:
 with http.open("http://127.0.0.1:8000"+path,timeout=15)as res:value=json.loads(res.read())
 if path=="/healthcheck":assert value["status"]=="ok"and value["request_num"]==0
 else:assert len(value["replicas"])==2and all(not x["active_requests"]and not x["group_faulted"]and not x["draining"]for x in value["replicas"])
controls.verify("base_policy")
from retained_store import retained_responses
assert len(retained_responses())==2
trace=native/"token_memo_trace.jsonl";raw=trace.read_bytes()
atomic_json(r/"memo_trace_cursor.json",dict(path=str(trace),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
atomic_json(r/"prepared_public_service.json",proof);atomic_json(r/"prepared_state.json",dict(at=utc(),native32_same=True,policies=controls.policies,shape_split_idle_spill=True,models_operations=0,frontend_operations=0,audit226=dict(path=str(audit),sha256='b5d96084daf0d411d2256518474a479d3d60035994eecaa446d84d963335b2a7')))

prev223=json.loads((r.parent/"GLM-RUN-0231/state.json").read_text());assert prev223["status"]=="completed"and not same_process(prev223["owner"])and prev223["completed_stages"]==["prepare","dynamic"]
assert not(native/"diagnostic_enable.json").exists()

log=Path("/data/tiankuan/wio/glm52-pd/deploy/logs/local_pp228_166.log");atomic_json(r/"queue_log_cursor.json",dict(bytes=log.stat().st_size,policy=json.loads((r/"queue_policy_frozen.json").read_text())))
