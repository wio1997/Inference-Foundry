from pathlib import Path
import sys,json,subprocess,hashlib,urllib.request
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc,same_process
from native_engines_service_config import checked_config
from owner_guard import start_watchdog,guard
import controls
start_watchdog();r=Path(__file__).parent;p=r.parents[1];old=r.parent/"GLM-RUN-0171"
s=json.loads((old/"state.json").read_text());assert s["status"]=="completed"and not same_process(s["owner"])
f=p/"jobs/AUDIT-RUN171-20261003T1056Z/reduction.json";assert hashlib.sha256(f.read_bytes()).hexdigest()=="95ea7c3094b0c182a0e8580a7c8a64ed10c9b46067f4755547bae43fc083b41c"
proof=json.loads((p/"jobs/AUDIT-RUN168V2-20261003T1030Z/public_service_proof.json").read_text());assert same_process(proof["host"])and [x.decode()for x in Path("/proc/"+str(proof["host"]["pid"])+"/cmdline").read_bytes().split(bytes([0]))if x]==proof["argv"]
assert hashlib.sha256(Path(proof["container"]["config"]["path"]).read_bytes()).hexdigest()==proof["container"]["config"]["sha256"]
config,_=checked_config(r/"service_config.json");assert config["native_domains"]==proof["container"]["native_domains"]
z=subprocess.run([sys.executable,str(r/"epoch_check.py"),"prepare"],capture_output=True,timeout=240);(r/"prepare_epoch.stdout").write_bytes(z.stdout);(r/"prepare_epoch.stderr").write_bytes(z.stderr);z.check_returncode()
http=urllib.request.build_opener(urllib.request.ProxyHandler({}))
for path in["/healthcheck","/control/replicas"]:
 with http.open("http://127.0.0.1:8000"+path,timeout=15)as res:value=json.loads(res.read())
 if path=="/healthcheck":assert value["status"]=="ok"and value["request_num"]==0
 else:assert len(value["replicas"])==2and all(not x["active_requests"]and not x["group_faulted"]and not x["draining"]and x["work_ranking_calibrated"]for x in value["replicas"])
controls.verify("base_policy")
atomic_json(r/"prepared_public_service.json",proof);atomic_json(r/"prepared_state.json",dict(at=utc(),native32_same=True,policy=controls.policy,both_work_hints=True,models_operations=0,frontend_operations=0))

from retained_store import retained_responses
assert len(retained_responses())==2
