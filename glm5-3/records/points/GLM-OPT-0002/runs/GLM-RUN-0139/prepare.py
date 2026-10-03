from pathlib import Path
import json,sys,subprocess,hashlib,urllib.request
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc,same_process
from native_engines_service_config import checked_config
from owner_guard import start_watchdog,guard
start_watchdog();r=Path(__file__).parent;old=r.parent/"GLM-RUN-0138";p=r.parents[1]
state=json.loads((old/"state.json").read_text());assert state["status"]=="completed"and not same_process(state["owner"])
pin=json.loads((r/"expected_prior_audit.json").read_text());f=Path(pin["path"]);raw=f.read_bytes();assert len(raw)==pin["bytes"]and hashlib.sha256(raw).hexdigest()==pin["sha256"];proof=json.loads(raw);assert proof["measurement_valid"]and proof["functional_acceptance"]and proof["API_count"]==2and proof["NPU32_same"]and proof["SDK_client_init_finalize0"]
z=subprocess.run([sys.executable,str(r/"epoch_check.py"),"adopt_initial"],capture_output=True,timeout=240);(r/"adopt_initial.stdout").write_bytes(z.stdout);(r/"adopt_initial.stderr").write_bytes(z.stderr);z.check_returncode()
config,_=checked_config(r/"service_config.json");assert len(config["native_domains"])==2and {v["id"]for v in config["native_domains"]}=={"local-166","local-167"}and config["placement"]==dict(kind="shape_split",input_threshold_bytes=8192,prefill_members=["D1"],decode_members=["D0"])
http=urllib.request.build_opener(urllib.request.ProxyHandler({}))
with http.open("http://127.0.0.1:8000/healthcheck",timeout=10)as res:h=json.loads(res.read())
with http.open("http://127.0.0.1:8000/control/replicas",timeout=10)as res:placement=json.loads(res.read())
assert h["status"]=="ok"and h["request_num"]==0and len(placement["replicas"])==2and all(v["active_requests"]==0and not v["temporarily_unhealthy"]and not v["group_faulted"]and not v["draining"]for v in placement["replicas"])
atomic_json(r/"prepared_state.json",dict(at=utc(),two_native_engines_retained=True,all32same=True,idle=True,public137_healthy=True,new_models=0,signals=0,prior_audit=pin,existing_fullAPI127_133_reused_contract=True,scope="Only exactownedpublic137 frontendreplacement; native model roots unchanged; fullactualtwodomainAPIcontract/cost/SDK/wire"))
print("independent native137 engines adopted for fullAPI",flush=True)
