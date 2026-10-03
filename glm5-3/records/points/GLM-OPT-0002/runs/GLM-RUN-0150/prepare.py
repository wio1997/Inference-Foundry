from pathlib import Path
import json,sys,subprocess,hashlib,urllib.request
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,same_process,utc
from native_engines_service_config import checked_config
from owner_guard import guard,start_watchdog
start_watchdog();r=Path(__file__).parent;old=r.parent/"GLM-RUN-0149";s=json.loads((old/"state.json").read_text());assert s["status"]=="failed"and s["failure_phase"]=="switch"and not same_process(s["owner"])
failed=json.loads((old/"client_final_summary.json").read_text());assert not failed["valid"]and failed["error_type"]=="AssertionError"and failed["effective_output_tokens"]==59
assert failed["requests"][-1]["name"]=="D0_retained"and failed["requests"][-1]["status"]==200and failed["requests"][0]["name"]=="D0_retained"
public=json.loads((old/"restored/public_service_proof.json").read_text());assert same_process(public["host"])and [x.decode()for x in Path("/proc/"+str(public["host"]["pid"])+"/cmdline").read_bytes().split(bytes([0]))if x]==public["argv"]
assert hashlib.sha256(Path(public["config"]["path"]).read_bytes()).hexdigest()==public["config"]["sha256"]
config,_=checked_config(r/"service_config.json");assert config["native_domains"]==public["container_marker"]["native_domains"]
z=subprocess.run([sys.executable,str(r/"epoch_check.py"),"prepare"],capture_output=True,timeout=240);(r/"prepare.stdout").write_bytes(z.stdout);(r/"prepare.stderr").write_bytes(z.stderr);z.check_returncode()
atomic_json(r/"prepared_state.json",dict(at=utc(),same149_native32=True,retained_public=public["host"],original149_FAILED=True,no_model_frontend_replay=True))
