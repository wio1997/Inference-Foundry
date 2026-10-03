from pathlib import Path
import json,sys,subprocess,hashlib
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc,same_process
from standalone_service_config import render
from owner_guard import start_watchdog
start_watchdog();r=Path(__file__).parent;old=r.parent/"GLM-RUN-0132"
s=json.loads((old/"state.json").read_text());assert s["status"]=="completed"and not same_process(s["owner"])
f=r.parents[1]/"jobs/AUDIT-RUN132-20261003T0155Z/reduction.json";assert hashlib.sha256(f.read_bytes()).hexdigest()=="d8b71f2076c2d378808ff8c0411c594aaebd1e61eaa541583b59a495e6734555"
z=subprocess.run([sys.executable,str(r/"epoch_check.py"),"prepare_epoch"],capture_output=True,timeout=240);(r/"prepare_epoch.stdout").write_bytes(z.stdout);(r/"prepare_epoch.stderr").write_bytes(z.stderr);z.check_returncode()
config=render(r,r.parent/"GLM-RUN-0125");atomic_json(r/"service_config.json",config)
assert config["native_domains"]==json.loads((old/"service_config.json").read_text())["native_domains"]
atomic_json(r/"prepared_state.json",dict(at=utc(),no_model_operations=True,common_native_domain_same=True,state125_preserved=True))
