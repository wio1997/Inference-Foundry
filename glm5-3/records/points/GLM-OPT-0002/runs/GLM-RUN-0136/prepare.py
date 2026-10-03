from pathlib import Path
import json,sys,subprocess,hashlib,urllib.request
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc,same_process
from standalone_service_config import checked_config
from owner_guard import start_watchdog,guard
start_watchdog();r=Path(__file__).parent;old=r.parent/"GLM-RUN-0135";p=r.parents[1]
s=json.loads((old/"state.json").read_text());assert s["status"]=="completed"and not same_process(s["owner"])
f=p/"jobs/AUDIT-RUN135-20261003T0221Z/reduction.json";assert hashlib.sha256(f.read_bytes()).hexdigest()=="b169c3ac52b2a294fb84a2b83600267ed36e9e943ae276c824fc6bde00aa9bc2"
proof=json.loads((p/"jobs/AUDIT-RUN135-20261003T0221Z/public_service_proof.json").read_text());ident=proof["host"];assert same_process(ident)
assert [v.decode()for v in Path("/proc/"+str(ident["pid"])+"/cmdline").read_bytes().split(bytes([0]))if v]==proof["argv"]
config,_=checked_config(r/"service_config.json");assert config["native_domains"]==proof["container"]["native_domains"]
z=subprocess.run([sys.executable,str(r/"epoch_check.py"),"prepare_epoch"],capture_output=True,timeout=240);(r/"prepare_epoch.stdout").write_bytes(z.stdout);(r/"prepare_epoch.stderr").write_bytes(z.stderr);z.check_returncode()
http=urllib.request.build_opener(urllib.request.ProxyHandler({}))
with http.open("http://127.0.0.1:8000/healthcheck",timeout=10)as reply:h=json.loads(reply.read())
assert h["status"]=="ok"and h["request_num"]==0
atomic_json(r/"prepared_state.json",dict(at=utc(),retained_public_host=ident,same_native_cohort="GLM-COHORT-0131",models_and_frontends_operations=0,source114_workpackage_reused=True))

import controls
controls.verify("resident_cadence1serial3")
atomic_json(r/"effective_policy_initial.json",controls.policy)
