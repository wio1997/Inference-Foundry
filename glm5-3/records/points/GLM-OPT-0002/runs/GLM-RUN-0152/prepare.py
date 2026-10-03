from pathlib import Path
import sys,json,subprocess,hashlib,urllib.request
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc,same_process
from native_engines_service_config import checked_config
from owner_guard import start_watchdog,guard
start_watchdog();r=Path(__file__).parent;p=r.parents[1];old=r.parent/"GLM-RUN-0151"
s=json.loads((old/"state.json").read_text());assert s["status"]=="completed"and not same_process(s["owner"])
f=p/"jobs/AUDIT-RUN151V2-20261003T0523Z/reduction.json";assert hashlib.sha256(f.read_bytes()).hexdigest()=="e462870205dd66f3c50c8382ca54ff5e4552a27d9c7c91c1f44714cde98f8414"
proof=json.loads((f.parent/"public_service_proof.json").read_text());assert same_process(proof["host"])and [x.decode()for x in Path("/proc/"+str(proof["host"]["pid"])+"/cmdline").read_bytes().split(bytes([0]))if x]==proof["argv"]
assert hashlib.sha256(Path(proof["container"]["config"]["path"]).read_bytes()).hexdigest()==proof["container"]["config"]["sha256"]
config,_=checked_config(r/"service_config.json");assert config["native_domains"]==proof["container"]["native_domains"]
z=subprocess.run([sys.executable,str(r/"epoch_check.py"),"prepare"],capture_output=True,timeout=240);(r/"prepare_epoch.stdout").write_bytes(z.stdout);(r/"prepare_epoch.stderr").write_bytes(z.stderr);z.check_returncode()
http=urllib.request.build_opener(urllib.request.ProxyHandler({}))
def request(path,payload=None,method=None):
 guard();req=urllib.request.Request("http://127.0.0.1:8000"+path,data=None if payload is None else json.dumps(payload).encode(),headers={"Content-Type":"application/json"},method=method)
 with http.open(req,timeout=15)as res:return json.loads(res.read())
h=request("/healthcheck");assert h["status"]=="ok"and h["request_num"]==0
before=request("/control/replicas");assert len(before["replicas"])==2and all(not x["active_requests"]and not x["group_faulted"]and not x["draining"]for x in before["replicas"])
peers=config["environment"]["GLM_REPLICAS"];peers=json.loads(peers)if isinstance(peers,str)else peers;peer=next(x for x in peers if x["id"]=="D0")
assert peer["decode_tps"]==53.477750862288055and peer["prefill_bytes_per_s"]==31790.86100046442
# Readd full immutable compiler peer fields, restore proven rates for unchangedD0epoch.
request("/control/replicas/D0",method="DELETE")
try:request("/control/replicas",peer,"POST")
except BaseException:
 request("/control/replicas",peer,"POST");raise
after=request("/control/replicas");assert all(x["work_ranking_calibrated"]and not x["active_requests"]and not x["group_faulted"]for x in after["replicas"])
for row in after["replicas"]:
 exp=next(x for x in peers if x["id"]==row["id"]);assert row["decode_tps_hint"]==exp["decode_tps"]and row["prefill_bytes_per_s_hint"]==exp["prefill_bytes_per_s"]
atomic_json(r/"logical_D0_hint_restore.json",dict(at=utc(),before=before,after=after,compiler_peer=peer,same_native_epochs=True,models_operations=0,frontend_operations=0,no_new_inference=True))
atomic_json(r/"prepared_public_service.json",proof)
atomic_json(r/"prepared_state.json",dict(at=utc(),same151_native32=True,public_retained=True,placement="work_seconds",both_rate_hints=True,native_policy=dict(budget_tokens=4096,prefill_threshold_tokens=1024,prefill_cadence=1,serial=3),source128formal_adapter_reused=True))
