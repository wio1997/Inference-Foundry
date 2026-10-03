from pathlib import Path
import sys,json,subprocess,hashlib,urllib.request
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc,same_process
from native_engines_service_config import checked_config
from owner_guard import start_watchdog,guard
import controls
start_watchdog();r=Path(__file__).parent;p=r.parents[1];old=r.parent/"GLM-RUN-0160"
s=json.loads((old/"state.json").read_text());assert s["status"]=="failed"and not same_process(s["owner"])
f=p/"jobs/AUDIT-RUN160-20261003T0806Z/reduction.json";assert hashlib.sha256(f.read_bytes()).hexdigest()=="564612447f9488e5c9a7882df0be9f515458528640568dcb4261bc4492902e5f"
proof=json.loads((p/"jobs/AUDIT-RUN160-20261003T0806Z/public_service_proof.json").read_text());assert same_process(proof["host"])and [x.decode()for x in Path("/proc/"+str(proof["host"]["pid"])+"/cmdline").read_bytes().split(bytes([0]))if x]==proof["argv"]
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

cpu=p/"jobs/TOKEN-INPUT-CPU3-20261003T0837Z/reduction.json";assert hashlib.sha256(cpu.read_bytes()).hexdigest()=="dfd0544bee78d2a14064d4330ad46037fc9084918463420c133e1437ef5bb949"
v=json.loads(cpu.read_text());assert v["valid"]and v["native32_same"]and v["native_counter_deltas0"]
for row in v["records"]:
 for key in ["ids","request","response"]:
  ref=row[key];b=Path(ref["path"]).read_bytes();assert len(b)==ref["bytes"]and hashlib.sha256(b).hexdigest()==ref["sha256"]
code="from pathlib import Path;import hashlib;print(hashlib.sha256(Path('/vllm-workspace/vllm/vllm/renderers/online_renderer.py').read_bytes()).hexdigest())"
import shlex
args=["ssh","-o","BatchMode=yes","root@172.16.10.167",shlex.join(["docker","exec","glm52-single","python3","-c",code])]
z=subprocess.run(args,capture_output=True,timeout=30);(r/"native167_renderer.stdout").write_bytes(z.stdout);(r/"native167_renderer.stderr").write_bytes(z.stderr);z.check_returncode();assert z.stdout.decode().strip()=="5a8500fc56ca5025005ec3931f7949859ff800ad7e18c8766f7d2d36de8b207b"
