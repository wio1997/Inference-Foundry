from pathlib import Path
import json,sys,subprocess,hashlib,shlex,re,urllib.request
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,same_process,process_identity
from owner_guard import start_watchdog,guard
start_watchdog();r=Path(__file__).parent;p=r.parents[1];old=r.parent/"GLM-RUN-0127"
s=json.loads((old/"state.json").read_text());assert s["status"]=="completed"and not same_process(s["owner"])
audit=p/"jobs/AUDIT-RUN127-20261003T0005Z/reduction.json";assert hashlib.sha256(audit.read_bytes()).hexdigest()=="19447bc4a94d4f133058ea48da3d10c3df065c454dc55574ec75ddfbf99596de"
proof=json.loads(audit.read_text())["public_service"];host=proof["host_identity"];service=proof["container_owner"]
assert same_process(host)and[v.decode()for v in Path("/proc/"+str(host["pid"])+"/cmdline").read_bytes().split(bytes([0]))if v]==service["argv"]
assert hashlib.sha256(Path(service["config"]["path"]).read_bytes()).hexdigest()==service["config"]["sha256"]
assert json.loads((r/"execution_groups.json").read_text())==service["native_domains"]
z=subprocess.run([sys.executable,str(r/"epoch_check.py"),"epoch_prepare"],capture_output=True,timeout=240);(r/"epoch_prepare.stdout").write_bytes(z.stdout);(r/"epoch_prepare.stderr").write_bytes(z.stderr);z.check_returncode()
http=urllib.request.build_opener(urllib.request.ProxyHandler({}))
with http.open("http://127.0.0.1:8000/control/replicas",timeout=10)as res:placement=json.loads(res.read())
assert len(placement["replicas"])==2and all(not x["temporarily_unhealthy"]and not x["group_faulted"]and not x["draining"]and x["active_requests"]==0for x in placement["replicas"])
atomic_json(r/"prepared_public_service.json",dict(owner=service,host_identity=process_identity(host["pid"]),placement=placement,model_operations=0,gateway_operations=0))
print("Actual127public/P89+D123currentidle/P3D1/SDKfullAPIreused; 4full61440closedc2/D1only/newAISBenchexecution")
