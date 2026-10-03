from pathlib import Path
import sys,json,subprocess,hashlib,urllib.request
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc,same_process
from native_engines_service_config import checked_config
from owner_guard import start_watchdog,guard
import controls
start_watchdog();r=Path(__file__).parent;p=r.parents[1];old=r.parent/"GLM-RUN-0184"
s=json.loads((old/"state.json").read_text());assert s["status"]=="completed"and not same_process(s["owner"])
f=p/"jobs/AUDIT-RUN184-20261003T1504Z/reduction.json";assert hashlib.sha256(f.read_bytes()).hexdigest()=="54747dd953c1a08ff5db07d0891fc212cd9b9372d7b2d469f5c04b2238645518"
proof=json.loads((p/"jobs/AUDIT-RUN184-20261003T1504Z/public_service_proof.json").read_text());assert same_process(proof["host"])and [x.decode()for x in Path("/proc/"+str(proof["host"]["pid"])+"/cmdline").read_bytes().split(bytes([0]))if x]==proof["argv"]
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

f=p/"jobs/TOKEN-MEMO-CPU2-20261003T1323Z/reduction.json";assert hashlib.sha256(f.read_bytes()).hexdigest()=="53dd034cbe3fd69d85135972115f42fec38f8def928b547dfbcd9f67e8d44a18"

trace=r.parent/"GLM-RUN-0184/token_memo_trace.jsonl";raw=trace.read_bytes()
atomic_json(r/"memo_trace_cursor.json",dict(path=str(trace),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))

f=p/"jobs/DURABLE-CLIENT-CPU2-20261003T1501Z/reduction.json";assert hashlib.sha256(f.read_bytes()).hexdigest()=="9732640dd08a92800105fd5c345d6fcd48d6dd8b398758c473124bde81c9d813"

from native_client_execution import run_native_client
import shlex
native=["python3","-u","/data/tiankuan/wio/glm52-pd/deploy/plugins/local_engines137/native_acl_lifecycle.py","script",str(r/"lifecycle_client.py")]
shell="ulimit -c 0; export PD_LOCAL_IP=172.16.10.166 PD_NIC=business; source /data/tiankuan/wio/glm52-pd/deploy/scripts/pd_common_env.sh; export PYTHONPATH="+str(r)+":/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime:$PYTHONPATH; exec "+shlex.join(native)
exe=run_native_client(["docker","exec","glm52-single","bash","-c",shell],record_path=r/"SDK_CPU_execution.json",stdout_path=r/"SDK_CPU.stdout",stderr_path=r/"SDK_CPU.stderr",expected_native_argv=native,timeout_s=120,guard=guard)
acks=[json.loads(l)for l in(r/"SDK_CPU.stdout").read_text().splitlines()if l.startswith('{"event":')]
assert acks[0]["event"]=="task_acl_init"and acks[-1]["event"]=="task_acl_finalize"and acks[0]["returncode"]==acks[-1]["returncode"]==0
assert any(x["event"]=="native_client_CPU_probe"and x["inference"]==x["NPU_tensors"]==x["models_created"]==0for x in acks)
assert exe["native_client"]and not exe["native_client_alive"]and not exe["signal_attempts"]
controls.verify("after_SDK_CPU")
atomic_json(r/"SDK_CPU_acceptance.json",dict(at=utc(),actual_Docker_owned_client_capture=True,SDK_init_finalize0=[acks[0],acks[-1]],native_client_execution=exe,new_inference=0,models_or_tensors_created=0,no_Docker_cancellation_test_claim=True))
