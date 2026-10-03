from pathlib import Path
import json,sys,subprocess,hashlib,urllib.request
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc,same_process
from native_engines_service_config import checked_config
from owner_guard import start_watchdog,guard
start_watchdog();r=Path(__file__).parent;old=r.parent/"GLM-RUN-0141";p=r.parents[1]
s=json.loads((old/"state.json").read_text());assert s["status"]=="completed"and not same_process(s["owner"])
f=p/"jobs/AUDIT-RUN141-20261003T0321Z/reduction.json";assert hashlib.sha256(f.read_bytes()).hexdigest()=="6481d798db1d16d1e780965fe1c97326da668965cd4cc0e468b5dc79338ed2e0"
proof=json.loads((p/"jobs/AUDIT-RUN141-20261003T0321Z/public_service_proof.json").read_text());ident=proof["host"];assert same_process(ident)
assert [v.decode()for v in Path("/proc/"+str(ident["pid"])+"/cmdline").read_bytes().split(bytes([0]))if v]==proof["argv"]
config,_=checked_config(r/"service_config.json");assert config["native_domains"]==proof["container"]["native_domains"]
z=subprocess.run([sys.executable,str(r/"epoch_check.py"),"prepare_epoch"],capture_output=True,timeout=240);(r/"prepare_epoch.stdout").write_bytes(z.stdout);(r/"prepare_epoch.stderr").write_bytes(z.stderr);z.check_returncode()
http=urllib.request.build_opener(urllib.request.ProxyHandler({}))
with http.open("http://127.0.0.1:8000/healthcheck",timeout=10)as reply:h=json.loads(reply.read())
assert h["status"]=="ok"and h["request_num"]==0
import shlex
policy=dict(schema_version=1,cohort_id="GLM-COHORT-0137",budget_tokens=4096,prefill_threshold_tokens=1024,prefill_cadence=1,serial=1)
for node in["166","167"]:
 guard();code="from pathlib import Path;import json,hashlib;print(json.dumps(dict(policy=json.loads(Path('/data/tiankuan/wio/glm52-pd/deploy/plugins/local_engines137/issue_budget_policy.json').read_text()),source_sha256=hashlib.sha256(Path('/data/tiankuan/wio/glm52-pd/deploy/plugins/local_engines137/issue_budget_scheduler_v3.py').read_bytes()).hexdigest())))"
 args=["python3","-c",code]
 if node=="167":args=["ssh","-o","BatchMode=yes","root@172.16.10.167",shlex.join(args)]
 z=subprocess.run(args,capture_output=True,timeout=30);z.check_returncode();v=json.loads(z.stdout);assert v["policy"]==policy and v["source_sha256"]==hashlib.sha256((r.parents[4]/"runtime/issue_budget_scheduler_v3.py").read_bytes()).hexdigest();atomic_json(r/("policy_before_"+node+".json"),v)
assert config["placement"]==dict(kind="active_count")
atomic_json(r/"effective_policy_initial.json",policy)
atomic_json(r/"prepared_state.json",dict(at=utc(),retained_public_host=ident,same_native_cohort="GLM-COHORT-0137",models_and_frontends_operations=0,source114_workpackage_reused=True))
