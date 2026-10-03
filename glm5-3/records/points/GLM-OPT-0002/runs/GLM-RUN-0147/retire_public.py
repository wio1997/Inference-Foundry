from pathlib import Path
import json,sys,subprocess,re,time,signal,hashlib,urllib.request,os
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc,same_process
from owner_guard import start_watchdog,guard
start_watchdog();r=Path(__file__).parent;p=r.parents[1];old=r.parent/"GLM-RUN-0141"
z=subprocess.run([sys.executable,str(r/"epoch_check.py"),"before_retire"],capture_output=True,timeout=240);(r/"before_retire.stdout").write_bytes(z.stdout);(r/"before_retire.stderr").write_bytes(z.stderr);z.check_returncode()
proof=json.loads((p/"jobs/AUDIT-RUN146-20261003T0405Z/public_service_proof.json").read_text());ident=proof["host"];pid=ident["pid"];container=proof["container"]
assert same_process(ident)and[v.decode()for v in Path("/proc/"+str(pid)+"/cmdline").read_bytes().split(bytes([0]))if v]==proof["argv"]
assert hashlib.sha256(Path(container["config"]["path"]).read_bytes()).hexdigest()==container["config"]["sha256"]
http=urllib.request.build_opener(urllib.request.ProxyHandler({}))
with http.open("http://127.0.0.1:8000/healthcheck",timeout=10)as res:h=json.loads(res.read())
assert h["status"]=="ok"and h["request_num"]==0
guard();os.kill(pid,signal.SIGTERM)
for _ in range(200):
 if not same_process(ident):break
 time.sleep(.1)
else:raise RuntimeError("exactfrontend didnot stop")
assert not re.search(r":8000\s",subprocess.check_output(["ss","-ltnp"],text=True))
acks=[json.loads(l)for l in(old/"public.gateway.log").read_text().splitlines()if l.startswith('{"event":')]
assert[x["event"]for x in acks]==["task_acl_init","task_acl_finalize"]and all(x["returncode"]==0for x in acks)
state=r.parent/"GLM-RUN-0125";fault=json.loads((state/"response_owners.json.fault").read_text());assert not fault["open"]and all(not v["faulted"]for v in fault["groups"].values())
atomic_json(r/"retire_public.json",dict(at=utc(),exact_owned_host=ident,SDKinit_finalize0=True,state125_preserved=True,fault_journal_closed=True,models_signalled=0))
