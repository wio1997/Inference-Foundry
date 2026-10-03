from pathlib import Path
import sys,json,hashlib,subprocess
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc
from native_client_execution import run_native_client,NativeClientExecutionError
j=Path(__file__).parent;checks=[]
def paths(name):
 return dict(record_path=j/(name+".execution.json"),stdout_path=j/(name+".stdout"),stderr_path=j/(name+".stderr"))
def child(name,source):
 f=j/(name+".py");f.write_text(source);return [sys.executable,"-u",str(f)]
seen=[]
args=child("durable","import sys,time\nprint('synthetic-BEGIN',flush=True)\nprint('synthetic-ERR',file=sys.stderr,flush=True)\ntime.sleep(.7)\nprint('synthetic-END',flush=True)\n")
def watch():
 f=j/"durable.stdout";err=j/"durable.stderr"
 if f.exists()and b"synthetic-BEGIN"in f.read_bytes()and b"synthetic-END"not in f.read_bytes()and err.exists()and b"synthetic-ERR"in err.read_bytes():seen.append(True)
r=run_native_client(args,expected_native_argv=args,timeout_s=4,host_only=True,guard=watch,**paths("durable"))
assert r["status"]=="succeeded"and r["exit_code"]==0and seen and not r["native_client_alive"]and not r["signal_attempts"]
assert (j/"durable.stdout").read_bytes()==b"synthetic-BEGIN\nsynthetic-END\n"and(j/"durable.stderr").read_bytes()==b"synthetic-ERR\n";checks.append("stdout-stderr durable before child completion; exact raw bytes/exit")
args=child("nonzero","import sys\nprint('synthetic-failure',flush=True)\nsys.exit(7)\n")
try:run_native_client(args,expected_native_argv=args,timeout_s=4,host_only=True,**paths("nonzero"));raise AssertionError("missing error")
except NativeClientExecutionError as e:assert e.record["status"]=="failed"and e.record["exit_code"]==7and not e.record["timed_out"]
checks.append("actual nonzero CPU child exit preserved with raw output")
args=child("deadline","import signal,time\nsignal.signal(signal.SIGTERM,lambda *_: (_ for _ in ()).throw(SystemExit(0)))\ntry:\n print('synthetic-start',flush=True)\n time.sleep(8)\nfinally:print('synthetic-finally',flush=True)\n")
try:run_native_client(args,expected_native_argv=args,timeout_s=.5,grace_s=2,host_only=True,**paths("deadline"));raise AssertionError("missing timeout")
except NativeClientExecutionError as e:
 r=e.record;assert r["status"]=="timed_out"and r["timed_out"]and r["exit_code"]==0 and not r["native_client_alive"]
 assert len(r["signal_attempts"])==1and r["signal_attempts"][0]["signal"]=="SIGTERM"and r["signal_attempts"][0]["pid"]==r["native_client"]["pid"]
 assert(j/"deadline.stdout").read_bytes()==b"synthetic-start\nsynthetic-finally\n"
checks.append("deadline remains invalid despite exit0; only exact spawned CPU PID SIGTERM; raw finally retained")
try:run_native_client(args,expected_native_argv=args,timeout_s=4,host_only=True,**paths("deadline"));raise AssertionError("replayed")
except FileExistsError:pass
checks.append("existing execution artifacts refuse replay")
for expected in [["python3","cli","serve","/model"],["python3","script","/tmp/not-owned-client.py"]]:
 try:run_native_client(["docker","exec","glm52-single","ignored"],expected_native_argv=expected,timeout_s=4,**paths("unsafe"));raise AssertionError("unsafe accepted")
 except ValueError:pass
assert not(j/"unsafe.execution.json").exists();checks.append("model argv and non-Run scripts rejected before container access")
for t in [0,-1,float("inf")]:
 try:run_native_client(args,expected_native_argv=args,timeout_s=t,host_only=True,**paths("invalid"));raise AssertionError("invalid bound accepted")
 except ValueError:pass
checks.append("invalid deadline bounds rejected before spawn")
out=dict(at=utc(),valid=True,simulation=True,CPU_contract_checks=checks,new_native_inference_requests=0,model_operations=0,NPU_workers_or_tensors_created=0,real_SDK_acknowledgements_tested=False,
 source=dict(path="/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime/native_client_execution.py",sha256=hashlib.sha256(Path("/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime/native_client_execution.py").read_bytes()).hexdigest()),
 limits=["HOST synthetic CPU child tests, not SDK init-final evidence or actual Docker cancellation proof","Existing sole184 controller may continue inference; CPU contract job adds none","Actual native-client E2E/deadline ownership logs still required"])
atomic_json(j/"reduction.json",out);b=(j/"reduction.json").read_bytes();e=dict(id="reduction",path=str(j/"reduction.json"),bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),locator="explicitCPU-simulation/durable-output/actual-hostchild-exit/exactPIDdeadline/no-replay")
atomic_json(j/"result.json",dict(schema_version=1,job_id=j.name,status="completed",summary="Six explicit CPU simulation contracts validate durable raw stdout/stderr, actual exit, exact owned PID deadline cancellation and replay/unsafe bounds rejection; no NPU/SDK claim",execution=dict(inner_exit_code=0,acceptance="passed",processes=[]),findings=[],evidence=[e],unknowns=out["limits"],decision_request=None,next_check_at=None));print(json.dumps(out))
