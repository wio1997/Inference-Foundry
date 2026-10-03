from pathlib import Path
import json,sys,subprocess,shlex
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc
from owner_guard import start_watchdog,guard
import controls
start_watchdog();r=Path(__file__).parent;path="/data/tiankuan/wio/glm52-pd/deploy/logs/local_pp168_167.log"
code="from pathlib import Path;import json;print(json.dumps(dict(path="+repr(path)+",bytes=Path("+repr(path)+").stat().st_size)))";args=["ssh","-o","BatchMode=yes","root@172.16.10.167",shlex.join(["python3","-c",code])];z=subprocess.run(args,capture_output=True,timeout=60);z.check_returncode();offset=json.loads(z.stdout);atomic_json(r/"policy_log_offset_167.json",offset)
controls.update_D1(dict(controls.policies["167"],prefill_threshold_tokens=1024,serial=10),"D1_threshold1024")
try:
 z=subprocess.run([sys.executable,str(r/"formal.py")],capture_output=True,timeout=1750);(r/"experiment_inner.stdout").write_bytes(z.stdout);(r/"experiment_inner.stderr").write_bytes(z.stderr);z.check_returncode()
 code="from pathlib import Path;import sys;sys.stdout.buffer.write(Path("+repr(path)+").read_bytes()["+str(offset["bytes"])+":])";args=["ssh","-o","BatchMode=yes","root@172.16.10.167",shlex.join(["python3","-c",code])];z=subprocess.run(args,capture_output=True,timeout=60);z.check_returncode();(r/"native_policy_window_167.log").write_bytes(z.stdout)
 states=[json.loads(l.split("GLM_ISSUE_BUDGET_SELECTED ",1)[1])for l in z.stdout.decode().splitlines()if"GLM_ISSUE_BUDGET_SELECTED "in l]
 assert len(states)==1and states[0]["cohort"]=="GLM-COHORT-0137"and states[0]["serial"]==10and states[0]["budget_tokens"]==8192and states[0]["prefill_threshold_tokens"]==1024and states[0]["prefill_cadence"]==1and not states[0]["fallback"]and states[0]["native_max"]==8192
 atomic_json(r/"policy_consumption.json",dict(at=utc(),native_D1_selected=states,D0_no_policy_changes=True,operator_math_changes=0))
finally:
 controls.update_D1(dict(controls.policies["167"],prefill_threshold_tokens=4096,serial=controls.policies["167"]["serial"]+1),"D1_threshold4096_restore")
 z=subprocess.run([sys.executable,str(r/"epoch_check.py"),"epoch_terminal"],capture_output=True,timeout=240);(r/"epoch_terminal.stdout").write_bytes(z.stdout);(r/"epoch_terminal.stderr").write_bytes(z.stderr);z.check_returncode()
 atomic_json(r/"restore_summary.json",dict(at=utc(),D0_policy_unchanged=controls.policy,D1_policy_restored=controls.policies["167"],private_policy_writes=2,models_operations=0,frontend_operations=0,native32same_idle=True,logical_D0_peer_restored_by_client=True,limits=["Files/source/ownedidle verified; nextnative schedule consumesrestore11, noextra inference forSELECTED marker"]))
m=json.loads((r/"manifest.json").read_text());m.update(status="completed",valid=True,verdict="INCONCLUSIVE",results=json.loads((r/"formal_summary.json").read_text()));atomic_json(r/"manifest.json",m)
