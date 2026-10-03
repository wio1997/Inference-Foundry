from pathlib import Path
import json,sys,subprocess,shlex
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc
from owner_guard import start_watchdog,guard
import controls
start_watchdog();r=Path(__file__).parent
LOG={"166":"/data/tiankuan/wio/glm52-pd/deploy/logs/local_pp175_166.log","167":"/data/tiankuan/wio/glm52-pd/deploy/logs/local_pp168_167.log"}
def remote(node,code,stem):
 guard();args=["python3","-c",code]
 if node=="167":args=["ssh","-o","BatchMode=yes","root@172.16.10.167",shlex.join(args)]
 z=subprocess.run(args,capture_output=True,timeout=60);(r/(stem+".stdout")).write_bytes(z.stdout);(r/(stem+".stderr")).write_bytes(z.stderr);z.check_returncode();return z.stdout
offsets={}
for node,path in LOG.items():
 code="from pathlib import Path;import json;print(json.dumps(dict(path="+repr(path)+",bytes=Path("+repr(path)+").stat().st_size)))"
 offsets[node]=json.loads(remote(node,code,"policy_log_offset_"+node));atomic_json(r/("policy_log_offset_"+node+".json"),offsets[node])
try:
 controls.update_D0(dict(controls.policies["166"],prefill_threshold_tokens=512,serial=2),"D0_threshold512")
 controls.update_D1(dict(controls.policies["167"],prefill_threshold_tokens=512,serial=12),"D1_threshold512")
 z=subprocess.run([sys.executable,str(r/"formal.py")],capture_output=True,timeout=1750);(r/"experiment_inner.stdout").write_bytes(z.stdout);(r/"experiment_inner.stderr").write_bytes(z.stderr);z.check_returncode()
 selected={}
 for node,path in LOG.items():
  code="from pathlib import Path;import sys;sys.stdout.buffer.write(Path("+repr(path)+").read_bytes()["+str(offsets[node]["bytes"])+":])"
  raw=remote(node,code,"native_policy_window_"+node);(r/("native_policy_window_"+node+".log")).write_bytes(raw)
  states=[json.loads(l.split("GLM_ISSUE_BUDGET_SELECTED ",1)[1])for l in raw.decode().splitlines()if "GLM_ISSUE_BUDGET_SELECTED "in l]
  serial=2 if node=="166"else 12
  assert len(states)==1and states[0]["cohort"]=="GLM-COHORT-0137"and states[0]["serial"]==serial and states[0]["budget_tokens"]==8192and states[0]["prefill_threshold_tokens"]==512and states[0]["prefill_cadence"]==1and not states[0]["fallback"]and states[0]["native_max"]==8192
  selected[node]=states
 atomic_json(r/"policy_consumption.json",dict(at=utc(),native_selected=selected,both_threshold512=True,allocated8192_unchanged=True,operator_math_changes=0))
finally:
 # Always restore whichever policy this single controller actually changed, without inventing successful writes.
 if controls.policies["166"]["serial"]==2:controls.update_D0(dict(controls.policies["166"],prefill_threshold_tokens=1024,serial=3),"D0_threshold1024_restore")
 if controls.policies["167"]["serial"]==12:controls.update_D1(dict(controls.policies["167"],prefill_threshold_tokens=4096,serial=13),"D1_threshold4096_restore")
 z=subprocess.run([sys.executable,str(r/"epoch_check.py"),"epoch_terminal"],capture_output=True,timeout=240);(r/"epoch_terminal.stdout").write_bytes(z.stdout);(r/"epoch_terminal.stderr").write_bytes(z.stderr);z.check_returncode()
 atomic_json(r/"restore_summary.json",dict(at=utc(),D0_policy_restored=controls.policies["166"],D1_policy_restored=controls.policies["167"],private_policy_writes=sum((r/(stem+"_ack.json")).exists()for stem in["D0_threshold512","D1_threshold512","D0_threshold1024_restore","D1_threshold4096_restore"]),models_operations=0,frontend_operations=0,native32same_idle=True,logical_D0_peer_restored_by_client=True,limits=["Policyfiles/source/ownedidle verified; nextnative schedules consume restored3/13 without extra inference for markers"]))
m=json.loads((r/"manifest.json").read_text());m.update(status="completed",valid=True,verdict="INCONCLUSIVE",results=json.loads((r/"formal_summary.json").read_text()));atomic_json(r/"manifest.json",m)
