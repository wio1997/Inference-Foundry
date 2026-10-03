from pathlib import Path
import json,sys,subprocess,shlex
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc
from owner_guard import start_watchdog,guard
import controls
start_watchdog();r=Path(__file__).parent
# Log offsets bind consumption proof to this execution; replacement167 log is149.
for node,name in[("166","local_engines137_166.log"),("167","local_engines149_167.log")]:
 guard();path="/data/tiankuan/wio/glm52-pd/deploy/logs/"+name
 code="from pathlib import Path;import json;print(json.dumps(dict(path="+repr(path)+",bytes=Path("+repr(path)+").stat().st_size)))";args=["python3","-c",code]
 if node=="167":args=["ssh","-o","BatchMode=yes","root@172.16.10.167",shlex.join(args)]
 z=subprocess.run(args,capture_output=True,timeout=60);z.check_returncode();atomic_json(r/("policy_log_offset_"+node+".json"),json.loads(z.stdout))
controls.verify("baseline1024")
try:
 z=subprocess.run([sys.executable,str(r/"formal.py")],capture_output=True,timeout=1750);(r/"experiment_inner.stdout").write_bytes(z.stdout);(r/"experiment_inner.stderr").write_bytes(z.stderr);z.check_returncode()
 selected={}
 for node in["166","167"]:
  guard();o=json.loads((r/("policy_log_offset_"+node+".json")).read_text());code="from pathlib import Path;import sys;sys.stdout.buffer.write(Path("+repr(o["path"])+").read_bytes()["+str(o["bytes"])+":])";args=["python3","-c",code]
  if node=="167":args=["ssh","-o","BatchMode=yes","root@172.16.10.167",shlex.join(args)]
  z=subprocess.run(args,capture_output=True,timeout=60);z.check_returncode();(r/("native_policy_window_"+node+".log")).write_bytes(z.stdout)
  states=[json.loads(l.split("GLM_ISSUE_BUDGET_SELECTED ",1)[1])for l in z.stdout.decode().splitlines()if"GLM_ISSUE_BUDGET_SELECTED "in l]
  assert len(states)==1and states[0]["cohort"]=="GLM-COHORT-0137"and states[0]["serial"]==5and states[0]["budget_tokens"]==4096and states[0]["prefill_threshold_tokens"]==1024and states[0]["prefill_cadence"]==1and states[0]["fallback"]is False
  selected[node]=states
 atomic_json(r/"policy_consumption.json",dict(at=utc(),native_selected_both_nodes=selected,operator_math_changes=0))
finally:
 controls.verify("baseline1024_after")
 atomic_json(r/"expected_restored_policy.json",controls.policy)
 z=subprocess.run([sys.executable,str(r/"epoch_check.py"),"epoch_restored"],capture_output=True,timeout=240);(r/"epoch_restored.stdout").write_bytes(z.stdout);(r/"epoch_restored.stderr").write_bytes(z.stderr);z.check_returncode()
 atomic_json(r/"restore_summary.json",dict(at=utc(),files_both_restored=True,policy_writes=0,policy=controls.policy,native32_same=True,idle=True,limits=["Nextnative schedule consumesrestore, no extra inference forlogmarker"]))
m=json.loads((r/"manifest.json").read_text());m.update(status="completed",valid=True,verdict="INCONCLUSIVE",results=json.loads((r/"formal_summary.json").read_text()));atomic_json(r/"manifest.json",m)
