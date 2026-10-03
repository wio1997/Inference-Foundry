from pathlib import Path
import json,sys,subprocess,shlex
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc
from owner_guard import start_watchdog,guard
import controls
start_watchdog();r=Path(__file__).parent;controls.policy=json.loads((r/"effective_policy.json").read_text())
try:
 z=subprocess.run([sys.executable,str(r/"dynamic_inner.py")],capture_output=True,timeout=1600);(r/"inner.stdout").write_bytes(z.stdout);(r/"inner.stderr").write_bytes(z.stderr);z.check_returncode()
 selected={}
 for node in["166","167"]:
  guard();offset=json.loads((r/("policy_log_offset_"+node+".json")).read_text());code="from pathlib import Path;import sys;sys.stdout.buffer.write(Path("+repr(offset["path"])+").read_bytes()["+str(offset["bytes"])+":])"
  args=["python3","-c",code]
  if node=="167":args=["ssh","-o","BatchMode=yes","root@172.16.10.167",shlex.join(args)]
  z=subprocess.run(args,capture_output=True,timeout=60);z.check_returncode();(r/("native_policy_window_"+node+".log")).write_bytes(z.stdout)
  states=[json.loads(l.split("GLM_ISSUE_BUDGET_SELECTED ",1)[1])for l in z.stdout.decode().splitlines()if"GLM_ISSUE_BUDGET_SELECTED "in l]
  assert len(states)==1and states[0]["cohort"]=="GLM-COHORT-0137"and states[0]["serial"]==2and states[0]["budget_tokens"]==2048and states[0]["prefill_threshold_tokens"]==1024and states[0]["prefill_cadence"]==1and states[0]["fallback"]is False
  selected[node]=states
 atomic_json(r/"policy_consumption.json",dict(at=utc(),native_selected_both_nodes=selected,operator_math_changes=0))
finally:
 controls.policy=json.loads((r/"effective_policy.json").read_text())
 controls.update(dict(controls.policy,budget_tokens=4096,serial=controls.policy["serial"]+1),"budget4096_restore")
 atomic_json(r/"expected_restored_policy.json",controls.policy)
 z=subprocess.run([sys.executable,str(r/"epoch_check.py"),"epoch_restored"],capture_output=True,timeout=240);(r/"epoch_restored.stdout").write_bytes(z.stdout);(r/"epoch_restored.stderr").write_bytes(z.stderr);z.check_returncode()
 atomic_json(r/"restore_summary.json",dict(at=utc(),files_both_restored=True,policy=controls.policy,native32_same=True,idle=True,limits=["Idle restored file CAS verified; next native schedule consumes policy, no extra inference solely to produce selected marker"]))
