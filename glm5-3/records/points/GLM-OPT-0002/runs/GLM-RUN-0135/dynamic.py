from pathlib import Path
import json,sys,subprocess
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc
from owner_guard import start_watchdog
import controls
start_watchdog();r=Path(__file__).parent
controls.policy=json.loads((r/"effective_policy.json").read_text())
try:
 z=subprocess.run([sys.executable,str(r/"dynamic_inner.py")],capture_output=True,timeout=1600);(r/"inner.stdout").write_bytes(z.stdout);(r/"inner.stderr").write_bytes(z.stderr);z.check_returncode()
 offset=json.loads((r/"policy_log_offset.json").read_text());raw=Path(offset["path"]).read_bytes()[offset["bytes"]:];(r/"native_policy_window.log").write_bytes(raw)
 states=[json.loads(l.split("GLM_ISSUE_BUDGET_SELECTED ",1)[1])for l in raw.decode().splitlines()if "GLM_ISSUE_BUDGET_SELECTED "in l]
 assert len(states)==1and states[0]["cohort"]=="GLM-COHORT-0131"and states[0]["serial"]==2and states[0]["prefill_cadence"]==2and states[0]["budget_tokens"]==4096and states[0]["prefill_threshold_tokens"]==1024and states[0]["fallback"]is False
 atomic_json(r/"policy_consumption.json",dict(at=utc(),API_EngineCore_node="166",headless_workers_node="167",native_selected=states,operator_math_changes=0))
finally:
 controls.policy=json.loads((r/"effective_policy.json").read_text())
 controls.update(dict(controls.policy,prefill_cadence=1,serial=controls.policy["serial"]+1),"cadence1_restore")
 atomic_json(r/"expected_restored_policy.json",controls.policy)
 z=subprocess.run([sys.executable,str(r/"epoch_check.py"),"epoch_restored"],capture_output=True,timeout=240);(r/"epoch_restored.stdout").write_bytes(z.stdout);(r/"epoch_restored.stderr").write_bytes(z.stderr);z.check_returncode()
 atomic_json(r/"restore_summary.json",dict(at=utc(),files_both_restored=True,policy=controls.policy,native32_same=True,idle=True,limits=["Idle restored file CAS verified; next native schedule consumes this policy, no extra inference solely to produce selected marker"]))
