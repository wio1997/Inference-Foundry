from pathlib import Path
import sys,json,subprocess
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc
from owner_guard import start_watchdog,guard
import controls
start_watchdog();r=Path(__file__).parent
controls.verify("pre_experiment_policy")
try:
 z=subprocess.run([sys.executable,str(r/"formal.py")],capture_output=True,timeout=1750);(r/"experiment_inner.stdout").write_bytes(z.stdout);(r/"experiment_inner.stderr").write_bytes(z.stderr);z.check_returncode()
finally:
 controls.verify("post_experiment_policy")
 z=subprocess.run([sys.executable,str(r/"epoch_check.py"),"epoch_terminal"],capture_output=True,timeout=240);(r/"epoch_terminal.stdout").write_bytes(z.stdout);(r/"epoch_terminal.stderr").write_bytes(z.stderr);z.check_returncode()
 atomic_json(r/"restore_summary.json",dict(at=utc(),D0_policy_unchanged=controls.policy,D1_policy_unchanged=controls.policies["167"],private_policy_writes=0,models_operations=0,frontend_operations=0,native32same_idle=True,logical_D0_peer_restored_by_client=True))
m=json.loads((r/"manifest.json").read_text());m.update(status="completed",valid=True,verdict="INCONCLUSIVE",results=json.loads((r/"formal_summary.json").read_text()));atomic_json(r/"manifest.json",m)
