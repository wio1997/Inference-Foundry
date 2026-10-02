from pathlib import Path
import sys,subprocess,json
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json
from owner_guard import start_watchdog
import controls
start_watchdog();r=Path(__file__).parent
try:
 z=subprocess.run([sys.executable,str(r/"dynamic_inner.py")],capture_output=True,timeout=1600);(r/"inner.stdout").write_bytes(z.stdout);(r/"inner.stderr").write_bytes(z.stderr);z.check_returncode()
finally:
 controls.policy=json.loads((r/"effective_policy.json").read_text())
 controls.update(dict(controls.policy,budget_tokens=4096,prefill_threshold_tokens=1024,serial=3),"restore_D109_budget4096_threshold1024")
 atomic_json(r/"expected_D_policy.json",controls.policy)
 z=subprocess.run([sys.executable,str(r/"epoch_check.py"),"epoch_restored"],capture_output=True,timeout=240);(r/"epoch_restored.stdout").write_bytes(z.stdout);(r/"epoch_restored.stderr").write_bytes(z.stderr);z.check_returncode()
