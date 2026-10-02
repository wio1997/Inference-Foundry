from pathlib import Path
import json,sys,subprocess,hashlib
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import same_process
from owner_guard import start_watchdog
start_watchdog();r=Path(__file__).parent;old=r.parent/"GLM-RUN-0089";s=json.loads((old/"state.json").read_text());assert s["status"]=="completed"and not same_process(s["owner"])
assert hashlib.sha256((old/"reduction_brief.json").read_bytes()).hexdigest()=="7442f90d4128448f5be7c89c51a198a3758a807a012df51fef5fd46c4e637b75"
a=json.loads((old/"reduction_brief.json").read_text());assert a["functional_acceptance"]and a["effective_public_output_tokens"]==214 and a["retained_D88_same_epoch_NPU16"]
for f in["adopted_model_identities.json","native_member_identities.json"]:(r/f).write_bytes((old/f).read_bytes())
z=subprocess.run([sys.executable,str(r/"epoch_check.py")],capture_output=True,timeout=240);(r/"initial_epoch.stdout").write_bytes(z.stdout);(r/"initial_epoch.stderr").write_bytes(z.stderr);z.check_returncode()
print("sameP89D88API2NPU32nativeidle/budget/source/epochs validated; directnativePDdiagnostic")
