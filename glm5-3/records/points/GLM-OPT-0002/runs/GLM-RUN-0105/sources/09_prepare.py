from pathlib import Path
import json,sys,subprocess,hashlib
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import same_process
from owner_guard import start_watchdog
start_watchdog();r=Path(__file__).parent;s=json.loads((r.parent/"GLM-RUN-0104/state.json").read_text());assert s["status"]=="completed"and not same_process(s["owner"])
f=r.parent/"GLM-RUN-0104/reduction_brief.json";assert hashlib.sha256(f.read_bytes()).hexdigest()=="edf073ca23f396969821ba04ebb88c734cd7ae16e00da643e197822934ca8248"
z=subprocess.run([sys.executable,str(r/"epoch_check.py"),"epoch_prepare"],capture_output=True,timeout=240);(r/"epoch_prepare.stdout").write_bytes(z.stdout);(r/"epoch_prepare.stderr").write_bytes(z.stderr);z.check_returncode()
