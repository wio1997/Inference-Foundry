from pathlib import Path
import json,sys,subprocess,hashlib
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import same_process
from owner_guard import start_watchdog
start_watchdog();r=Path(__file__).parent;p=r.parents[1];s=json.loads((r.parent/"GLM-RUN-0090/state.json").read_text());assert s["status"]=="completed"and not same_process(s["owner"])
f=p/"jobs/NATIVE-PD-CPU-20261002T1652Z/reduction.json";assert hashlib.sha256(f.read_bytes()).hexdigest()=="a3718ce6f9bab4f1640e8ff933ee77042642b79acf1a376688ea70ece4b54c15"
assert json.loads(f.read_text())["CPU_contracts"]["tests_run"]==30
z=subprocess.run([sys.executable,str(r/"epoch_check.py"),"epoch_prepare"],capture_output=True,timeout=240);(r/"epoch_prepare.stdout").write_bytes(z.stdout);(r/"epoch_prepare.stderr").write_bytes(z.stderr);z.check_returncode()
print("sameP89D88/CPU30/SDK0/nativeidle/independentEP16/1024c1/pinnedsource")
