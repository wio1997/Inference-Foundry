from pathlib import Path
import json,sys,subprocess,hashlib
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import same_process
from owner_guard import start_watchdog
start_watchdog();r=Path(__file__).parent;p=r.parents[1];s=json.loads((r.parent/"GLM-RUN-0092/state.json").read_text());assert s["status"]=="completed"and not same_process(s["owner"])
f=p/"jobs/NATIVE-PD-VALIDATION-CPU-20261002T1710Z/reduction.json";assert hashlib.sha256(f.read_bytes()).hexdigest()=="3b92c18c90662ba0847cce6b2bd31ee32cbbaa842054eb97bd47af0c2e724411"
assert json.loads(f.read_text())["CPU_contracts"]["tests_run"]==33
f=r.parent/"GLM-RUN-0091/reduction_brief.json";assert hashlib.sha256(f.read_bytes()).hexdigest()=="9143c289204e75836ba2dd2f87c476150858f77958f0d7db99b340bc5b8713c7"
assert json.loads(f.read_text())["functional_acceptance"]
f=r.parent/"GLM-RUN-0092/reduction_brief.json";assert hashlib.sha256(f.read_bytes()).hexdigest()=="7a0995785345bcb49af5d2ffaaa4101808453e1d27ef3c55a94dcae3c2f70130"
z=subprocess.run([sys.executable,str(r/"epoch_check.py"),"epoch_prepare"],capture_output=True,timeout=240);(r/"epoch_prepare.stdout").write_bytes(z.stdout);(r/"epoch_prepare.stderr").write_bytes(z.stderr);z.check_returncode()
print("sameP89D88/CPU30/SDK0/nativeidle/independentEP16/1024c1/pinnedsource")
