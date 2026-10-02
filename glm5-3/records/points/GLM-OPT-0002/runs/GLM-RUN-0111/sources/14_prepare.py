from pathlib import Path
import json,sys,subprocess,hashlib
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import same_process
from owner_guard import start_watchdog
start_watchdog();r=Path(__file__).parent;p=r.parents[1];s=json.loads((r.parent/"GLM-RUN-0109/state.json").read_text());assert s["status"]=="completed"and not same_process(s["owner"])
f=p/"jobs/NATIVE-PD-GEOMETRY-CPU-20261002T2124Z-v2/reduction.json";assert hashlib.sha256(f.read_bytes()).hexdigest()=="2e93975a6fc642b087675cbc73306b4e4eec485491a43dd8995a065e0e5adb17"
assert json.loads(f.read_text())["CPU_contracts"]["tests_run"]==22
z=subprocess.run([sys.executable,str(r/"epoch_check.py"),"epoch_prepare"],capture_output=True,timeout=240);(r/"epoch_prepare.stdout").write_bytes(z.stdout);(r/"epoch_prepare.stderr").write_bytes(z.stderr);z.check_returncode()
print("sameP89_D109PP2TP8DCP8_noEP_noKV/correctV10factory/CPU22_actualCLI_nohelper/SDK0/nativeidle/1024c1/source")
