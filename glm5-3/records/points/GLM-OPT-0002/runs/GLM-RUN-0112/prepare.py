from pathlib import Path
import json,sys,subprocess,hashlib
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import same_process
from owner_guard import start_watchdog
start_watchdog();r=Path(__file__).parent;p=r.parents[1];s=json.loads((r.parent/"GLM-RUN-0111/state.json").read_text());assert s["status"]=="completed"and not same_process(s["owner"])
f=p/"jobs/NATIVE-PD-GEOMETRY-CPU-20261002T2124Z-v2/reduction.json";assert hashlib.sha256(f.read_bytes()).hexdigest()=="2e93975a6fc642b087675cbc73306b4e4ec485491a43dd8995a065e0e5adb17"
assert json.loads(f.read_text())["CPU_contracts"]["tests_run"]==22
f=r.parent/"GLM-RUN-0111/reduction_brief.json";assert hashlib.sha256(f.read_bytes()).hexdigest()=="1212b7927df0897c7084f172c8e28649cb1df40459a37d8471eae4eb9e4014f3"
z=subprocess.run([sys.executable,str(r/"epoch_check.py"),"epoch_prepare"],capture_output=True,timeout=240);(r/"epoch_prepare.stdout").write_bytes(z.stdout);(r/"epoch_prepare.stderr").write_bytes(z.stderr);z.check_returncode()
print("sameP89_D109TP8PP2DCP8noEPnoKV/V10correctfactory/nativegeometryfallback/CPU22SDK0/idle/budget4096t1024c1/source")
