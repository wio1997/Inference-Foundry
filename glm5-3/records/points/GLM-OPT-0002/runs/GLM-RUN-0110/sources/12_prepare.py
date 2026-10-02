from pathlib import Path
import json,sys,subprocess,hashlib
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import same_process
from owner_guard import start_watchdog
start_watchdog();r=Path(__file__).parent;p=r.parents[1];s=json.loads((r.parent/"GLM-RUN-0109/state.json").read_text());assert s["status"]=="completed"and not same_process(s["owner"])
f=p/"jobs/NATIVE-PD-GEOMETRY-CPU-20261002T2112Z/reduction.json";assert hashlib.sha256(f.read_bytes()).hexdigest()=="b1db3133fb5737c669682da35f155691a5fc6300795a836f0e6fac988541c6fa"
assert json.loads(f.read_text())["CPU_contracts"]["tests_run"]==21
z=subprocess.run([sys.executable,str(r/"epoch_check.py"),"epoch_prepare"],capture_output=True,timeout=240);(r/"epoch_prepare.stdout").write_bytes(z.stdout);(r/"epoch_prepare.stderr").write_bytes(z.stderr);z.check_returncode()
print("sameP89_K5_D100_K3/SP_D16,32_P6,12,24,48/CPU33/SDK0/nativeidle/independentEP16/1024c1/pinnedsource")
