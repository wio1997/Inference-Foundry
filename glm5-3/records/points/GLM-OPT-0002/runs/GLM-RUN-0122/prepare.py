from pathlib import Path
import json,sys,subprocess,hashlib
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import same_process
from owner_guard import start_watchdog
start_watchdog();r=Path(__file__).parent;p=r.parents[1];s=json.loads((r.parent/"GLM-RUN-0121/state.json").read_text());assert s["status"]=="completed"and not same_process(s["owner"])
f=p/"jobs/SHAPE-SPLIT-V11-CPU-20261002T2152Z/reduction.json";assert hashlib.sha256(f.read_bytes()).hexdigest()=="864814766cf6bda0cbfd4760461433bad53cce30ed51ade020ccc235384a395d"
assert json.loads(f.read_text())["CPU_contracts"]["tests_run"]==29
f=r.parent/"GLM-RUN-0121/reduction_brief.json";assert hashlib.sha256(f.read_bytes()).hexdigest()=="2139d17a37fa90a71ffe856b1d3b1985b690a59a7dbc50e6cbc3bd351656e31f"
z=subprocess.run([sys.executable,str(r/"epoch_check.py"),"epoch_prepare"],capture_output=True,timeout=240);(r/"epoch_prepare.stdout").write_bytes(z.stdout);(r/"epoch_prepare.stderr").write_bytes(z.stderr);z.check_returncode()
print("sameP89_D121TP8PP2DCP8noEPnoKV_K5Graph48/V11actualfactory/nativegeometryfallback/CPU29SDK0/idle/budget4096t1024c1/source")
