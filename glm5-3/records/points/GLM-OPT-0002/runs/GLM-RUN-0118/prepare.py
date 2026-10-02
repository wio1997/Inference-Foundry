from pathlib import Path
import json,sys,subprocess,hashlib
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import same_process
from owner_guard import start_watchdog
start_watchdog();r=Path(__file__).parent;p=r.parents[1];s=json.loads((r.parent/"GLM-RUN-0117/state.json").read_text());assert s["status"]=="completed"and not same_process(s["owner"])
f=p/"jobs/SHAPE-SPLIT-V11-CPU-20261002T2152Z/reduction.json";assert hashlib.sha256(f.read_bytes()).hexdigest()=="864814766cf6bda0cbfd4760461433bad53cce30ed51ade020ccc235384a395d"
assert json.loads(f.read_text())["CPU_contracts"]["tests_run"]==29
f=r.parent/"GLM-RUN-0117/reduction_brief.json";assert hashlib.sha256(f.read_bytes()).hexdigest()=="0eb1a7edd28ed41ded49e9e049c8a60e7ba92c121a21dba9b7cc193dac6afe97"
z=subprocess.run([sys.executable,str(r/"epoch_check.py"),"epoch_prepare"],capture_output=True,timeout=240);(r/"epoch_prepare.stdout").write_bytes(z.stdout);(r/"epoch_prepare.stderr").write_bytes(z.stderr);z.check_returncode()
print("sameP89_D109TP8PP2DCP8noEPnoKV/V11actualfactory/nativegeometryfallback/CPU29SDK0/idle/budget4096t1024c1/source")
