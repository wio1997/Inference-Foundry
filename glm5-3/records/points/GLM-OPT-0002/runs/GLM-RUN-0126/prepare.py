from pathlib import Path
import json,sys,subprocess,hashlib
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import same_process
from owner_guard import start_watchdog
start_watchdog();r=Path(__file__).parent;p=r.parents[1];s=json.loads((r.parent/"GLM-RUN-0123/state.json").read_text());assert s["status"]=="completed"and not same_process(s["owner"])
f=p/"jobs/SHAPE-SPLIT-V11-CPU-20261002T2152Z/reduction.json";assert hashlib.sha256(f.read_bytes()).hexdigest()=="864814766cf6bda0cbfd4760461433bad53cce30ed51ade020ccc235384a395d"
assert json.loads(f.read_text())["CPU_contracts"]["tests_run"]==29
z=subprocess.run([sys.executable,str(r/"epoch_check.py"),"epoch_prepare"],capture_output=True,timeout=240);(r/"epoch_prepare.stdout").write_bytes(z.stdout);(r/"epoch_prepare.stderr").write_bytes(z.stderr);z.check_returncode()
print("sameP89_D109PP2TP8DCP8_noEP_noKV/V11shapeactualfactory/CPU29_actualCLI_twohealthyroles_nohelper/SDK0/nativeidle/1024c1/source")

assert not __import__("re").search(r":8000\s",subprocess.check_output(["ss","-ltnp"],text=True))
f=p/"jobs/SERVICE-ENTRY-CPU-20261002T2316Z/reduction.json";assert hashlib.sha256(f.read_bytes()).hexdigest()=="da262c63752a6f0a33dd8e6a7daf5879776dc94fd43bcb174cd91d410486ecfb"
f=r.parent/"GLM-RUN-0123/reduction_brief.json";assert hashlib.sha256(f.read_bytes()).hexdigest()=="37614751c06594e4fccb45118f8dfbd4106927b2d5c90c1473f4412e83ac3142"

f=p/"jobs/AUDIT-124-CHILD-CPU-20261002T2353Z/reduction.json";assert hashlib.sha256(f.read_bytes()).hexdigest()=="2a87772f5475de446a687803a9394108501c8fb89fc5d5512a9d7ad86615f793"
s=json.loads((r.parent/"GLM-RUN-0124/state.json").read_text());assert s["status"]=="failed"and not same_process(s["owner"])

f=p/"jobs/SERVICE-TERM-CPU-20261002T2357Z/reduction.json";assert hashlib.sha256(f.read_bytes()).hexdigest()=="610c7efc592817050b228b7c6ccbac62e4f7a45ba8db477faf66daffe5594401"
s=json.loads((r.parent/"GLM-RUN-0125/state.json").read_text());assert s["status"]=="completed"and not same_process(s["owner"])
