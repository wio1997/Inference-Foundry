from pathlib import Path
import json,sys,subprocess,shlex,hashlib,urllib.request,re,time
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc,same_process
from owner_guard import start_watchdog,guard
start_watchdog();r=Path(__file__).parent;old=r.parent/"GLM-RUN-0084";s=json.loads((old/"state.json").read_text());assert s["status"]=="completed"and not same_process(s["owner"])
proof=json.loads((old/"reduction_brief.json").read_text());assert proof["functional_acceptance"]and proof["effective_public_output_tokens"]==10112

import controls
controls.verify("initial")
ports=subprocess.check_output(["ss","-ltnp"],text=True);assert not re.search(r":8002\s",ports)
cpu=json.loads((r.parents[1]/"jobs/WORK-SECONDS-CPU-20261002T1445Z/reduction.json").read_text())
assert cpu["CPU_contracts"]["passed"]and cpu["CPU_contracts"]["tests_run"]==24
controls.update(dict(controls.policy,prefill_threshold_tokens=1024,serial=2),"threshold1024")
atomic_json(r/"prepared_state.json",dict(at=utc(),retained_native_API2_NPU32=True,policy=controls.policy,port8002_free=True,CPU_gatewayV7_24contracts=True,model_operations=0))
