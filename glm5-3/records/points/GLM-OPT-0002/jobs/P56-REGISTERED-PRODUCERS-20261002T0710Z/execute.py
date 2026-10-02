from pathlib import Path
import sys,json,hashlib
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json
from registered_kv_identity import collect,verify_registered
j=Path(__file__).parent;r=j.parents[1]/"runs/GLM-RUN-0056"
owners=json.loads((r/"adopted_model_identities.json").read_text());planned=json.loads((r/"planned_launch.json").read_text());actual=collect(j,owners,planned);verify_registered(actual)
assert all(len(x["worker_handshakes"])==len(x["physical_workers"])==16 and x["num_blocks"]==41 for x in actual.values())
limits=["32actualPworker readonlyGET_META/socketPID/APIroot provenance, no inference/KVtransfer/SDKfinish/free","Native UUID/DPrank IDs are actualregistered IDs, CLI label only seed; consumer metadata must preserveobservedID","Cachegeometry/rawhandshake no liveKV bytes/allshard completion/capacity proof"]
source=Path("/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime/registered_kv_identity.py");out=dict(producers=actual,source=dict(path=str(source),sha256=hashlib.sha256(source.read_bytes()).hexdigest()),signals=0,new_requests=0,models=0,limits=limits);atomic_json(j/"reduction.json",out);b=(j/"reduction.json").read_bytes();atomic_json(j/"result.json",dict(schema_version=1,job_id=j.name,status="completed",summary="All32actualPworkers native registeredIDs/41blocks/socketPID exactAPIroot andbootticks verified; no newinference/signals/transfer",execution=dict(inner_exit_code=0,acceptance="passed",processes=[]),findings=[],evidence=[dict(id="reduction",path=str(j/"reduction.json"),bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),locator="32native readonlyGET_META +physicalworker/root identities")],unknowns=limits,decision_request=None,next_check_at=None));print(json.dumps(dict(engines={k:x["engine_id"]for k,x in actual.items()},source=out["source"],sha256=hashlib.sha256(b).hexdigest())))
