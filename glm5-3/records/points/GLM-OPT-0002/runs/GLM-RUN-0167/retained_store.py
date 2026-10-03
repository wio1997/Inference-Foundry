"""Resolve immutable native STORE evidence by audited response identity, before HTTP."""
from pathlib import Path
import json,hashlib
SUMMARY=Path('/data/tiankuan/wio/Inference-Foundry/glm5-3/records/points/GLM-OPT-0002/runs/GLM-RUN-0164/client_final_summary.json')
SUMMARY_SHA='74f73bb334177a14a40b027b1f357043d05d346c3e0ee9c998cb2a0a95786f03'
def retained_responses():
 b=SUMMARY.read_bytes();assert hashlib.sha256(b).hexdigest()==SUMMARY_SHA
 value=json.loads(b);assert value["valid"]and value["mode"]=="final"
 row=next(x for x in value["requests"]if x["name"]=="newD1_create");ref=row["wire"];wire=Path(ref["path"]);raw=wire.read_bytes()
 assert len(raw)==ref["bytes"]and hashlib.sha256(raw).hexdigest()==ref["sha256"]
 v=json.loads(raw);assert v["id"]=="resp_glm_run164_D1_new"and v["usage"]["output_tokens"]==32and row["native_owner"]=="D1"
 base=SUMMARY.parent.parent/"GLM-RUN-0139/base_json.wire";v=json.loads(base.read_text());assert v["id"]=="resp_glm_run139_base"and v["usage"]["output_tokens"]==32
 return [("resp_glm_run139_base",str(base)),("resp_glm_run164_D1_new",str(wire))]
