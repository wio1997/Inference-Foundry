from pathlib import Path
import json,hashlib
p=Path(__file__).resolve().parent.parent
REFS=[("GLM-RUN-0175","c6d1c360cd43c068d54bc1488ab592e13fce1fda7715dd6c2646a84a72f79c2c","newD0_create","resp_glm_run175_D0_new"),("GLM-RUN-0168","40a57d1213315d940945ac0400554cff4ab078a0db658a7f0d51fea3d7cd034f","newD1_create","resp_glm_run168_D1_new")]
def retained_responses():
 out=[]
 for run,sha,name,native_id in REFS:
  f=p/run/"client_final_summary.json";b=f.read_bytes();assert hashlib.sha256(b).hexdigest()==sha
  row=next(x for x in json.loads(b)["requests"]if x["name"]==name);wire=Path(row["wire"]["path"]);assert hashlib.sha256(wire.read_bytes()).hexdigest()==row["wire"]["sha256"]
  assert json.loads(wire.read_bytes())["id"]==native_id;out.append((native_id,str(wire)))
 return out
