from pathlib import Path
import json,hashlib
p=Path(__file__).resolve().parent.parent
REFS=[("GLM-RUN-0188","c58c8ee956e2925d51978d28b35b4d0152ad6bec4a822c198bde7027835aead4","newD0_create","resp_glm_run188_D0_new"),("GLM-RUN-0184","5ee32e45c0b5d73c4d2029ff571904a01f971cef54057c153655ace05cbf5481","newD1_create","resp_glm_run184_D1_new")]
def retained_responses():
 out=[]
 for run,sha,name,native_id in REFS:
  f=p/run/"client_final_summary.json";b=f.read_bytes();assert hashlib.sha256(b).hexdigest()==sha
  row=next(x for x in json.loads(b)["requests"]if x["name"]==name);wire=Path(row["wire"]["path"]);assert hashlib.sha256(wire.read_bytes()).hexdigest()==row["wire"]["sha256"]
  assert json.loads(wire.read_bytes())["id"]==native_id;out.append((native_id,str(wire)))
 return out
