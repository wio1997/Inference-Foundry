from pathlib import Path
import json,hashlib
p=Path(__file__).resolve().parent.parent
REFS=[('GLM-RUN-0226', '7bc7814f5ded16c4dd96bf55241c884b1536b21f3cded1c02497c86be32afb68', 'newD0_create', 'resp_glm_run226_D0_new'), ('GLM-RUN-0211', '9c7be1de9ac5558ca0a8ba25f24bcdf6fca054cb0de4810f5740c4339e8f8a77', 'newD1_create', 'resp_glm_run211_D1_new')]
def retained_responses():
 out=[]
 for run,sha,name,native_id in REFS:
  f=p/run/"client_final_summary.json";b=f.read_bytes();assert hashlib.sha256(b).hexdigest()==sha
  row=next(x for x in json.loads(b)["requests"]if x["name"]==name);wire=Path(row["wire"]["path"]);assert hashlib.sha256(wire.read_bytes()).hexdigest()==row["wire"]["sha256"]
  assert json.loads(wire.read_bytes())["id"]==native_id;out.append((native_id,str(wire)))
 return out
