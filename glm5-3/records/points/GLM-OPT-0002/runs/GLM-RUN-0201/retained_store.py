from pathlib import Path
import json,hashlib
p=Path(__file__).resolve().parent.parent
REFS=[('GLM-RUN-0200', '32ada6815a0673356610889f0261f8ec4f27be11eb42ab146b99429a83e54cc3', 'D0_create', 'resp_glm_run200_D0_new'), ('GLM-RUN-0200', '32ada6815a0673356610889f0261f8ec4f27be11eb42ab146b99429a83e54cc3', 'D1_create', 'resp_glm_run200_D1_new')]
def retained_responses():
 out=[]
 for run,sha,name,native_id in REFS:
  f=p/run/"client_final_summary.json";b=f.read_bytes();assert hashlib.sha256(b).hexdigest()==sha
  row=next(x for x in json.loads(b)["requests"]if x["name"]==name);wire=Path(row["wire"]["path"]);assert hashlib.sha256(wire.read_bytes()).hexdigest()==row["wire"]["sha256"]
  assert json.loads(wire.read_bytes())["id"]==native_id;out.append((native_id,str(wire)))
 return out
