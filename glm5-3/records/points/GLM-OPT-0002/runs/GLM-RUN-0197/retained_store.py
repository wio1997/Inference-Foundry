from pathlib import Path
import json,hashlib
p=Path(__file__).resolve().parent.parent
def retained_responses():
 f=p/"GLM-RUN-0196/client_final_summary.json";raw=f.read_bytes();assert hashlib.sha256(raw).hexdigest()=='c2fa27dc79690412116eb0fc5b1feb61a44e728332d6ad54b617a068bbe61203'
 row=next(x for x in json.loads(raw)["requests"]if x["name"]=="joint_create");wire=Path(row["wire"]["path"])
 assert hashlib.sha256(wire.read_bytes()).hexdigest()==row["wire"]["sha256"]and json.loads(wire.read_bytes())["id"]=="resp_glm_run196_joint_new"
 return [("resp_glm_run196_joint_new",str(wire))]
