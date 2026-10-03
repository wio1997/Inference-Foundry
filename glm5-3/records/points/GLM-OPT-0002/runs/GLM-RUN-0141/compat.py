from pathlib import Path
import json,sys,urllib.request,urllib.error,hashlib,re
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc
from owner_guard import start_watchdog,guard
from vllm.entrypoints.openai.responses.protocol import ResponsesResponse
start_watchdog();r=Path(__file__).parent;old=r.parent/"GLM-RUN-0139";http=urllib.request.build_opener(urllib.request.ProxyHandler({}));rows=[]
for label,ident,owner in[("base_json","resp_glm_run139_base","D0"),("other_base_json","resp_glm_run139_other","D1")]:
 guard()
 with http.open("http://127.0.0.1:8000/v1/responses/"+ident,timeout=30)as reply:raw=reply.read();assert reply.status==200
 (r/(label+"_retained.wire")).write_bytes(raw);v=json.loads(raw);assert ResponsesResponse.model_validate(v).id==ident
 assert v==json.loads((old/(label+".wire")).read_text())and v["usage"]["output_tokens"]==32
 binding=json.loads((r.parent/"GLM-RUN-0125/response_owners.json").read_text())["owners"][ident];assert binding["replica"]==owner
 rows.append(dict(id=ident,owner=owner,status=200,wire_sha256=hashlib.sha256(raw).hexdigest(),unchanged=True,new_inference=0))
atomic_json(r/"policy_switch_affinity.json",dict(at=utc(),valid=True,placement="active_count",same_native_epochs=True,old139bindings=rows,new_inference=0,output_credit=0))
print("policy switch retainsboth139native STORE ownerbindings/no newinference")
