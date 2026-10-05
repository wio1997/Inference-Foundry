from pathlib import Path
import sys,json,hashlib,urllib.request,urllib.error,time
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc
from owner_guard import start_watchdog,guard
start_watchdog();r=Path(__file__).parent;http=urllib.request.build_opener(urllib.request.ProxyHandler({}))
def ref(f):
 b=Path(f).read_bytes();return dict(path=str(f),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def ctl(path,body=None,method=None):
 guard();q=urllib.request.Request("http://127.0.0.1:8000"+path,data=None if body is None else json.dumps(body).encode(),headers={"content-type":"application/json"},method=method)
 with http.open(q,timeout=15)as res:return json.loads(res.read())
# Same frozen217 input except cold salt and request header. No shrinking math case.
body=json.loads((r.parent/"GLM-RUN-0217/client_final_add2_3.body").read_text());body["cache_salt"]=r.name+"-diagnostic-add2";raw=json.dumps(body,ensure_ascii=False,separators=(",",":")).encode();(r/"diagnostic.body").write_bytes(raw)
ctl("/control/replicas/D1",method="DELETE")
try:
 guard();q=urllib.request.Request("http://127.0.0.1:8000/v1/chat/completions",data=raw,headers={"content-type":"application/json","accept-encoding":"identity","x-request-id":r.name+"-diagnostic-add2"});start=time.monotonic()
 try:
  with http.open(q,timeout=180)as res:wire=res.read();status=res.status
 except urllib.error.HTTPError as e:wire=e.read();status=e.code
 (r/"diagnostic.wire").write_bytes(wire)
 semantic=False;output=0
 if status==200:
  v=json.loads(wire);output=v["usage"]["completion_tokens"];semantic=v["choices"][0]["message"]["content"].strip()=="4"and v["choices"][0]["finish_reason"]=="stop"
 out=dict(at=utc(),measurement_kind="blocking_diagnostic_not_performance",HTTP_status=status,wall_s=time.monotonic()-start,body=ref(r/"diagnostic.body"),wire=ref(r/"diagnostic.wire"),semantic_pass=semantic,effective_output_tokens=output,Current=None)
 atomic_json(r/"diagnostic_client_summary.json",out);print(json.dumps(out))
finally:
 peers=json.loads((r/"restored/service_config.json").read_text())["environment"]["GLM_REPLICAS"];peers=json.loads(peers)if isinstance(peers,str)else peers;ctl("/control/replicas",next(x for x in peers if x["id"]=="D1"),"POST")
