import json,sys,time,hashlib,urllib.request,urllib.error,re,subprocess,shlex
from pathlib import Path
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc
from sse_observer import NativeSSEObserver
from owner_guard import start_watchdog,guard
start_watchdog()
r=Path(__file__).parent;http=urllib.request.build_opener(urllib.request.ProxyHandler({}));rows=[]
owners=json.loads((r.parent/"restored/adopted_model_identities.json").read_text())
def snapshot(label):
 guard()
 for key,x in owners.items():
  if key!="D0":continue
  for ep in ["health","metrics"]:
   with http.open("http://172.16.10."+x["host"]+":"+str(x["port"])+"/"+ep,timeout=10)as z:
    assert z.status==200;(r/(label+"_"+key+"."+ep)).write_bytes(z.read())
def req(name,key,body,kind,expected,min_prompt=None):
 guard();x=owners[key];f=r/(name+".body.json");atomic_json(f,body);raw=b"";start=time.monotonic();first=None
 row={"name":name,"owner":key,"kind":kind,"attempted_at":utc(),"request_body_sha256":hashlib.sha256(f.read_bytes()).hexdigest(),"expected_output_tokens":expected,"completed":False,"effective_public_output_credit":0};rows.append(row);atomic_json(r/"attempts.json",rows)
 request=urllib.request.Request("http://172.16.10."+x["host"]+":"+str(x["port"])+"/v1/chat/completions",data=f.read_bytes(),headers={"Content-Type":"application/json","X-Request-ID":"GLM-RUN-0200-0-"+name})
 try:
  with http.open(request,timeout=180)as response:
   row["http_status"]=response.status
   while True:
    block=response.read1(65536)
    if not block:break
    raw+=block
    if first is None and body.get("stream"):
     for frame in raw.replace(b"\r\n",b"\n").split(b"\n\n"):
      data=b"\n".join(v[5:].removeprefix(b" ")for v in frame.splitlines()if v.startswith(b"data:"))
      try:o=json.loads(data)
      except (ValueError,UnicodeDecodeError):continue
      for c in o.get("choices",[]):
       d=c.get("delta")or{}
       if any(d.get(z)for z in ["content","reasoning","reasoning_content","tool_calls"]):first=time.monotonic();break
  assert row["http_status"]==200
  if body.get("stream"):
   obs=NativeSSEObserver(max_bytes=2097152,collect_contract=True);obs.feed(raw);contract=obs.contract();assert contract["done"]and not contract["native_error"]and not contract["unknown"];usage=contract["usage"];assert contract["finish_reasons"].get("0")=="length";value=None
  else:
   value=json.loads(raw);assert not value.get("error");usage=value["usage"];assert len(value["choices"])==1 and value["choices"][0]["finish_reason"]=="length";contract={"nativeJSON":True,"finish_reason":"length"}
  assert type(usage["completion_tokens"])is int and usage["completion_tokens"]==expected and usage["total_tokens"]==usage["prompt_tokens"]+usage["completion_tokens"]
  if min_prompt is not None:assert min_prompt<usage["prompt_tokens"]<144352
  row.update(completed=True,usage=usage,contract=contract,wall_s=time.monotonic()-start,ttft_s=None if first is None else first-start,effective_public_output_credit=0 if kind=="internal_P_KV_helper"else expected)
  return value
 except urllib.error.HTTPError as e:
  raw=e.read();row.update(http_status=e.code,error="nativeHTTPerror");raise
 except BaseException as e:row.update(error_type=type(e).__name__,error=str(e));raise
 finally:
  f=r/(name+".wire");f.write_bytes(raw);row["wire"]={"path":str(f),"bytes":len(raw),"sha256":hashlib.sha256(raw).hexdigest()};atomic_json(r/"attempts.json",rows)
try:
 snapshot("initial")
 for key in["D0"]:
  short=dict(model="glm-52",messages=[dict(role="user",content="List three properties of a correct inference service.")],temperature=0,seed=20260930,ignore_eos=True,max_tokens=32,stream=True,stream_options=dict(include_usage=True),return_token_ids=True,cache_salt=r.parent.name+"-"+r.name+"-short-"+key)
  snapshot(key+"_short_before");req("short_"+key,key,short,"public_standalone_short",32);assert rows[-1]["usage"]["prompt_tokens"]==21;snapshot(key+"_short_after")
 provenance=json.loads((r.parent/"input_provenance.json").read_text());assert hashlib.sha256((r.parent/"canonical_81932.body.json").read_bytes()).hexdigest()==provenance["sha256"]
 canonical=json.loads((r.parent/"canonical_81932.body.json").read_text());canonical.update(max_tokens=64,stream=True,stream_options={"include_usage":True});canonical.pop("kv_transfer_params",None)
 for n,key in enumerate(["D0"]):
  body=dict(canonical);body["cache_salt"]="GLM-RUN-0200-0-D-local-full-"+str(n);body["return_token_ids"]=True
  snapshot(key+"_before");req("full_"+key,key,body,"public_standalone_cold_prefill",64)
  assert rows[-1]["usage"]["prompt_tokens"]==81932
  snapshot(key+"_after")
 out=dict(at=utc(),measurement_valid=True,functional_acceptance=True,requests=rows,effective_public_output_tokens=sum(x["effective_public_output_credit"]for x in rows),verdict="INCONCLUSIVE",limits=["HOST controller verifiesbothnative roots/allNPU32 beforeandafterclient; containercontains noSSH anddoesnotprobeHOSTpid. Newindependent200 D0 PP4TP4DCP4 K3on166/K1on167, c1/V5/emptytoken guard, short32+NEWcold81932to64; no capacityclaim"])
 atomic_json(r/"pilot_summary.json",out);print(json.dumps(out))
except BaseException as e:
 atomic_json(r/"pilot_summary.json",dict(at=utc(),measurement_valid=False,functional_acceptance=False,error_type=type(e).__name__,error=str(e),requests=rows,effective_public_output_tokens=sum(x["effective_public_output_credit"]for x in rows),limits=["Allactualattempts retained, no successfulfullE2E/capacity credit"]));raise
