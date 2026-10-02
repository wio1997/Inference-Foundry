import json,sys,time,hashlib,urllib.request,urllib.error,re,subprocess,shlex
from pathlib import Path
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc
from sse_observer import NativeSSEObserver
from owner_guard import start_watchdog,guard
start_watchdog()
r=Path(__file__).parent;http=urllib.request.build_opener(urllib.request.ProxyHandler({}));rows=[]
owners=json.loads((r/"adopted_model_identities.json").read_text())
def ownercheck():
 for x in owners.values():
  code="import pathlib,json;p=pathlib.Path('/proc/"+str(x["pid"])+"');s=(p/'stat').read_text();print(json.dumps({'boot_id':pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'start_ticks':s[s.rfind(')')+2:].split()[19],'argv':[x.decode()for x in(p/'cmdline').read_bytes().split(bytes([0]))if x]}))";args=["python3","-c",code]
  if x["host"]=="167":args=["ssh","-o","BatchMode=yes","root@172.16.10.167",shlex.join(args)]
  p=subprocess.run(args,capture_output=True,timeout=20);p.check_returncode();i=json.loads(p.stdout);assert {k:i[k]for k in ["boot_id","start_ticks"]}==x["identity"]and i["argv"]==x["argv"]
def snapshot(label):
 guard();ownercheck()
 for key,x in owners.items():
  for ep in ["health","metrics"]:
   with http.open("http://172.16.10."+x["host"]+":"+str(x["port"])+"/"+ep,timeout=10)as z:
    assert z.status==200;(r/(label+"_"+key+"."+ep)).write_bytes(z.read())
def req(name,key,body,kind,expected,min_prompt=None):
 guard();ownercheck();x=owners[key];f=r/(name+".body.json");atomic_json(f,body);raw=b"";start=time.monotonic();first=None
 row={"name":name,"owner":key,"kind":kind,"attempted_at":utc(),"request_body_sha256":hashlib.sha256(f.read_bytes()).hexdigest(),"expected_output_tokens":expected,"completed":False,"effective_public_output_credit":0};rows.append(row);atomic_json(r/"attempts.json",rows)
 request=urllib.request.Request("http://172.16.10."+x["host"]+":"+str(x["port"])+"/v1/chat/completions",data=f.read_bytes(),headers={"Content-Type":"application/json","X-Request-ID":"GLM-RUN-0062-"+name})
 try:
  with http.open(request,timeout=600)as response:
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
   obs=NativeSSEObserver(collect_contract=True);obs.feed(raw);contract=obs.contract();assert contract["done"]and not contract["native_error"]and not contract["unknown"];usage=contract["usage"];assert contract["finish_reasons"].get("0")=="length";value=None
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
 canonical=json.loads((r/"canonical_81932.body.json").read_text());canonical.update(max_tokens=64,stream=True,stream_options={"include_usage":True});canonical.pop("kv_transfer_params",None)
 for n,key in enumerate(["D0","D1"]):
  body=dict(canonical);body["cache_salt"]="GLM-RUN-0062-D-local-full-"+str(n);body["return_token_ids"]=True
  snapshot(key+"_before");req("full_"+key,key,body,"public_D_cold_full_localprefill",64)
  assert rows[-1]["usage"]["prompt_tokens"]==81932
  snapshot(key+"_after")
 out=dict(at=utc(),measurement_valid=True,functional_acceptance=True,requests=rows,effective_public_output_tokens=sum(x["effective_public_output_credit"]for x in rows),verdict="INCONCLUSIVE",limits=["Two finite directnativeD-only fullprefill windows; no publicgateway/PD/stablecapacity/KEEP","D56 same32physicalworkers/API/K5Graph/nativeoperators; P61 stopped, memory/co-resident load changed; no isolatedchunk/HCCL effect","Independentnativecache_salt tokencontract/coldmiss counters require terminalauditor; failedwork zerocredit"])
 atomic_json(r/"pilot_summary.json",out);print(json.dumps(out))
except BaseException as e:
 atomic_json(r/"pilot_summary.json",dict(at=utc(),measurement_valid=False,functional_acceptance=False,error_type=type(e).__name__,error=str(e),requests=rows,effective_public_output_tokens=sum(x["effective_public_output_credit"]for x in rows),limits=["Allactualattempts retained, no successfulfullE2E/capacity credit"]));raise
