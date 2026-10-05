import json,sys,urllib.request,time,hashlib
from pathlib import Path
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json
from sse_observer import NativeSSEObserver
import os
from owner_guard import start_watchdog
start_watchdog()
root=Path(__file__).parent/("tools_"+os.environ["GLM_TOOL_REPLICA"]);root.mkdir()
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
url=os.environ["GLM_TOOL_URL"]
tools=[{"type":"function","function":{"name":"get_weather","description":"Retrieve current weather for a city.","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"],"additionalProperties":False}}}]
rows=[]
try:
 for name,choice,stream in [("auto","auto",True),("required","required",False),("named",{"type":"function","function":{"name":"get_weather"}},True),("none","none",False)]:
  body={"model":"glm-52","messages":[{"role":"user","content":"Call get_weather for Shanghai." if name!="none" else "Reply with exactly this literal text: <tool_call>get_weather</tool_call>"}],"tools":tools,"tool_choice":choice,"stream":stream,"max_tokens":1024,"temperature":0,"seed":20260930,"chat_template_kwargs":{"enable_thinking":False},"return_token_ids":True,"cache_salt":Path(__file__).parent.name+"-"+os.environ["GLM_TOOL_REPLICA"]+"-"+name}
  if stream:body["stream_options"]={"include_usage":True}
  p=root/(name+".body.json");atomic_json(p,body);started=time.monotonic()
  attempt={"name":name,"stream":stream,"valid":False,"status":"started"};rows.append(attempt)
  req=urllib.request.Request(url+"/v1/chat/completions",data=p.read_bytes(),headers={"Content-Type":"application/json","X-Request-ID":Path(__file__).parent.name+"-tool-"+os.environ["GLM_TOOL_REPLICA"]+"-"+name})
  rawpath=root/(name+".wire");status=None;raw=b""
  try:
   with opener.open(req,timeout=180) as response:
    status=response.status
    while block:=response.read1(65536):raw+=block
  except urllib.error.HTTPError as error:
   status=error.code;raw=error.read()
   attempt.update(status=status,native_error=True)
  finally:
   rawpath.write_bytes(raw);attempt["wire"]={"path":str(rawpath),"bytes":len(raw),"sha256":hashlib.sha256(raw).hexdigest()}
  assert status==200
  calls={};content=[];token_ids=[];usage=None;finish={};done=not stream
  if stream:
   observer=NativeSSEObserver(collect_contract=True);observer.feed(raw);contract=observer.contract()
   assert not contract["unknown"] and not contract["native_error"] and contract["done"]
   usage=contract["usage"];finish=contract["finish_reasons"];done=contract["done"]
   for frame in raw.replace(b"\r\n",b"\n").split(b"\n\n"):
    data=b"\n".join(x[5:].removeprefix(b" ") for x in frame.splitlines() if x.startswith(b"data:"))
    if not data or data==b"[DONE]":continue
    event=json.loads(data)
    for c in event.get("choices",[]):
     token_ids.extend(c.get("token_ids") or []);d=c.get("delta",{});content.append(d.get("content") or "")
     for t in d.get("tool_calls") or []:
      row=calls.setdefault((c["index"],t.get("index",0)),{"name":"","arguments":""})
      for key in ["name","arguments"]:row[key]+=(t.get("function") or {}).get(key) or ""
  else:
   value=json.loads(raw);assert not value.get("error");usage=value["usage"]
   for c in value["choices"]:
    token_ids.extend(c.get("token_ids") or []);finish[str(c["index"])]=c["finish_reason"];msg=c["message"];content.append(msg.get("content") or "")
    for i,t in enumerate(msg.get("tool_calls") or []):calls[c["index"],i]=t["function"]
  assert isinstance(usage,dict) and type(usage["completion_tokens"]) is int and usage["completion_tokens"]>0 and usage["total_tokens"]==usage["prompt_tokens"]+usage["completion_tokens"]
  assert len(token_ids)==usage["completion_tokens"]
  if name=="none":
   assert not calls and "".join(content) and finish.get("0") in ("stop","length")
  else:
   assert len(calls)==1 and finish.get("0") in ("stop","tool_calls")
   call=next(iter(calls.values()));assert call["name"]=="get_weather" and json.loads(call["arguments"])=={"city":"Shanghai"}
  attempt.update({"name":name,"stream":stream,"status":status,"valid":True,"usage":usage,"finish_reasons":finish,"done":done,"tools":list(calls.values()),"content_chars":len("".join(content)),"token_ids":token_ids,"token_ids_sha256":hashlib.sha256(json.dumps(token_ids).encode()).hexdigest(),"literal_tool_marker_preserved":"<tool_call>" in "".join(content),"wall_s":time.monotonic()-started,"wire":{"path":str(rawpath),"bytes":len(raw),"sha256":hashlib.sha256(raw).hexdigest()}})
  atomic_json(root/"summary.json",{"valid":False,"status":"running","requests":rows})
except Exception as error:
 atomic_json(root/"summary.json",{"valid":False,"status":"failed","requests":rows,"error_type":type(error).__name__,"error":str(error)})
 raise
finally:
 end=time.monotonic()+60;idle=False
 while time.monotonic()<end:
  import re
  values=[]
  for key,nativeurl in json.loads(os.environ["GLM_TOOL_METRICS_URLS"]).items():
   with opener.open(nativeurl+"/metrics",timeout=5)as response:metrics=response.read().decode()
   (root/("final_"+key+".metrics")).write_text(metrics)
   values.extend(float(x)for name in["num_requests_running","num_requests_waiting"]for x in re.findall(r"^vllm:"+name+r"(?:\{[^\n]*\})?\s+([0-9.eE+-]+)",metrics,re.M))
  if len(values)>=2 and all(x==0 for x in values):idle=True;break
  time.sleep(2)
 cleanup={"native_idle":idle};atomic_json(root/"cleanup.json",cleanup)
 assert idle,"tool request cleanup not idle"
atomic_json(root/"summary.json",{"valid":True,"status":"completed","requests":rows,"effective_output_tokens":sum(x["usage"]["completion_tokens"] for x in rows),"limits":["Finite native complete gateway tool API smoke; no performance/capacity KEEP","CPU exact marker regression separate; literal live generation occurrence recorded"],"cleanup":cleanup})
print(json.dumps({"valid":True,"cases":len(rows),"outputs":sum(x["usage"]["completion_tokens"] for x in rows)}))
