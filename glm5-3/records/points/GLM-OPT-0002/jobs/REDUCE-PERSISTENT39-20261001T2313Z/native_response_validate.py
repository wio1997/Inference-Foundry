import json,pathlib,typing,importlib,sys,hashlib
from pydantic import BaseModel
from vllm.entrypoints.openai.responses.protocol import ResponsesResponse
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json
r=pathlib.Path(__file__).parent
classes={}
# Native's StreamingResponsesResponse alias omits some actual generic renderer
# classes. Recover actual installed response event models, with native overrides.
for modname in ["openai.types.responses","vllm.entrypoints.openai.responses.streaming_events","vllm.entrypoints.openai.responses.protocol"]:
 module=importlib.import_module(modname)
 for name,cls in vars(module).items():
  if not isinstance(cls,type) or not issubclass(cls,BaseModel) or "type"not in cls.model_fields:continue
  for tag in typing.get_args(cls.model_fields["type"].annotation):
   if isinstance(tag,str) and (tag.startswith("response.") or tag=="error"):classes[tag]=cls
atomic_json(r/"native_event_models.json",{tag:cls.__module__+"."+cls.__name__ for tag,cls in classes.items()})
def parse_frame(frame):
 lines=frame.decode("utf8").splitlines();data="\n".join(x[5:].lstrip(" ")for x in lines if x.startswith("data:"));event="\n".join(x[6:].lstrip(" ")for x in lines if x.startswith("event:"))
 if not data:return None
 obj=json.loads(data);tag=obj.get("type");assert tag in classes,("unknown native event",tag)
 classes[tag].model_validate(obj);assert event==tag and tag!="error" and not obj.get("error"),obj
 return obj
def committed(value):
 obj=ResponsesResponse.model_validate(value);assert obj.status in ["completed","incomplete"] and not value.get("error") and obj.usage is not None
 usage=obj.usage.model_dump();assert usage["output_tokens"]>0 and usage["input_tokens"]>0 and usage["total_tokens"]==usage["input_tokens"]+usage["output_tokens"]
 assert obj.max_output_tokens==64 and usage["output_tokens"]<=64
 return usage


source=pathlib.Path("/data/tiankuan/wio/Inference-Foundry/glm5-3/records/points/GLM-OPT-0002/runs/GLM-RUN-0039")
rows=json.loads((source/"restart/attempts.json").read_text());out={}
for row in rows:
 if not row["id"].startswith("responses"):continue
 raw=pathlib.Path(row["wire"]["path"]).read_bytes();assert hashlib.sha256(raw).hexdigest()==row["wire"]["sha256"]
 if row["id"]=="responses_sse":
  events=[parse_frame(f)for f in raw.replace(b"\r\n",b"\n").split(b"\n\n")if f.strip()];events=[x for x in events if x is not None]
  assert events[0]["type"]=="response.created" and [x["sequence_number"]for x in events]==list(range(len(events)))
  terminal=[e for e in events if e["type"]=="response.completed"];assert len(terminal)==1
  value=terminal[0]["response"];assert events==json.loads((source/"restart/responses_sse.events.json").read_text())
  assert len({e["response"]["id"]for e in events if isinstance(e.get("response"),dict)})==1
 else:value=json.loads(raw)
 usage=committed(value);assert value["status"]=="completed" and "".join(c.get("text","")for o in value["output"]for c in o.get("content",[])if c.get("type")=="output_text").strip()=="READY"
 out[row["id"]]={"usage":usage,"status":value["status"],"wire_sha256":row["wire"]["sha256"],"text":"READY"}
atomic_json(r/"native_responses_validation.json",{"valid":True,"validated":out,"native_inference_calls":0,"scope":"offline native typed protocol validation of originalwire; no HTTP/models/EngineCore/collectives"})
print(json.dumps({"valid":True,"native_inference_calls":0,"responses":len(out)}))
