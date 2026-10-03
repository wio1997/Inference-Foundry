"""Exact native Chat token reuse; benchmark messages/sampling remain unchanged."""
from pathlib import Path
import json,hashlib,uuid
from ais_bench.benchmark.models import VLLMCustomAPIChatStream
from ais_bench.benchmark.registry import MODELS
J=Path("/data/tiankuan/wio/Inference-Foundry/glm5-3/records/points/GLM-OPT-0002/jobs/TOKEN-INPUT-CPU3-20261003T0837Z")
raw=(J/"reduction.json").read_bytes();assert hashlib.sha256(raw).hexdigest()=="dfd0544bee78d2a14064d4330ad46037fc9084918463420c133e1437ef5bb949"
v=json.loads(raw);assert v["valid"]and v["native32_same"]and v["native_counter_deltas0"]
def key(messages):return hashlib.sha256(json.dumps(messages,ensure_ascii=False,separators=(",",":")).encode()).hexdigest()
RECORDS={}
for row in v["records"]:
 b=Path(row["ids"]["path"]).read_bytes();assert len(b)==row["ids"]["bytes"]and hashlib.sha256(b).hexdigest()==row["ids"]["sha256"];ids=json.loads(b);assert len(ids)==row["count"]
 b=Path(row["request"]["path"]).read_bytes();assert hashlib.sha256(b).hexdigest()==row["request"]["sha256"];messages=json.loads(b)["messages"]
 RECORDS[key(messages)]=(row,ids)
@MODELS.register_module()
class NativeTokenReuseChat(VLLMCustomAPIChatStream):
 async def get_request_body(self,input,max_out_len,output,**args):
  body=await super().get_request_body(input,max_out_len,output,**args)
  assert isinstance(body,dict)and "kv_transfer_params"not in body and not body.get("tools")
  row,ids=RECORDS[key(body["messages"])]
  assert body["max_tokens"]==(1 if row["label"]=="warmup0"else 64)
  body["kv_transfer_params"]={"prompt_token_ids":ids}
  b=json.dumps(body).encode();folder=Path.cwd()/"actual_request_bodies";folder.mkdir(exist_ok=True)
  path=folder/(uuid.uuid4().hex+".body");path.write_bytes(b)
  receipt=dict(label=row["label"],body=dict(path=str(path),bytes=len(b),sha256=hashlib.sha256(b).hexdigest()),native_prompt_ids_source=row["ids"],count=row["count"],messages_preserved=True,sampling_preserved=True)
  phase="warmup"if max_out_len==1 else"full"
  with (Path.cwd()/(phase+"_request_index.jsonl")).open("a")as f:f.write(json.dumps(receipt)+"\n")
  return body
