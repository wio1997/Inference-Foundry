import asyncio,json,sys,re,fcntl
from datetime import datetime,timezone
from pathlib import Path
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc
from loadgen import execute
r=Path(__file__).parent
owner=json.loads(Path("/data/tiankuan/wio/glm52-pd/controller-owner.json").read_text())
def guard():
 c=json.loads(Path("/data/tiankuan/wio/glm52-pd/controller-owner.json").read_text());age=(datetime.now(timezone.utc)-datetime.fromisoformat(c["heartbeat_at"])).total_seconds()
 assert c["run_id"]==r.name and c["status"]=="running" and c["owner"]==owner["owner"] and -2<=age<=30
 for n in [".controller.lock",".formal-test.lock"]:
  with (Path("/data/tiankuan/wio/glm52-pd")/n).open("a+") as f:
   try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
   except BlockingIOError:pass
   else:raise RuntimeError("unique controller lock lost")
async def main():
 import httpx
 async with httpx.AsyncClient(trust_env=False,timeout=10) as client:
  async def one(rank,node,port):
   d=r/("control_DP"+str(rank));d.mkdir()
   async def metrics(name):
    reply=await client.get("http://172.16.10."+node+":"+str(port)+"/metrics");reply.raise_for_status();(d/(name+".metrics")).write_text(reply.text)
    for key in ["num_requests_running","num_requests_waiting"]:
     vals=re.findall(r"^vllm:"+key+r"(?:\{[^\n]*\})?\s+([0-9.eE+-]+)",reply.text,re.M);assert vals and all(float(v)==0 for v in vals)
   await metrics("initial")
   body=(r.parent/"GLM-RUN-0022/control_D167/request.body.json").read_bytes();p=d/"request.body.json";p.write_bytes(body)
   plan={"kind":"diagnostic","endpoint":"http://172.16.10."+node+":"+str(port)+"/v1/chat/completions","requests":[{"id":"cold_DP"+str(rank),"arrival_s":0,"body_path":str(p),"timeout_s":1200,"expected":{"prompt_tokens":81932,"output_tokens":2048},"request_header_id":r.name+"-cold-DP"+str(rank)}]}
   atomic_json(d/"plan.json",plan);result=await execute(plan,d);row=result["requests"][0]
   assert result["valid"] and row["done"] and row["finish_reasons"]=={"0":"length"} and not row["error"]
   for _ in range(60):
    try:await metrics("final");break
    except AssertionError:await asyncio.sleep(1)
   else:raise RuntimeError("cold native idle not observed")
   return {"rank":rank,"result":str(d/"load_result.json"),"body_sha256":row["body_sha256"],"ttft_s":row["ttft_s"],"tpot_s":row["tpot_s"],"usage":row["usage"]}
  rows=await asyncio.gather(one(0,"166",9081),one(1,"167",9900));assert len({v["body_sha256"] for v in rows})==1
  atomic_json(r/"cold_summary.json",{"valid":True,"kind":"diagnostic","requests":rows,"new_completed_inference_requests":2,"effective_output_tokens":4096,"cache":"both fresh engine, actual initial/final native counters; Graph startup separate","limits":["coupled EP32 with two simultaneous identical long prompts; not isolated per-node or full61440 comparison","native operator implementation unchanged; placement/communication/weights/cache change"]})
async def owned():
 async def watch():
  while True:guard();await asyncio.sleep(2)
 run=asyncio.create_task(main());watcher=asyncio.create_task(watch())
 try:
  done,_=await asyncio.wait([run,watcher],return_when=asyncio.FIRST_COMPLETED)
  if watcher in done:watcher.result();raise RuntimeError("owner watcher stopped")
  run.result()
 finally:
  run.cancel();watcher.cancel();await asyncio.gather(run,watcher,return_exceptions=True)
asyncio.run(owned())
