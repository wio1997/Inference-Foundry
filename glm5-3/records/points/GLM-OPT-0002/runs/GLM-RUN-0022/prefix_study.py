"""Finite matched-prefix prefill budget diagnostic. No formal/stable-capacity credit."""
import asyncio,json,sys,hashlib,re,urllib.request,os,fcntl
from datetime import datetime,timezone
from pathlib import Path
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/adapters/aisbench")
from phase_runner import atomic_json,utc
from loadgen import execute
from dataset_generator import create_prefix_dataset
root=Path(__file__).parent;d=root/"prefix_study";d.mkdir()
owner=json.loads(Path("/data/tiankuan/wio/glm52-pd/controller-owner.json").read_text())
def verify_owner():
 current=json.loads(Path("/data/tiankuan/wio/glm52-pd/controller-owner.json").read_text())
 age=(datetime.now(timezone.utc)-datetime.fromisoformat(current["heartbeat_at"])).total_seconds()
 assert current["run_id"]==root.name and current["status"]=="running" and current["owner"]==owner["owner"] and -2<=age<=30
 assert current["owner"]["boot_id"]==Path("/proc/sys/kernel/random/boot_id").read_text().strip()
 for name in [".controller.lock",".formal-test.lock"]:
  with (Path("/data/tiankuan/wio/glm52-pd")/name).open("a+") as lock:
   try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
   except BlockingIOError:pass
   else:raise RuntimeError("unique controller lock lost")
verify_owner()
model="/data/tiankuan/wio/GLM-5.2-w8a8"
prefix_path,full_path=create_prefix_dataset(model,81920,2,str(d/"dataset"),1,.9,20261002,1)
prefix=json.loads(Path(prefix_path).read_text().splitlines()[0])["question"]
full=[json.loads(l)["question"] for l in Path(full_path).read_text().splitlines()]
from transformers import AutoTokenizer
tok=AutoTokenizer.from_pretrained(model,trust_remote_code=True)
def body(text,n):
 x={"model":"glm-52","messages":[{"role":"user","content":text}],"max_tokens":n,"temperature":0,"ignore_eos":True,"stream":True,"stream_options":{"include_usage":True}}
 actual=len(tok.apply_chat_template(x["messages"],tokenize=True,add_generation_prompt=True))
 assert actual==(73740 if n==1 else 81932),(actual,n)
 return x
payloads=[body(prefix,1)]+[body(x,2048) for x in full]
for i,b in enumerate(payloads):atomic_json(d/("canonical_"+str(i)+".body.json"),b)
atomic_json(d/"dataset_identity.json",{"seed":20261002,"raw_input":81920,"raw_prefix":73728,"actual_full":81932,"actual_prefix":73740,"sampling":"temperature0 ignore_eosTrue, requestseed omitted/native1024","payload_hashes":[hashlib.sha256((d/("canonical_"+str(i)+".body.json")).read_bytes()).hexdigest() for i in range(3)],"limits":["fresh deterministic seed, not same output trajectory as Run21","cache residency observed without reset; no assumed full hit","different hosts and new D deployment; finite conditional budget comparison"]})
async def main():
 import httpx
 results=[]
 async with httpx.AsyncClient(trust_env=False,timeout=10) as c:
  async def metrics(base,label):
   r=await c.get(base+"/metrics");r.raise_for_status();p=d/(label+".metrics");p.write_text(r.text)
   for key in ["num_requests_running","num_requests_waiting"]:
    v=re.findall(r"^vllm:"+key+r"(?:\{[^\n]*\})?\s+([0-9.eE+-]+)",r.text,re.M)
    assert v and all(float(x)==0 for x in v),(label,key)
  async def one(replica,url,index):
   p=d/(replica+"_"+str(index));p.mkdir()
   await metrics(url,replica+"_"+str(index)+"_before")
   b=p/"request.body.json";b.write_bytes((d/("canonical_"+str(index)+".body.json")).read_bytes())
   n=1 if index==0 else 2048;expected=73740 if index==0 else 81932
   plan={"kind":"diagnostic","endpoint":url+"/v1/chat/completions","requests":[{"id":"prefix" if index==0 else "tail"+str(index),"arrival_s":0,"body_path":str(b),"request_header_id":root.name+"-"+replica+"-"+str(index),"timeout_s":900,"expected":{"prompt_tokens":expected,"output_tokens":n}}]}
   atomic_json(p/"plan.json",plan);r=await execute(plan,p);row=r["requests"][0]
   assert r["valid"] and row["finish_reasons"]=={"0":"length"} and row["done"] and not row["error"],replica+" native strict diagnostic failed"
   for _ in range(60):
    try:await metrics(url,replica+"_"+str(index)+"_after");break
    except AssertionError:await asyncio.sleep(1)
   else:raise RuntimeError("native post-request idle not observed")
   results.append({"replica":replica,"index":index,"result":str(p/"load_result.json"),"body_sha256":row["body_sha256"],"ttft_s":row["ttft_s"],"tpot_s":row["tpot_s"],"usage":row["usage"]})
  for i in range(3):
   verify_owner()
   await asyncio.gather(one("P166","http://172.16.10.166:9081",i),one("D167","http://172.16.10.167:9900",i))
  for i in range(3):
   pair=[x for x in results if x["index"]==i];assert len(pair)==2 and len({x["body_sha256"] for x in pair})==1
  summary={"run_id":root.name,"valid":True,"kind":"diagnostic","new_requests":6,"effective_output_tokens":8194,"requests":results,"verdict":"INCONCLUSIVE","completed_at":utc(),"limits":["P4096 retained K5/Graph, D8192 new K5/Graph, same operator implementation","2 novel full tails per replica, native usage/length/DONE/idle","not formal61440 performance or repeated KEEP/stable capacity; host/cache/source state conditions retained"]}
  atomic_json(d/"summary.json",summary);print(json.dumps(summary))
async def owned():
 async def watch():
  while True:
   verify_owner()
   await asyncio.sleep(2)
 run=asyncio.create_task(main());guard=asyncio.create_task(watch())
 try:
  done,pending=await asyncio.wait([run,guard],return_when=asyncio.FIRST_COMPLETED)
  if guard in done:guard.result();raise RuntimeError("owner guard stopped")
  run.result()
 finally:
  run.cancel();guard.cancel()
  await asyncio.gather(run,guard,return_exceptions=True)
asyncio.run(owned())
