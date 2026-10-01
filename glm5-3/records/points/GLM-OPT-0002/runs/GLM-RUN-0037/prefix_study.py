"""Finite matched-prefix prefill budget diagnostic. No formal/stable-capacity credit."""
import asyncio,json,sys,hashlib,re,urllib.request,os,fcntl
from datetime import datetime,timezone
from pathlib import Path
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/adapters/aisbench")
from phase_runner import atomic_json,utc
from loadgen import execute
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
prior=root.parent/"GLM-RUN-0023/prefix_study"
files=[]
for i in range(3):
 p=prior/("canonical_"+str(i)+".body.json");raw=p.read_bytes()
 (d/p.name).write_bytes(raw);files.append({"path":str(p),"sha256":hashlib.sha256(raw).hexdigest(),"new_body":str(d/p.name)})
assert len(files)==3
atomic_json(d/"dataset_identity.json",{"reused_canonical":files,"sampling":"Run23 canonical seed omitted/native1024 temp0","actual_inputs":[73740,81932,81932],"expected_outputs":[1,2048,2048],"new_credit":True,"limits":["same requests but topology/cache/native trajectory changes, not accuracy comparison"]})
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
   await asyncio.gather(one("DP0","http://172.16.10.166:9081",i),one("DP1","http://172.16.10.166:9082",i),one("DP2","http://172.16.10.167:9900",i),one("DP3","http://172.16.10.167:9901",i))
  for i in range(3):
   pair=[x for x in results if x["index"]==i];assert len(pair)==4 and len({x["body_sha256"] for x in pair})==1
  summary={"run_id":root.name,"valid":True,"kind":"diagnostic","new_requests":12,"effective_output_tokens":16388,"requests":results,"verdict":"INCONCLUSIVE","completed_at":utc(),"limits":["Actual Run37 DP4/TP8/DCP8/EP32 budget16384 gmu.80 K5/FULL, same native operator implementation","2 novel full tails per replica, native usage/length/DONE/idle","not formal61440 performance or repeated KEEP/stable capacity; host/cache/source state conditions retained"]}
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
