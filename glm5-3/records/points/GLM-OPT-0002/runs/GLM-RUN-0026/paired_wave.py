"""Finite dynamic long-context complete-request pilot, no capacity/KEEP claim."""
import asyncio,fcntl,hashlib,json,os,re,signal,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc
from loadgen import execute
BACKENDS=[("DP0","http://172.16.10.166:9081"),("DP1","http://172.16.10.167:9900")]
async def main(root,run_id,policy="active_count"):
 import httpx
 root=Path(root);root.mkdir();parent=root.parent;owner=json.loads(Path("/data/tiankuan/wio/glm52-pd/controller-owner.json").read_text())
 assert owner["run_id"]==run_id and owner["status"]=="running" and owner["owner"]["boot_id"]==Path("/proc/sys/kernel/random/boot_id").read_text().strip()
 for name in [".controller.lock",".formal-test.lock"]:
  with (Path("/data/tiankuan/wio/glm52-pd")/name).open("a+") as f:
   try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
   except BlockingIOError:pass
   else:raise RuntimeError("controller lock unowned")
 results={};events=[];gateway=None;log=None;observer=None;stop=asyncio.Event()
 def event(kind,**fields):events.append({"at":utc(),"event":kind,**fields});atomic_json(root/"pilot_events.json",events)
 async with httpx.AsyncClient(trust_env=False,timeout=10) as client:
  async def metrics(label):
   gauges={};at=utc()
   for name,url in BACKENDS:
    r=await client.get(url+"/metrics");r.raise_for_status();p=root/(label+"_"+name+".metrics");p.write_text(r.text)
    values={}
    for key in ["num_requests_running","num_requests_waiting"]:
     found=re.findall(r"^vllm:"+key+r"(?:\{[^\n]*\})?\s+([0-9.eE+-]+)",r.text,re.M)
     if not found:raise RuntimeError("native gauges missing")
     values[key]=sum(float(v) for v in found)
    gauges[name]=values
   event("metrics",label=label,gauges=gauges,observed_at=at);return gauges
  async def idle(label):
   end=time.monotonic()+60
   while True:
    g=await metrics(label)
    if all(all(v==0 for v in x.values()) for x in g.values()):return
    if time.monotonic()>end:raise RuntimeError("native idle not verified")
    await asyncio.sleep(1)
  async def observe():
   index=0
   while not stop.is_set():
    try:await metrics("sample"+str(index))
    except Exception as e:event("observer_error",error=str(e))
    index+=1
    try:await asyncio.wait_for(stop.wait(),timeout=2)
    except asyncio.TimeoutError:pass
  async def load(name,cases,endpoint):
   d=root/name;d.mkdir();requests=[]
   for payload,ident,arrival,expected in cases:
    path=d/(ident+".body.json");atomic_json(path,payload);requests.append({"id":ident,"arrival_s":arrival,"body_path":str(path),"timeout_s":600,"expected":expected,"request_header_id":run_id+"-"+root.name+"-"+ident})
   plan={"kind":"diagnostic","endpoint":endpoint,"requests":requests};atomic_json(d/"plan.json",plan)
   result=await execute(plan,d);results[name]=result
   event("load_completed",name=name,valid=result["valid"],outputs=result["effective_output_tokens"],wall_s=result["elapsed_s"])
   if not result["valid"]:raise RuntimeError(name+" real request acceptance failed")
   return result
  try:
   await idle("initial");observer=asyncio.create_task(observe())
   long=json.loads((parent.parent/"GLM-RUN-0012/pd_compat/request.body.json").read_text());long["max_tokens"]=2048
   # Separate native controls observe cache reuse; no assumed warm/cold TTFT claim.
   for name,source in [(name,parent.parent/"GLM-RUN-0025"/("control_"+name+"/load_result.json")) for name,url in BACKENDS]:
    data=json.loads(source.read_text())
    if not data["valid"] or data["effective_output_tokens"]!=2048:raise RuntimeError("reused native control invalid")
    results["control_"+name]=data;event("reused_control",source=str(source),sha256=hashlib.sha256(source.read_bytes()).hexdigest())
   log=(root/"gateway.log").open("wb");env=dict(os.environ,GLM_REPLICAS=json.dumps([{"id":n,"url":u} for n,u in BACKENDS]),GLM_ROUTER_TRACE_PATH=str(root/"router_trace.jsonl"),GLM_PLACEMENT_POLICY=policy)
   gateway=subprocess.Popen([sys.executable,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime/replica_gateway.py","--port","8002"],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   for _ in range(100):
    if gateway.poll() is not None:raise RuntimeError("gateway exited")
    try:
     r=await client.get("http://127.0.0.1:8002/healthcheck")
     if r.status_code==200:break
    except httpx.HTTPError:pass
    await asyncio.sleep(.1)
   else:raise RuntimeError("gateway readiness timeout")
   event("gateway_started",pid=gateway.pid)
   previous=parent.parent/"GLM-RUN-0012/e2e"
   short=json.loads((previous/"mixed_open_arrivals/0.body.json").read_text())
   medium=json.loads((previous/"mixed_open_arrivals/5.body.json").read_text())
   short2=json.loads((previous/"local_P166/0.body.json").read_text());short2["max_tokens"]=256
   long2=dict(long);long2["max_tokens"]=4096
   cases=[(short,"short0",0,{"prompt_tokens":205,"output_tokens":128}),(dict(long),"long0",0,{"prompt_tokens":81932,"output_tokens":2048}),(dict(long),"long1",.5,{"prompt_tokens":81932,"output_tokens":2048}),(medium,"medium",2,{"prompt_tokens":8205,"output_tokens":512}),(long2,"long2",6,{"prompt_tokens":81932,"output_tokens":4096}),(short2,"short1",10,{"prompt_tokens":334,"output_tokens":256})]
   await load("dynamic",cases,"http://127.0.0.1:8002/v1/chat/completions")
   state=(await client.get("http://127.0.0.1:8002/control/replicas")).json();atomic_json(root/"final_placement.json",state)
   if any(r["active_requests"] for r in state["replicas"]):raise RuntimeError("leases not released")
   await idle("final")
   summary={"run_id":run_id,"policy":policy,"valid":True,"kind":"diagnostic","completed_at":utc(),"load_results":{k:{field:v[field] for field in ["valid","request_count","successful_inference_requests","effective_output_tokens","elapsed_s","effective_tps"]} for k,v in results.items()},"verdict":"INCONCLUSIVE","limits":["finite dynamic pilot is not stable capacity or formal KEEP","Both native DP ranks share EP32/hierarchy communication; weight/cache/collective deployment change distinct from code gain","cache state observed with native metrics and usage, not presumed","metrics observer2sec on; no prof; no token accuracy/quantization comparison"]}
   atomic_json(root/"pilot_summary.json",summary);return 0
  except BaseException as e:
   event("failure",error_type=type(e).__name__,message=str(e));atomic_json(root/"pilot_summary.json",{"run_id":run_id,"valid":False,"verdict":"INVALID","error":str(e),"load_results":{k:{"valid":v["valid"],"effective_output_tokens":v["effective_output_tokens"]} for k,v in results.items()}});raise
  finally:
   stop.set()
   if observer:await observer
   if gateway:
    if gateway.poll() is None:
     os.killpg(gateway.pid,signal.SIGTERM)
     try:await asyncio.to_thread(gateway.wait,10)
     except subprocess.TimeoutExpired:os.killpg(gateway.pid,signal.SIGKILL);gateway.wait()
    event("gateway_stopped",exit_code=gateway.returncode)
   if log:log.close()

