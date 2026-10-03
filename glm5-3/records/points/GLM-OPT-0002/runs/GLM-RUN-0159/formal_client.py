"""One owned full-output native replica E2E; raw logs remain on the server."""
import argparse,asyncio,fcntl,json,os,re,signal,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc,same_process
BACKENDS=[("D0","http://172.16.10.166:9081"),("D1","http://172.16.10.167:9900")]
async def main(root,run_id,cache_mode="native_reset"):
 import httpx
 root=Path(root);root.mkdir();owner=json.loads(Path("/data/tiankuan/wio/glm52-pd/controller-owner.json").read_text())
 assert owner["run_id"]==run_id and owner["status"]=="running" and owner["owner"]["boot_id"]==Path("/proc/sys/kernel/random/boot_id").read_text().strip()
 for name in [".controller.lock",".formal-test.lock"]:
  with (Path("/data/tiankuan/wio/glm52-pd")/name).open("a+") as f:
   try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
   except BlockingIOError:pass
   else:raise RuntimeError("global task controller locks not owned")
 events=[];gateway=None;log=None;observer=None;watchdog=None;stop=asyncio.Event();process=None;accepted=False;drained=False
 def event(kind,**fields):
  events.append({"at":utc(),"event":kind,**fields});atomic_json(root/"formal_events.json",events)
 async with httpx.AsyncClient(trust_env=False,timeout=10) as client:
  async def metrics(label):
   values={}
   for node,url in BACKENDS:
    response=await client.get(url+"/metrics");response.raise_for_status();text=response.text;(root/(label+"_"+node+".metrics")).write_text(text);values[node]={}
    for key in ["num_requests_running","num_requests_waiting"]:
     rows=re.findall(r"^vllm:"+key+r"(?:\{[^\n]*\})?\s+([0-9.eE+-]+)",text,re.M)
     if not rows:raise RuntimeError("native gauges unavailable")
     values[node][key]=sum(float(x) for x in rows)
   event("metrics",label=label,gauges=values);return values
  async def idle(label):
   end=time.monotonic()+60
   while True:
    values=await metrics(label)
    if all(v==0 for node in values.values() for v in node.values()):return
    if time.monotonic()>end:raise RuntimeError("native idle not verified")
    await asyncio.sleep(1)
  async def observe():
   index=0
   while not stop.is_set():
    try:await metrics("sample"+str(index))
    except Exception as e:event("observer_error",error=str(e))
    index+=1
    try:await asyncio.wait_for(stop.wait(),timeout=10)
    except asyncio.TimeoutError:pass
  main_task=asyncio.current_task()
  async def watch_owner():
   from datetime import datetime,timezone
   while not stop.is_set():
    try:
     current=json.loads(Path('/data/tiankuan/wio/glm52-pd/controller-owner.json').read_text())
     age=(datetime.now(timezone.utc)-datetime.fromisoformat(current['heartbeat_at'])).total_seconds()
     if current['run_id']!=run_id or current['status']!='running' or age>30:raise RuntimeError('owned controller lost/terminal/stale')
     for name in ['.controller.lock','.formal-test.lock']:
      with (Path('/data/tiankuan/wio/glm52-pd')/name).open('a+') as f:
       try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
       except BlockingIOError:pass
       else:raise RuntimeError('global controller lock lost')
    except Exception as error:
     event('ownership_lost',reason=str(error));main_task.cancel();return
    try:await asyncio.wait_for(stop.wait(),timeout=2)
    except asyncio.TimeoutError:pass
  try:
   watchdog=asyncio.create_task(watch_owner())
   await idle("before_reset")
   if cache_mode=="native_reset":
    for node,url in BACKENDS:
     response=await client.post(url+"/reset_prefix_cache")
     record={"endpoint":url+"/reset_prefix_cache","status":response.status_code,"body":response.text,"at":utc()}
     atomic_json(root/("reset_"+node+".json"),record)
     response.raise_for_status()
     if response.json().get("success") is not True:raise RuntimeError("local cache reset not successful; no full request replay")
   else:
    event("resident_cache_observed_without_reset",reason="Run18 current endpoint404; preserve resident engines, explicitly incomparable cold state")
   atomic_json(root/"cache_condition.json",{"mode":cache_mode,"reset_claim":cache_mode=="native_reset","external_cache_reset":False,"cache_state":"observed in native metrics, not assumed identical to historical cold/restart","actual_full_starts_after_both_native_domains_prefix_warmup":True})
   await idle("after_cache_condition");observer=asyncio.create_task(observe())
   trace=root/"router_trace.jsonl";raw=Path("/data/tiankuan/wio/Inference-Foundry/glm5-3/records/points/GLM-OPT-0002/runs/GLM-RUN-0125/router_wire");bench=root/"benchmark";bench.mkdir()
   live_trace=Path("/data/tiankuan/wio/Inference-Foundry/glm5-3/records/points/GLM-OPT-0002/runs/GLM-RUN-0125/router_trace.jsonl")
   atomic_json(root/"trace_cursor.json",dict(path=str(live_trace),bytes=live_trace.stat().st_size,sha256=__import__("hashlib").sha256(live_trace.read_bytes()).hexdigest()))
   env=dict(os.environ,GLM_REPLICAS=json.dumps([{"id":n,"url":u} for n,u in BACKENDS]),GLM_PLACEMENT_POLICY="work_seconds",GLM_ROUTER_TRACE_PATH=str(trace),GLM_ROUTER_AUDIT_DIR=str(raw),GLM_FORMAL_ROUTER_TRACE_PATH=str(trace),GLM_FORMAL_RECEIPTS_PATH=str(bench))
   for key in ["HTTP_PROXY","HTTPS_PROXY","ALL_PROXY","http_proxy","https_proxy","all_proxy"]:env.pop(key,None)
   response=await client.get("http://127.0.0.1:8000/healthcheck");response.raise_for_status()
   event("retained_public_service_used",owner=json.loads(Path("/data/tiankuan/wio/Inference-Foundry/glm5-3/records/points/GLM-OPT-0002/jobs/AUDIT-RUN158V2-20261003T0747Z/public_service_proof.json").read_text())["container"],gateway_operations=0)
   from retained_store import retained_responses
   for native_id,wire in retained_responses():
    expected=json.loads(Path(wire).read_text())
    res=await client.get("http://127.0.0.1:8000/v1/responses/"+native_id);res.raise_for_status();assert res.json()==expected;(root/(native_id+"_retained.wire")).write_bytes(res.content)
   event("oldSTORE_pointers_retained_before_bothdomain_prefix_warmup",new_inference=0)
   from owner_guard import guard
   guard();res=await client.delete("http://127.0.0.1:8000/control/replicas/D0");res.raise_for_status();drained=True
   snap=(await client.get("http://127.0.0.1:8000/control/replicas")).json();assert len(snap["replicas"])==1and snap["replicas"][0]["id"]=="D1"and not snap["replicas"][0]["active_requests"]
   event("controlled_D1_only",logical_D0_drain=True,model_operations=0,policy_changes=0)
   base=Path("/data/tiankuan/wio/Inference-Foundry/glm5-3")
   argv=["/opt/aisbench-venv/bin/python",str(Path(__file__).with_name("bench_entry.py")),"--input_len","81920","--output_len","64","--data_num","2","--concurrency","2","--request_rate","0","--repeat_rate","0.9","--prefix_num","1","--seed","20260930","--host_ip","127.0.0.1","--host_port","8000","--model_name","glm-52","--model_path","/data/tiankuan/wio/GLM-5.2-w8a8","--npu_num","16","--dp","1","--test_type","stream","--work_path","/opt/aisbench-benchmark","--pod_info","172.16.10.167:9900","--dataset_path",str(bench/"dataset"),"--output_dir",str(bench/"aisbench_output"),"--result_csv",str(bench/"result.csv"),"--result_jsonl",str(bench/"result.jsonl")]
   env["GLM_FORMAL_TRACE_CURSOR"]=str(root/"trace_cursor.json");env["PATH"]="/opt/aisbench-venv/bin:"+env.get("PATH","");env["PYTHONPATH"]=str(base/"runtime")+":"+str(base/"adapters/aisbench")+":/opt/aisbench-benchmark"
   atomic_json(root/"benchmark_command.json",{"argv":argv,"cwd":str(bench),"started_at":utc(),"dataset_seed":20260930,"request_seed":"omitted in unchanged generation_kwargs; native engine seed1024; dataset seed is distinct"})
   with (root/"benchmark.log").open("wb") as stream:
    process=await asyncio.create_subprocess_exec(*argv,cwd=str(bench),env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    code=await process.wait()
   event("benchmark_completed",inner_exit_code=code)
   if code!=0:raise RuntimeError("AISBench command/phase/native acceptance failed")
   for phase in ["warmup","full"]:
    receipt=json.loads((bench/(phase+"_wire_acceptance.json")).read_text());assert receipt["measurement_valid"]
   state=(await client.get("http://127.0.0.1:8000/control/replicas")).json();atomic_json(root/"final_placement.json",state)
   if any(x["active_requests"] for x in state["replicas"]):raise RuntimeError("native wire leases not released")
   await idle("final")
   atomic_json(root/"formal_summary.json",{"run_id":run_id,"valid":True,"kind":"prefix_tail_TTFT_diagnostic","completed_at":utc(),"inner_exit_code":code,"effective_output_tokens":128,"warmup_output_tokens":1,"verdict":"INCONCLUSIVE","limits":["Exactfirst2full152inputlines/firstprefix1/closedc2/controlledD1only/warm1/full2x64/current156epochs/native16active32resident/publicsame; D0logicaldrain-readd withnoD0inference, no capacityKEEP","Freshsharedcache_salt onbothwarm/full request, actualhits verified; bodyhashes differ152metadataandmax_tokens","D0private4096t1024serial11 unchanged/zeroinference; D1 allocated8192/private8192t4096serial1 unchanged; no privatepolicy changes","Native IDs-body-wire-usage-length-DONE/cost/clientSDK0 required; no operator/math/guards edits"]})
   accepted=True
  except BaseException as e:
   event("failure",error_type=type(e).__name__,message=str(e));atomic_json(root/"formal_summary.json",{"run_id":run_id,"valid":False,"verdict":"INVALID","error":str(e)});raise
  finally:
   # Inner phase runner uses its own groups: stop only pinned live CLI groups.
   inner=[]
   for phase in ['warmup','full']:
    p=root/'benchmark'/(phase+'_execution.json')
    if p.exists():
     rec=json.loads(p.read_text());child=rec.get('child')
     if child and rec.get('status')=='running' and same_process(child):inner.append(child)
   for child in inner:
    if same_process(child):
     try:os.killpg(child['pgid'],signal.SIGTERM)
     except ProcessLookupError:pass
   if process and process.returncode is None:
    os.killpg(process.pid,signal.SIGTERM)
    try:await asyncio.wait_for(process.wait(),10)
    except asyncio.TimeoutError:os.killpg(process.pid,signal.SIGKILL);await process.wait()
   if inner:
    await asyncio.sleep(2)
    for child in inner:
     if same_process(child):
      try:os.killpg(child['pgid'],signal.SIGKILL)
      except ProcessLookupError:pass
    event('pinned_inner_groups_stopped',children=inner)
   if drained:
    from owner_guard import guard
    guard();config=json.loads(Path(__file__).with_name("service_config.json").read_text());peers=config["environment"]["GLM_REPLICAS"];peers=json.loads(peers)if isinstance(peers,str)else peers;peer=next(x for x in peers if x["id"]=="D0")
    res=await client.post("http://127.0.0.1:8000/control/replicas",json=peer);res.raise_for_status()
    placement=(await client.get("http://127.0.0.1:8000/control/replicas")).json();assert len(placement["replicas"])==2and all(x["work_ranking_calibrated"]and not x["group_faulted"]for x in placement["replicas"])
    event("D0_full_peer_restored",placement=placement,no_model_operations=True)
   stop.set()
   if watchdog:await watchdog
   if observer:await observer
   if gateway:
    if gateway.poll() is None:
     os.killpg(gateway.pid,signal.SIGTERM)
     try:await asyncio.to_thread(gateway.wait,10)
     except subprocess.TimeoutExpired:os.killpg(gateway.pid,signal.SIGKILL);gateway.wait()
    event("gateway_stopped",exit_code=gateway.returncode)
   if log:log.close()
   if not accepted:
    try:await idle("cleanup")
    except Exception as error:event("cleanup_native_idle_unknown",error=str(error))
if __name__=="__main__":
 parser=argparse.ArgumentParser();parser.add_argument("root");parser.add_argument("run_id");parser.add_argument("--cache_mode",choices=["native_reset","resident_observed"],default="native_reset");args=parser.parse_args();asyncio.run(main(args.root,args.run_id,args.cache_mode))
