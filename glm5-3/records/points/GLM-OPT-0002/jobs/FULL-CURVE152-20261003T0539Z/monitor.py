from pathlib import Path
import json,sys,subprocess,time,re,urllib.request,hashlib,os,socket
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc,same_process,process_identity
def work(path):
 job=json.loads(Path(path).read_text());j=Path(path).parent;r=Path(job["inputs"][0]["path"]);http=urllib.request.build_opener(urllib.request.ProxyHandler({}));me=dict(host=socket.gethostname(),**process_identity(os.getpid()));n=0;last=None
 initial={}
 for key,node in [("D0","166"),("D1","167")]:
  text=(r/("prepare_"+node+".metrics")).read_text()
  initial[key]=sum(float(x)for x in re.findall(r"^vllm:generation_tokens_total(?:\{[^\n]*\})?\s+([0-9.eE+-]+)",text,re.M))+1
 # Eachnativeownerdomain has2 confirmedfull requests. Warmup1perowner isexcluded.
 while True:
  state=json.loads((r/"state.json").read_text());active=state["status"]=="running"and same_process(state["owner"]);now=time.monotonic();nodes={};errors=[]
  for key,url in [("D0","http://172.16.10.166:9081"),("D1","http://172.16.10.167:9900")]:
   try:
    with http.open(url+"/metrics",timeout=10)as res:raw=res.read()
    file=j/("sample_%04d_"%n+key+".metrics");file.write_bytes(raw);values={}
    for metric in ["num_requests_running","num_requests_waiting","generation_tokens_total","prompt_tokens_total","kv_cache_usage_perc","num_preemptions_total","prefix_cache_hits_total","prefix_cache_queries_total"]:
     vals=re.findall(r"^vllm:"+metric+r"(?:\{[^\n]*\})?\s+([0-9.eE+-]+)",raw.decode(),re.M)
     values[metric]=sum(float(x)for x in vals)if vals else None
    generation=values["generation_tokens_total"];assert generation is not None
    delta=generation-initial[key];assert delta>=0
    nodes[key]=dict(metrics=values,full_generation_so_far=delta,mean_context_tokens_estimate=81932+delta/2,raw=dict(path=str(file),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
   except Exception as e:errors.append(dict(owner=key,error_type=type(e).__name__,error=str(e)))
  row=dict(at=utc(),monotonic_s=now,controller=state,controller_alive=active,observer=me,nodes=nodes,errors=errors,initial_full_generation=initial,limits=["Nativegeneration counters in activefullphase are partial progress, not final functional/output credit","Meancontext estimate divides byactual2requests/domain; individualprogress/latency/acceptance not inferred","ReadonlyextraHTTPmetrics observer startedafterfullphasebegan, overhead included, no GPU work/resources operations"])
  if last is not None and not errors and not last["errors"]:
   elapsed=now-last["monotonic_s"];row["interval_s"]=elapsed
   row["native_committed_tps"]={key:(nodes[key]["full_generation_so_far"]-last["nodes"][key]["full_generation_so_far"])/elapsed for key in nodes}
  file=j/"monitor_snapshot.json";atomic_json(file,row)
  with(j/"samples.jsonl").open("a")as stream:stream.write(json.dumps(row)+"\n")
  raw=file.read_bytes();atomic_json(j/"result.json",dict(schema_version=1,job_id=j.name,status="running"if active else"completed",summary="Native152 partialprogress/context-dependentcommittedrates readonly; nofinalE2E/capacitycredit",execution=dict(inner_exit_code=None if active else 0,acceptance="unverified"if active else"passed",processes=[dict(host=me["host"],pid=me["pid"],boot_id=me["boot_id"],start_ticks=me["start_ticks"],readiness="ready",evidence_ids=["snapshot"])]if active else[]),findings=[],evidence=[dict(id="snapshot",path=str(file),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),locator="partialnativecounter/contextcurve")],unknowns=row["limits"],decision_request=None,next_check_at=None))
  if not active:return
  last=row;n+=1;time.sleep(10)
def launch(path):
 j=Path(path).parent
 with(j/"monitor.log").open("ab")as stream:child=subprocess.Popen([sys.executable,__file__,"work",path],stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
 for _ in range(300):
  if(j/"result.json").exists():print(json.dumps(dict(observer_pid=child.pid)));return
  if child.poll()is not None:raise RuntimeError("readonlynativecounter observer failed "+str(child.returncode))
  time.sleep(.1)
 raise RuntimeError("readonlyobserver no firstsnapshot")
if __name__=="__main__":{"launch":launch,"work":work}[sys.argv[1]](sys.argv[2])
