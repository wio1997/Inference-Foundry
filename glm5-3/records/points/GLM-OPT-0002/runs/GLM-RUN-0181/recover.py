from pathlib import Path
import json,os,signal,time,subprocess,sys,urllib.request,re
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc,same_process
from owner_guard import start_watchdog,guard
import controls
start_watchdog();r=Path(__file__).parent;p=r.parents[1];old=p/"runs/GLM-RUN-0180"
state=json.loads((old/"state.json").read_text());assert state["status"]=="failed"and not same_process(state["owner"])
captured=json.loads((old/"readonly_client_process_capture.json").read_text());signals=[]
# This is task-client cleanup only. Every live group member must match actual pre-timeout capture.
for root in captured["descendants"]:
 if not same_process(dict(pid=root["pid"],**root["identity"])):continue
 proc=Path("/proc")/str(root["pid"])
 argv=[v.decode()for v in(proc/"cmdline").read_bytes().split(bytes([0]))if v]
 assert argv==root["argv"]
 if root["pgid"]!=root["pid"]:continue
 current=[]
 for f in Path("/proc").iterdir():
  if not f.name.isdigit():continue
  try:
   s=(f/"stat").read_text();v=s[s.rfind(")")+2:].split()
   if v[0]in["Z","X"]or int(v[2])!=root["pgid"]:continue
   pid=int(f.name);expected=next(x for x in captured["descendants"]if x["pid"]==pid)
   assert v[19]==expected["identity"]["start_ticks"]
   args=[v.decode()for v in(f/"cmdline").read_bytes().split(bytes([0]))if v];assert args==expected["argv"]
   assert not any("VLLMWorker"in x or "serve_local"in x for x in args)
   current.append(expected)
  except(FileNotFoundError,ProcessLookupError):pass
 assert current
 guard();os.killpg(root["pgid"],signal.SIGTERM);signals.append(dict(exact_owned_group=root["pgid"],targets=current,signal="SIGTERM"))
 end=time.monotonic()+15
 while any(same_process(dict(pid=x["pid"],**x["identity"]))for x in current)and time.monotonic()<end:guard();time.sleep(.1)
 remaining=[x for x in current if same_process(dict(pid=x["pid"],**x["identity"]))]
 if remaining:
  guard();os.killpg(root["pgid"],signal.SIGKILL);signals.append(dict(exact_owned_group=root["pgid"],targets=remaining,signal="SIGKILL"))
for x in captured["descendants"]:assert not same_process(dict(pid=x["pid"],**x["identity"]))
atomic_json(r/"client_cleanup.json",dict(at=utc(),signals=signals,only_known180_clients=True,native_model_signals=0,SDKfinalize_ACK_unknown=True))
http=urllib.request.build_opener(urllib.request.ProxyHandler({}))
deadline=time.monotonic()+120
while True:
 guard();values={}
 for key,url in[("D0","http://172.16.10.166:9081"),("D1","http://172.16.10.167:9900")]:
  raw=http.open(url+"/metrics",timeout=10).read();(r/("idle_"+key+".metrics")).write_bytes(raw)
  vals=re.findall(r"^vllm:num_requests_(?:running|waiting)(?:\{[^\n]*\})?\s+([0-9.eE+-]+)",raw.decode(),re.M);assert vals;values[key]=vals
 if all(float(x)==0for row in values.values()for x in row):break
 assert time.monotonic()<deadline,"nativeexisting180 worknotidle";time.sleep(1)
controls.verify("owned_idle_before_restore")
controls.update_D1(dict(controls.policies["167"],prefill_threshold_tokens=4096,serial=17),"D1_threshold4096_recovery")
public=json.loads((r/"public_service_proof.json").read_text());assert same_process(public["host"])
assert [v.decode()for v in(Path("/proc")/str(public["host"]["pid"])/"cmdline").read_bytes().split(bytes([0]))if v]==public["argv"]
def request(path,method="GET",body=None):
 guard();req=urllib.request.Request("http://127.0.0.1:8000"+path,data=None if body is None else json.dumps(body).encode(),headers={"content-type":"application/json"},method=method)
 return json.loads(http.open(req,timeout=15).read())
assert request("/healthcheck")["request_num"]==0
snap=request("/control/replicas");assert len(snap["replicas"])==2and all(not x["active_requests"]and not x["group_faulted"]for x in snap["replicas"])
peers=json.loads(json.loads((r/"service_config.json").read_text())["environment"]["GLM_REPLICAS"]);original=next(x for x in peers if x["id"]=="D0")
if not all(x["work_ranking_calibrated"]for x in snap["replicas"]):
 request("/control/replicas/D0","DELETE");request("/control/replicas","POST",original)
snap=request("/control/replicas");assert all(x["work_ranking_calibrated"]and not x["active_requests"]and not x["group_faulted"]for x in snap["replicas"])
for native_id,wire in __import__("retained_store").retained_responses():
 res=http.open("http://127.0.0.1:8000/v1/responses/"+native_id,timeout=15);raw=res.read();assert res.status==200and raw==Path(wire).read_bytes();(r/(native_id+"_retained.wire")).write_bytes(raw)
z=subprocess.run([sys.executable,str(r/"epoch_check.py"),"recovery_final"],capture_output=True,timeout=240);(r/"recovery_final.stdout").write_bytes(z.stdout);(r/"recovery_final.stderr").write_bytes(z.stderr);z.check_returncode()
controls.verify("owned_idle_restored")
out=dict(at=utc(),valid=True,kind="180deadline_recovery_no_inference",old_run180_status="INVALID",old_controller_dead=True,known180_clients_dead=True,client_signals=signals,native_model_signals=0,native32_same_idle=True,
 D0_policy_unchanged=controls.policies["166"],D1_policy_restored=controls.policies["167"],D0_full_peer_hints_restored=True,public179_same=True,placement=snap,bothSTORE_nativewire_same=True,new_inference_requests=0,model_operations=0,SDK180_final_ACK="unknown",
 limits=["1700s wrappertruncation notcapacity/modelfailure; preserve180raw/source/partialcommits independently","No unknown/foreign/modelprocess signals or requestreplay","SDK180 ACK absent; notinventfinalize0; currentpublic179 SDKinit0 remainsactive"])
atomic_json(r/"recovery_summary.json",out);m=json.loads((r/"manifest.json").read_text());m.update(status="completed",valid=True,verdict="INCONCLUSIVE",results=out);atomic_json(r/"manifest.json",m);print(json.dumps(out))
