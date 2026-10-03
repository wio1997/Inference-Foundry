from pathlib import Path
import os,sys,json,time,subprocess,signal,urllib.request,re,hashlib
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import process_identity,same_process,atomic_json
j=Path(sys.argv[1]);http=urllib.request.build_opener(urllib.request.ProxyHandler({}))
def metrics(label):
 out={}
 for k,url in{"D0":"http://172.16.10.166:9081","D1":"http://172.16.10.167:9900"}.items():
  with http.open(url+"/metrics",timeout=10)as res:b=res.read()
  (j/(label+"_"+k+".metrics")).write_bytes(b);vals={}
  for line in b.decode().splitlines():
   if line.startswith("vllm:")and any(v in line for v in["generation_tokens_total","prompt_tokens_total","request_success_total","num_requests_running","num_requests_waiting"]):
    name=line.split("{")[0].split()[0];vals[name]=vals.get(name,0)+float(line.rsplit(" ",1)[1])
  assert vals["vllm:num_requests_running"]==vals["vllm:num_requests_waiting"]==0;out[k]=vals
 return out
before=metrics("initial")
env=dict(os.environ,PYTHONPATH="/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime"+":"+os.environ.get("PYTHONPATH",""))
argv=[sys.executable,"/data/tiankuan/wio/glm52-pd/deploy/plugins/dp_run89/native_acl_lifecycle.py","script","/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime/service_entry.py","--config",str(j/"service_config.json"),"--host","127.0.0.1","--port","8002"]
with(j/"gateway.log").open("wb")as log:
 child=subprocess.Popen(argv,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);identity=process_identity(child.pid);atomic_json(j/"child_identity.json",dict(argv=argv,identity=identity))
 try:
  for _ in range(450):
   assert child.poll()is None
   try:
    with http.open("http://127.0.0.1:8002/healthcheck",timeout=1)as res:
     if res.status==200:break
   except OSError:pass
   time.sleep(.1)
  else:raise RuntimeError("owned CPU frontend notready")
  assert same_process(identity)and[v.decode()for v in Path("/proc/"+str(child.pid)+"/cmdline").read_bytes().split(bytes([0]))if v]==argv
  child.send_signal(signal.SIGTERM);child.wait(20);assert child.returncode==0
 finally:
  if child.poll()is None:
   assert same_process(identity)and[v.decode()for v in Path("/proc/"+str(child.pid)+"/cmdline").read_bytes().split(bytes([0]))if v]==argv
   child.send_signal(signal.SIGTERM);child.wait(20)
assert not same_process(identity)
after=metrics("final");assert before==after
log=(j/"gateway.log").read_text();acks=[json.loads(v)for v in log.splitlines()if v.startswith('{"event":')]
assert [v["event"]for v in acks]==["task_acl_init","task_acl_finalize"]and all(v["returncode"]==0for v in acks)
assert "Application shutdown complete"in log and "GLM_SERVICE_ENTRY_INSTALLED "in log
fault=json.loads((j/"state/response_owners.json.fault").read_text());assert not fault["open"]and all(not v["faulted"]for v in fault["groups"].values())
atomic_json(j/"CPU_proof.json",dict(actual_entry_private8002=True,argv=argv,owned_child_identity=identity,actual_SIGTERM=True,exit_code=child.returncode,SDK_init_finalize=acks,closed_journal=fault,native_metrics_identical=True,native_inference=0,native_model_signals=0,limits=["Actualprivatefrontend/noinference lifecycle proof; publicfullnativeAPI later","SIGTERM sent only to freshly checked owned child; currentpublic125/models unchanged"]))
