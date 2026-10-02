from pathlib import Path
import os,sys,time,json,subprocess,shlex,hashlib,socket
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,process_identity,same_process,utc
def work(path):
 job=json.loads(Path(path).read_text());j=Path(path).parent;r=Path(job["inputs"][0]["path"]);me={"host":socket.gethostname(),**process_identity(os.getpid())}
 while True:
  state=json.loads((r/"state.json").read_text());active=state["status"]=="running"and same_process(state["owner"]);tails=[]
  for node in ["166","167"]:
   files=["/data/tiankuan/wio/glm52-pd/deploy/logs/"+role+("_run69_"if role=="P"else"_run69_")+str(0 if node=="166"else 1)+".log"for role in ["D"]]
   args=["tail","-n","5"]+files
   if node=="167":args=["ssh","-o","BatchMode=yes","-o","ConnectTimeout=5","root@172.16.10.167",shlex.join(args)]
   t=subprocess.run(args,capture_output=True,text=True,timeout=15);tails.append({"node":node,"returncode":t.returncode,"tail":t.stdout,"stderr":t.stderr[-1000:]})
  out={"at":utc(),"controller":state,"controller_alive":active,"monitor":me,"tails":tails,"native_owners":json.loads((r/"adopted_model_identities.json").read_text())if(r/"adopted_model_identities.json").exists()else None}
  atomic_json(j/"monitor_snapshot.json",out)
  with(j/"monitor_samples.jsonl").open("a")as f:f.write(json.dumps(out)+"\n")
  status="running"if active else"completed"if state["status"]=="completed"else"needs_decision"
  f=j/"monitor_snapshot.json";b=f.read_bytes();ref={"id":"snapshot","path":str(f),"bytes":len(b),"sha256":hashlib.sha256(b).hexdigest(),"locator":"actualcontroller/owners/bothrole-logtails"}
  atomic_json(j/"result.json",{"schema_version":1,"job_id":j.name,"status":status,"summary":"Run73 "+state["status"]+"/"+str(state["active_stage"])+" readonlymonitor; nofit/capacityclaim","execution":{"inner_exit_code":None if active else 0,"acceptance":"unverified"if active else"passed","processes":[dict(host=me["host"],pid=me["pid"],boot_id=me["boot_id"],start_ticks=me["start_ticks"],readiness="ready",evidence_ids=["snapshot"])]if active else[]},"findings":[],"evidence":[ref],"unknowns":["NativeP/Dfit/transfer/fullinput/finitestability notcertifiedbylogs/health"],"decision_request":None if active or status=="completed"else"Reconcileexactnativephase/rawwithoutreplay","next_check_at":None})
  if not active:return
  time.sleep(45)
def launch(path):
 j=Path(path).parent
 with(j/"monitor.log").open("ab")as f:p=subprocess.Popen([sys.executable,__file__,"work",path],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 for _ in range(300):
  if(j/"result.json").exists():print(json.dumps({"monitor_pid":p.pid,"result":str(j/"result.json")}));return 0
  if p.poll()is not None:return p.returncode
  time.sleep(.05)
 return 85
if __name__=="__main__":sys.exit({"launch":launch,"work":work}[sys.argv[1]](sys.argv[2]))
