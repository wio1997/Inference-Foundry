import json,subprocess,sys,time,urllib.request,re,shlex,hashlib,os,signal,threading,concurrent.futures
from pathlib import Path
from datetime import datetime,timezone
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc,same_process
root=Path(__file__).parent;task=Path("/data/tiankuan/wio/glm52-pd");site=task/"deploy";events=[];event_lock=threading.Lock()
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
owner=json.loads((task/"controller-owner.json").read_text())
def guard():
 c=json.loads((task/"controller-owner.json").read_text())
 age=(datetime.now(timezone.utc)-datetime.fromisoformat(c["heartbeat_at"])).total_seconds()
 assert c["run_id"]==root.name and c["status"]=="running" and c["owner"]==owner["owner"] and same_process(c["owner"]) and -2<=age<=30
def event(kind,**fields):
 with event_lock:
  events.append({"at":utc(),"event":kind,**fields});atomic_json(root/"deployment_events.json",events);print(json.dumps(events[-1]),flush=True)
def cmd(node,argv,timeout=120,input=None):
 if node=="167":argv=["ssh","-o","BatchMode=yes","root@172.16.10.167",shlex.join(argv)]
 p=subprocess.run(argv,input=input,capture_output=True,timeout=timeout);event("command",node=node,argv=argv,exit_code=p.returncode,stdout=p.stdout.decode(errors="replace")[-2500:],stderr=p.stderr.decode(errors="replace")[-1000:])
 p.check_returncode();return p.stdout

guard()
old=root.parent/"GLM-RUN-0034";latest=root.parent/"GLM-RUN-0035"
for priorrun in [old,latest]:
 state=json.loads((priorrun/"state.json").read_text());assert state["status"]=="completed" and not same_process(state["owner"])
 assert json.loads((priorrun/"reduction.json").read_text())["measurement_valid"]
prior=json.loads((old/"adopted_model_identities.json").read_text());assert prior==json.loads((latest/"adopted_model_identities.json").read_text())
atomic_json(root/"prior_model_owners.json",prior)
cfg=root.parents[4]/"configs/coupled_tp32_ep32/start_node_run36.sh"
plugin=root.parents[4]/"runtime/glm_tool_contract.py"
dest=site/"scripts/start_tp32_ep32_run36.sh";plugdest=site/"scripts/glm_tool_contract_run36.py";stopdest=site/"scripts/stop_tp32_run36_prior.py"
for rank,node,port in [(0,"166",9081),(1,"167",9900)]:
 guard()
 x=prior["DP"+str(rank)];pid=x["pid"]
 code="import pathlib,json;p=pathlib.Path('/proc/"+str(pid)+"');s=(p/'stat').read_text();print(json.dumps({'boot_id':pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'start_ticks':s[s.rfind(')')+2:].split()[19],'argv':[x.decode() for x in (p/'cmdline').read_bytes().split(bytes([0])) if x]}))"
 ident=json.loads(cmd(node,["python3","-c",code]));assert {k:ident[k]for k in ["boot_id","start_ticks"]}==x["identity"] and ident["argv"]==x["argv"]
 for endpoint in ["health","metrics"]:
  with opener.open("http://172.16.10."+node+":"+str(port)+"/"+endpoint,timeout=5)as response:
   assert response.status==200;data=response.read()
  (root/("prior_"+node+"."+endpoint)).write_bytes(data)
  if endpoint=="metrics":
   for k in ["num_requests_running","num_requests_waiting"]:
    vals=re.findall(r"^vllm:"+k+r"(?:\{[^\n]*\})?\s+([0-9.eE+-]+)",data.decode(),re.M);assert vals and all(float(v)==0 for v in vals)
 (root/("prior_DP"+str(rank)+".log")).write_bytes(cmd(node,["cat",str(site/("logs/DP_run34_"+str(rank)+".log"))]))
 for source,target in [(cfg,dest),(plugin,plugdest),(root/"stop_owned_model.py",stopdest)]:
  if node=="166":target.write_bytes(source.read_bytes())
  else:
   p=subprocess.run(["scp","-q",str(source),"root@172.16.10.167:"+str(target)],capture_output=True);p.check_returncode()
  assert cmd(node,["sha256sum",str(target)]).decode().split()[0]==hashlib.sha256(source.read_bytes()).hexdigest()
 assert cmd(node,["sha256sum",str(site/"scripts/pd_common_env.sh")]).decode().split()[0]=="230e95e2ce618f2e4afb7adbc12afb5c23e588fbc6577a0b91246070edcca270"
probe=cmd("166",["docker","exec","glm52-single","python3",str(root/"full_config_probe.py")],timeout=180);(root/"full_config_probe.stdout").write_bytes(probe)
config=json.loads(probe.decode().splitlines()[-1])
assert config["model_requests"]==config["weights_loaded"]==0 and len(config["full_native_engine_config"])==2
assert all(row["dp"]==1 and row["tp"]==row["dcp"]==32 and row["world"]==32 and row["local_world"]==16 and row["worker_cls"]=="vllm_ascend.worker.worker.NPUWorker"for row in config["full_native_engine_config"])
event("actual_native_DP1_TP32_DCP32_headless_config_only_accepted",native_requests=0)
guard()
def stop_rank(rank,node):
 return cmd(node,["env","GLM_EXPECTED_OWNER="+json.dumps(prior["DP"+str(rank)]),"python3",str(stopdest)],timeout=100)
with concurrent.futures.ThreadPoolExecutor(max_workers=2)as pool:
 futures=[pool.submit(stop_rank,rank,node)for rank,node in [(0,"166"),(1,"167")]]
 for future in futures:future.result()
guard();event("exact_healthy_Run34_physical_group_jointly_stopped",old_queue_replayed=False)
for node in ["166","167"]:
 rows=cmd(node,["docker","top","glm52-single","-eo","pid,ppid,comm,args"]).decode()
 assert not any("/bin/vllm serve "in l or "VLLM::"in l for l in rows.splitlines())
 (root/("before_"+node+".processes")).write_text(rows)
 ports=cmd(node,["ss","-ltnp"]).decode();assert not re.search(r":(?:32620|32621|32630|9081|9900|8000|8002)\s",ports)
for rank,node in [(0,"166"),(1,"167")]:
 guard()
 cmd(node,["docker","exec","-d","-e","GLM_NODE_RANK="+str(rank),"-e","GLM_LOCAL_IP=172.16.10."+node,"-e","GLM_API_PORT=9081","-e","GLM_SUPERPOD_ID="+str(rank),"glm52-single","bash",str(dest)])
 event("physical_node_started",node=node,rank=rank,headless=bool(rank))
deadline=time.monotonic()+2400;startup_owners={}
while True:
 guard()
 for rank,node in [(0,"166"),(1,"167")]:
  top=cmd(node,["docker","top","glm52-single","-eo","pid,ppid,comm,args"]).decode()
  rows=[l for l in top.splitlines()if "/bin/vllm serve "in l and "--port 9081"in l]
  assert len(rows)==1,"native physical entry exited or ambiguous; preserve logs"
  pid=int(rows[0].split()[0])
  code="import pathlib,json;p=pathlib.Path('/proc/"+str(pid)+"');s=(p/'stat').read_text();print(json.dumps({'boot_id':pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'start_ticks':s[s.rfind(')')+2:].split()[19],'argv':[x.decode() for x in (p/'cmdline').read_bytes().split(bytes([0])) if x]}))"
  ident=json.loads(cmd(node,["python3","-c",code]));key="Node"+str(rank)
  value={"host":node,"pid":pid,"identity":{k:ident[k]for k in ["boot_id","start_ticks"]},"argv":ident["argv"]}
  if key in startup_owners:assert startup_owners[key]==value,"native entry changed during startup"
  startup_owners[key]=value;atomic_json(root/"startup_model_identities.json",startup_owners)
  log=cmd(node,["tail","-n","120",str(site/("logs/TP_run36_"+str(rank)+".log"))]).decode(errors="replace")
  if any(t in log for t in ["Engine core initialization failed","OutOfMemoryError","vllm serve: error:","ValidationError:","ModuleNotFoundError:","NameError:","SyntaxError:","RuntimeError:"]):
   raise RuntimeError("native TP32 startup failure; preserve raw, no oldqueue replay")
 try:
  with opener.open("http://172.16.10.166:9081/health",timeout=3)as response:ready=response.status==200
 except Exception:ready=False
 event("single_API_shared_group_readiness",ready=ready,physical_owners=startup_owners)
 if ready:break
 if time.monotonic()>deadline:raise RuntimeError("native TP32 readiness deadline; preserve raw")
 time.sleep(15)
(root/"headless_167_ports.txt").write_bytes(cmd("167",["ss","-ltnp"]))
exec(compile((root/"adopt_models.py").read_text(),str(root/"adopt_models.py"),"exec"))
event("native_single_scheduler_coupled_group_ready")
