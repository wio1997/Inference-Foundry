import json,subprocess,sys,time,urllib.request,re,shlex,hashlib,os,threading,concurrent.futures
from pathlib import Path
from datetime import datetime,timezone
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc,same_process
root=Path(__file__).parent;task=Path("/data/tiankuan/wio/glm52-pd");site=task/"deploy";events=[];lock=threading.Lock()
ENTRIES=[(0, '166', 9081, '0,1,2,3,4,5,6,7', 0), (1, '166', 9082, '8,9,10,11,12,13,14,15', 0), (2, '167', 9900, '0,1,2,3,4,5,6,7', 1), (3, '167', 9901, '8,9,10,11,12,13,14,15', 1)]
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
owner=json.loads((task/"controller-owner.json").read_text())
def guard():
 c=json.loads((task/"controller-owner.json").read_text());age=(datetime.now(timezone.utc)-datetime.fromisoformat(c["heartbeat_at"])).total_seconds()
 assert c["run_id"]==root.name and c["status"]=="running" and c["owner"]==owner["owner"] and same_process(c["owner"]) and -2<=age<=30
def event(kind,**fields):
 with lock:
  events.append({"at":utc(),"event":kind,**fields});atomic_json(root/"deployment_events.json",events);print(json.dumps(events[-1]),flush=True)
def cmd(node,argv,timeout=120):
 if node=="167":argv=["ssh","-o","BatchMode=yes","root@172.16.10.167",shlex.join(argv)]
 p=subprocess.run(argv,capture_output=True,timeout=timeout);event("command",node=node,argv=argv,exit_code=p.returncode,stdout=p.stdout.decode(errors="replace")[-1600:],stderr=p.stderr.decode(errors="replace")[-800:])
 p.check_returncode();return p.stdout
def ident(node,pid):
 code="import pathlib,json;p=pathlib.Path('/proc/"+str(pid)+"');s=(p/'stat').read_text();env=(p/'environ').read_bytes().split(bytes([0]));print(json.dumps({'boot_id':pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'start_ticks':s[s.rfind(')')+2:].split()[19],'argv':[x.decode()for x in(p/'cmdline').read_bytes().split(bytes([0]))if x],'visible':[x.decode()for x in env if x.startswith(b'ASCEND_RT_VISIBLE_DEVICES=')]}))"
 return json.loads(cmd(node,["python3","-c",code]))
guard();old=root.parent/"GLM-RUN-0036";state=json.loads((old/"state.json").read_text())
assert state["status"]=="failed" and state["failure_phase"]=="deploy" and state["completed_stages"]==[] and not same_process(state["owner"])
assert not list(old.glob("**/load_result.json")) and not list(old.glob("**/*.wire")) and not list(old.glob("**/*.events.jsonl"))
for pin in json.loads((old/"controller_spec.json").read_text())["stages"][0]["sources"]:assert hashlib.sha256(Path(pin["path"]).read_bytes()).hexdigest()==pin["sha256"]
prior=json.loads((old/"startup_model_identities.json").read_text());atomic_json(root/"prior_model_owners.json",prior)
cfg=root.parents[4]/"configs/coupled_dp4_ep32/start_rank_run38.sh";dest=site/"scripts/start_dp4_ep32_run38.sh";plugdest=site/"scripts/glm_tool_contract_run38.py";stopdest=site/"scripts/stop_dp4_run38_prior.py"
plugdir=site/"plugins/coupled_dp_run38"
for node in ["166","167"]:
 guard()
 ownerkey="Node0"if node=="166"else"Node1";x=prior[ownerkey];actual=ident(node,x["pid"])
 assert {k:actual[k]for k in ["boot_id","start_ticks"]}==x["identity"] and actual["argv"]==x["argv"]
 cmd(node,["mkdir","-p",str(plugdir)])
 for source,target in [(cfg,dest),(root.parents[4]/"runtime/glm_tool_contract.py",plugdest),(root/"stop_owned_model.py",stopdest),(root/"full_config_probe.py",site/"scripts/full_config_dp4_run38.py")]+[(root.parents[4]/("runtime/"+f),plugdir/f)for f in ["coupled_dp_worker_v2.py","coupled_dp_metadata_v2.py"]]:
  if node=="166":target.write_bytes(source.read_bytes())
  else:subprocess.run(["scp","-q",str(source),"root@172.16.10.167:"+str(target)],check=True)
  assert cmd(node,["sha256sum",str(target)]).decode().split()[0]==hashlib.sha256(source.read_bytes()).hexdigest()
 assert cmd(node,["sha256sum",str(site/"scripts/pd_common_env.sh")]).decode().split()[0]=="230e95e2ce618f2e4afb7adbc12afb5c23e588fbc6577a0b91246070edcca270"
 log=site/("logs/TP_run36_"+("0"if node=="166"else"1")+".log")
 (root/("prior_"+node+".log")).write_bytes(cmd(node,["cat",str(log)]))
 probe=cmd(node,["docker","exec","-e","PYTHONPATH="+str(plugdir),"glm52-single","python3",str(site/"scripts/full_config_dp4_run38.py")],timeout=180)
 (root/("full_config_"+node+".stdout")).write_bytes(probe)
 config=json.loads(probe.decode().splitlines()[-1]);assert config["accepted_config_only"] and config["model_requests"]==config["weights_loaded"]==0 and len(config["configurations"])==4
 # Both physical nodes preflight attribution before any model signal.
 proof=cmd(node,["env","GLM_EXPECTED_OWNER="+json.dumps(x),"GLM_CLEANUP_PREFLIGHT_ONLY=1","python3",str(stopdest)],timeout=30)
 (root/("cleanup_preflight_"+node+".json")).write_bytes(proof)
event("DP4_native_config_and_all_old_owners_accepted_before_signals",inference_attempts=0)
guard()
def stop(node):
 return cmd(node,["env","GLM_EXPECTED_OWNER="+json.dumps(prior["Node0"if node=="166"else"Node1"]),"python3",str(stopdest)],timeout=100)
with concurrent.futures.ThreadPoolExecutor(max_workers=2)as pool:
 for f in [pool.submit(stop,node)for node in ["166","167"]]:f.result()
event("exact_failed_Run36_group_jointly_removed",old_queue_replayed=False);guard()
for node in ["166","167"]:
 top=cmd(node,["docker","top","glm52-single","-eo","pid,ppid,comm,args"]).decode()
 assert not any("/bin/vllm serve "in l or "VLLM::"in l or "bishengir-compile "in l for l in top.splitlines())
 (root/("before_"+node+".processes")).write_text(top)
 ports=cmd(node,["ss","-ltnp"]).decode();assert not re.search(r":(?:32620|32621|32630|9081|9082|9900|9901|8000|8002)\s",ports)
for rank,node,port,visible,pod in ENTRIES:
 guard()
 cmd(node,["docker","exec","-d","-e","GLM_DP_RANK="+str(rank),"-e","GLM_LOCAL_IP=172.16.10."+node,"-e","GLM_API_PORT="+str(port),"-e","GLM_SUPERPOD_ID="+str(pod),"-e","GLM_VISIBLE_DEVICES="+visible,"glm52-single","bash",str(dest)])
 event("native_DP_engine_started",rank=rank,node=node,port=port,visible=visible,physical_superpod_id=pod)
deadline=time.monotonic()+1500;startup={}
while True:
 guard();ready=[]
 for rank,node,port,visible,pod in ENTRIES:
  rows=[l for l in cmd(node,["docker","top","glm52-single","-eo","pid,ppid,comm,args"]).decode().splitlines()if "/bin/vllm serve "in l and "--port "+str(port)+" "in l]
  assert len(rows)==1,"native DP entry exited or ambiguous; preserve logs"
  pid=int(rows[0].split()[0]);x=ident(node,pid);key="DP"+str(rank)
  assert x["visible"]==["ASCEND_RT_VISIBLE_DEVICES="+visible] and x["argv"][x["argv"].index("--data-parallel-rank")+1]==str(rank)
  val={"host":node,"pid":pid,"identity":{k:x[k]for k in ["boot_id","start_ticks"]},"argv":x["argv"]}
  if key in startup:assert startup[key]==val
  startup[key]=val;atomic_json(root/"startup_model_identities.json",startup)
  log=cmd(node,["tail","-n","100",str(site/("logs/DP_run38_"+str(rank)+".log"))]).decode(errors="replace")
  if any(t in log for t in ["Engine core initialization failed","OutOfMemoryError","vllm serve: error:","ValidationError:","ModuleNotFoundError:","NameError:","SyntaxError:","RuntimeError:"]):raise RuntimeError("native DP4 startup failure; preserve raw")
  try:
   with opener.open("http://172.16.10."+node+":"+str(port)+"/health",timeout=3)as response:ready.append(response.status==200)
  except Exception:ready.append(False)
 event("four_API_coupled_readiness",ready=ready,owners=startup)
 if all(ready):break
 if time.monotonic()>deadline:raise RuntimeError("native DP4 readiness deadline")
 time.sleep(15)
exec(compile((root/"adopt_models.py").read_text(),str(root/"adopt_models.py"),"exec"))
event("all_four_native_engines_ready",group=json.loads((root/"execution_groups.json").read_text()))
