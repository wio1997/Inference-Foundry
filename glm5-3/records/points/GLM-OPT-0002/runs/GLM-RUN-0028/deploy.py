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
old=root.parent/"GLM-RUN-0027";state=json.loads((old/"state.json").read_text())
assert state["status"]=="completed" and not same_process(state["owner"]) and json.loads((old/"reduction.json").read_text())["measurement_valid"]
jobs=root.parents[1]/"jobs"
for name in ["DP-METADATA-CONTRACTS-20261001T1900Z-v5","DP-METADATA-FULLCONFIG-20261001T1910Z"]:
 j=jobs/name
 p=subprocess.run([sys.executable,str(root.parents[4]/"scripts/zcode_bridge.py"),"validate",str(j/"job.json"),str(j/"result.json")],capture_output=True);p.check_returncode()
cpu=json.loads((jobs/"DP-METADATA-CONTRACTS-20261001T1900Z-v5/reduction.json").read_text());assert cpu["valid"] and cpu["case_count"]==12 and cpu["source_guard"]
prior=json.loads((old/"adopted_model_identities.json").read_text());assert prior==json.loads((root.parent/"GLM-RUN-0025/adopted_model_identities.json").read_text())
atomic_json(root/"prior_model_owners.json",prior)
# New versioned plugin directory, untouched native vendor/operator sources.
plugdir=site/"plugins/coupled_dp_run28"
for node in ["166","167"]:
 cmd(node,["mkdir","-p",str(plugdir)])
 for name in ["coupled_dp_metadata.py","coupled_dp_worker.py"]:
  source=root.parents[4]/"runtime"/name;target=plugdir/name
  if node=="166":target.write_bytes(source.read_bytes())
  else:
   p=subprocess.run(["scp","-q",str(source),"root@172.16.10.167:"+str(target)],capture_output=True);p.check_returncode()
  assert cmd(node,["sha256sum",str(target)]).decode().split()[0]==hashlib.sha256(source.read_bytes()).hexdigest()
 code="import sys,pathlib,json;sys.path.insert(0,"+repr(str(plugdir))+");from coupled_dp_metadata import make_sources;s=pathlib.Path('/vllm-workspace/vllm-ascend/vllm_ascend/worker/model_runner_v1.py').read_text();rows=make_sources(s);print(json.dumps({k:v['native_sha256'] for k,v in rows.items()}))"
 actual=json.loads(cmd(node,["docker","exec","glm52-single","python3","-c",code]))
 assert actual=={k:v["native_sha256"] for k,v in cpu["methods"].items()}
 # Before task-native stop, verify both actual argv/boot/start and idle.
 x=prior["DP"+str(["166","167"].index(node))];pid=x["pid"]
 code="import pathlib,json;p=pathlib.Path('/proc/"+str(pid)+"');s=(p/'stat').read_text();print(json.dumps({'boot_id':pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'start_ticks':s[s.rfind(')')+2:].split()[19],'argv':[x.decode() for x in (p/'cmdline').read_bytes().split(bytes([0])) if x]}))"
 ident=json.loads(cmd(node,["python3","-c",code]));assert {k:ident[k] for k in ["boot_id","start_ticks"]}==x["identity"] and ident["argv"]==x["argv"]
 port=9081 if node=="166" else 9900
 with opener.open("http://172.16.10."+node+":"+str(port)+"/health",timeout=5) as response:assert response.status==200
 with opener.open("http://172.16.10."+node+":"+str(port)+"/metrics",timeout=5) as response:text=response.read().decode()
 (root/("prior_"+node+".metrics")).write_text(text)
 for key in ["num_requests_running","num_requests_waiting"]:
  vals=re.findall(r"^vllm:"+key+r"(?:\{[^\n]*\})?\s+([0-9.eE+-]+)",text,re.M);assert vals and all(float(v)==0 for v in vals)
 log=site/("logs/DP_run25_"+str(["166","167"].index(node))+".log")
 (root/("prior_DP"+str(["166","167"].index(node))+".log")).write_bytes(cmd(node,["cat",str(log)]))
 helper=root/"stop_owned_model.py";target=site/"scripts/stop_coupled_run28.py"
 if node=="166":target.write_bytes(helper.read_bytes())
 else:
  p=subprocess.run(["scp","-q",str(helper),"root@172.16.10.167:"+str(target)],capture_output=True);p.check_returncode()
 assert cmd(node,["sha256sum",str(target)]).decode().split()[0]==hashlib.sha256(helper.read_bytes()).hexdigest()
guard();event("exact_joint_prior_ready_idle_sources_guarded_before_stop")
def stop_rank(rank,node):
 x=prior["DP"+str(rank)]
 return cmd(node,["env","GLM_EXPECTED_OWNER="+json.dumps(x),"python3",str(site/"scripts/stop_coupled_run28.py")],timeout=100)
# Both physical members stopped concurrently by exact identity; no PD proxy or
# other task process is targeted, no old requests replayed.
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
 futures=[pool.submit(stop_rank,rank,node) for rank,node in [(0,"166"),(1,"167")]]
 for f in futures:f.result()
guard();event("exact_joint_prior_stopped_native_worker_class_requires_rebuild")
atomic_json(root/"reused_terminal_run.json",{"run":old.name,"manifest":str(old/"manifest.json"),"sha256":hashlib.sha256((old/"manifest.json").read_bytes()).hexdigest(),"old_queue":"not replayed","new_work":"same bounded package to validate changed native control metadata; fresh cache observed"})
for node in ["166","167"]:
 rows=cmd(node,["docker","top","glm52-single","-eo","pid,ppid,comm,args"]).decode()
 assert not any("/bin/vllm serve " in l or "VLLM::" in l for l in rows.splitlines()),"task model processes remain; reconcile, no startup"
 (root/("before_"+node+".processes")).write_text(rows)
 ports=cmd(node,["ss","-ltnp"]).decode()
 assert not re.search(r":(?:32620|32621|32630|9081|9900|8000)\s",ports),"candidate ports remain occupied after exact cohort cleanup; reconcile"
event("both_task_containers_empty_proxy_still_stopped_no_requests")
cfg=root.parents[4]/"configs/coupled_dp2_ep32/start_rank_run28.sh"
assert cfg.exists(),str(cfg)
plugin=root.parents[4]/"runtime/glm_tool_contract.py"
dest=site/"scripts/start_dp2_ep32_run28.sh";plugdest=site/"scripts/glm_tool_contract_run28.py"
for node in ["166","167"]:
 for source,target in [(cfg,dest),(plugin,plugdest)]:
  if node=="166":target.write_bytes(source.read_bytes())
  else:
   p=subprocess.run(["scp","-q",str(source),"root@172.16.10.167:"+str(target)],capture_output=True);p.check_returncode()
  digest=cmd(node,["sha256sum",str(target)]).decode().split()[0];assert digest==hashlib.sha256(source.read_bytes()).hexdigest()
 # Same native common env used on both ranks; no operator source mutation.
 digest=cmd(node,["sha256sum",str(site/"scripts/pd_common_env.sh")]).decode().split()[0]
 assert digest=="230e95e2ce618f2e4afb7adbc12afb5c23e588fbc6577a0b91246070edcca270"
# Both cohort entries start before any readiness wait, so shared groups can initialize together.
for rank,node,port in [(0,"166",9081),(1,"167",9900)]:
 guard()
 cmd(node,["docker","exec","-d","-e","GLM_DP_RANK="+str(rank),"-e","GLM_LOCAL_IP=172.16.10."+node,"-e","GLM_API_PORT="+str(port),"-e","GLM_SUPERPOD_ID="+str(rank),"glm52-single","bash",str(dest)])
 event("coupled_rank_started",rank=rank,node=node,port=port)
startup_owners={}
deadline=time.monotonic()+2400
while True:
 guard();ready={}
 for rank,node,port in [(0,"166",9081),(1,"167",9900)]:
  try:
   with opener.open("http://172.16.10."+node+":"+str(port)+"/health",timeout=3) as response:ready[node]=response.status==200
  except Exception:ready[node]=False
  top=cmd(node,["docker","top","glm52-single","-eo","pid,ppid,comm,args"]).decode()
  rows=[l for l in top.splitlines() if "/bin/vllm serve " in l and "--port "+str(port) in l]
  if rows:
   assert len(rows)==1
   pid=int(rows[0].split()[0])
   code="import pathlib,json;p=pathlib.Path('/proc/"+str(pid)+"');s=(p/'stat').read_text();print(json.dumps({'boot_id':pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'start_ticks':s[s.rfind(')')+2:].split()[19],'argv':[x.decode() for x in (p/'cmdline').read_bytes().split(bytes([0])) if x]}))"
   identity=json.loads(cmd(node,["python3","-c",code]))
   startup_owners["DP"+str(rank)]={"host":node,"pid":pid,"identity":{k:identity[k] for k in ["boot_id","start_ticks"]},"argv":identity["argv"],"observed_at":utc()}
   atomic_json(root/"startup_model_identities.json",startup_owners)
  tail=cmd(node,["tail","-n","100",str(site/("logs/DP_run28_"+str(rank)+".log"))],timeout=15).decode(errors="replace")
  if any(text in tail for text in ["Engine core initialization failed","Engine core initialization failed.","OutOfMemoryError","vllm serve: error:","ValidationError:","native control source changed:","native metadata substitution mismatch:","GLM coupled metadata prototype requires","opt-in requires","ModuleNotFoundError:","NameError:","SyntaxError:"]):raise RuntimeError("native DP startup failed; preserve raw, no replay")
  if not rows and not ready[node]:raise RuntimeError("native task API entry exited; preserve raw, no replay")
 event("cohort_readiness",ready=ready)
 if all(ready.values()):break
 if time.monotonic()>deadline:raise RuntimeError("cohort readiness timed out; preserve raw, no replay")
 time.sleep(15)
for rank,node in [(0,"166"),(1,"167")]:
 log=cmd(node,["cat",str(site/("logs/DP_run28_"+str(rank)+".log"))]).decode(errors="replace")
 installed=[l for l in log.splitlines() if "GLM_DP_METADATA_INSTALLED " in l]
 assert len(installed)==16,(rank,len(installed))
 (root/("installed_DP"+str(rank)+".json")).write_text(json.dumps(installed,indent=2))
exec(compile((root/"adopt_models.py").read_text(),str(root/"adopt_models.py"),"exec"))
event("coupled_cohort_ready")
