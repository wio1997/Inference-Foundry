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
old=root.parent/"GLM-RUN-0045"
state=json.loads((old/"state.json").read_text());assert state["status"]=="failed" and not same_process(state["owner"])
audit=json.loads((old/"reduction.json").read_text());assert audit["diagnostic_evidence_valid"]and audit["verdict"]=="INVALID"and audit["inference_attempts"]==0 and audit["new_native_starts"]==0 and audit["old_Run44_cohort_cleanup_confirmed"]
cfg=root.parents[4]/"configs/coupled_tp32_ep32/start_node_run46.sh"
dest=site/"scripts/start_tp32_dcp1_ep32_run46.sh";plugdest=site/"scripts/glm_tool_contract_run46.py";probedest=site/"scripts/full_config_tp32_dcp1_run46.py"
for node in ["166","167"]:
 guard()
 top=cmd(node,["docker","top","glm52-single","-eo","pid,ppid,comm,args"]).decode()
 assert not any("/bin/vllm serve "in l or "VLLM::"in l or "bishengir-compile "in l for l in top.splitlines())
 (root/("before_"+node+".processes")).write_text(top)
 sockets=cmd(node,["ss","-ltnp"]).decode();(root/("before_"+node+".sockets")).write_text(sockets)
 assert not re.search(r":(?:32620|32621|32630|9081|9082|9900|9901|8000|8002)\s",sockets)
 assert cmd(node,["docker","exec","glm52-single","sha256sum","/vllm-workspace/vllm/vllm/distributed/device_communicators/shm_broadcast.py"]).decode().split()[0]=="a78bbfb8f750812049646a1224ccf1f12ca8e49f8765481cefb3ead93dff82a3"
 for source,target in [(cfg,dest),(root.parents[4]/"runtime/glm_tool_contract.py",plugdest),(root/"full_config_probe.py",probedest)]+[(root.parents[4]/"runtime"/f,site/"scripts"/f)for f in ["atomic_mq_bind.py","atomic_mq_worker.py"]]:
  if node=="166":target.write_bytes(source.read_bytes())
  else:subprocess.run(["scp","-q",str(source),"root@172.16.10.167:"+str(target)],check=True,capture_output=True)
  assert cmd(node,["sha256sum",str(target)]).decode().split()[0]==hashlib.sha256(source.read_bytes()).hexdigest()
 assert cmd(node,["sha256sum",str(site/"scripts/pd_common_env.sh")]).decode().split()[0]=="230e95e2ce618f2e4afb7adbc12afb5c23e588fbc6577a0b91246070edcca270"
 probe=cmd(node,["docker","exec","-e","PYTHONPATH="+str(site/"scripts"),"glm52-single","python3",str(probedest)],timeout=180);(root/("full_config_"+node+".stdout")).write_bytes(probe)
 c=json.loads(probe.decode().splitlines()[-1]);assert c["model_requests"]==c["weights_loaded"]==0 and len(c["full_native_engine_config"])==2
 assert all(row["dp"]==1 and row["tp"]==32 and row["dcp"]==1 and row["world"]==32 and row["local_world"]==16 and row["worker_cls"]=="atomic_mq_worker.AtomicMQWorker" and row["enable_reduce_sample"] is False for row in c["full_native_engine_config"])
 cmd(node,["docker","exec","glm52-single","bash","-c","export PD_LOCAL_IP=172.16.10."+node+"; export PD_NIC=business; source /data/tiankuan/wio/glm52-pd/deploy/scripts/pd_common_env.sh; export PYTHONPATH=\"/data/tiankuan/wio/glm52-pd/deploy/scripts${PYTHONPATH:+:$PYTHONPATH}\"; python3 -c 'from atomic_mq_worker import AtomicMQWorker; from vllm_ascend.worker.worker import NPUWorker; from atomic_mq_bind import install; assert issubclass(AtomicMQWorker,NPUWorker); install()'"],timeout=90)
event("both_fullCPU_configs_atomic_control_installed_prior_native_groups_absent",inference_attempts=0,model_signals=0,native_files_changed=False)
for rank,node in [(0,"166"),(1,"167")]:
 guard()
 cmd(node,["docker","exec","-d","-e","GLM_NODE_RANK="+str(rank),"-e","GLM_LOCAL_IP=172.16.10."+node,"-e","GLM_API_PORT=9081","-e","GLM_SUPERPOD_ID="+str(rank),"glm52-single","bash",str(dest)])
 event("physical_node_started",node=node,rank=rank,headless=bool(rank),DCP=1)
deadline=time.monotonic()+1500;startup_owners={}
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
  log=cmd(node,["tail","-n","120",str(site/("logs/TP_run46_"+str(rank)+".log"))]).decode(errors="replace")
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
