from pathlib import Path
import sys,json,subprocess,shlex,hashlib,urllib.request,urllib.error,re,time,os,signal
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc,same_process,process_identity
sys.path.insert(0,str(Path(__file__).parent/"runtime_bundle"))
from native_engines_service_config import render,checked_config
from owner_guard import start_watchdog,guard
start_watchdog();r=Path(__file__).parent;p=r.parents[1];g=p.parents[2];site=Path("/data/tiankuan/wio/glm52-pd/deploy");plug=site/"plugins/local_engines137";state=p/"runs/GLM-RUN-0125";bundle=r/"runtime_bundle"
http=urllib.request.build_opener(urllib.request.ProxyHandler({}))
def ref(f):
 b=Path(f).read_bytes();return dict(path=str(f),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def run(node,args,label,input=None,timeout=150):
 guard()
 if node=="167":args=["ssh","-o","BatchMode=yes","root@172.16.10.167",shlex.join(args)]
 z=subprocess.run(args,input=input,capture_output=True,timeout=timeout);(r/(label+".stdout")).write_bytes(z.stdout);(r/(label+".stderr")).write_bytes(z.stderr);atomic_json(r/(label+".receipt.json"),dict(at=utc(),exit_code=z.returncode,stdout_bytes=len(z.stdout),stdout_sha256=hashlib.sha256(z.stdout).hexdigest()));z.check_returncode();return z.stdout
def request(path,body=None,method=None):
 guard();req=urllib.request.Request("http://127.0.0.1:8000"+path,data=None if body is None else json.dumps(body).encode(),headers={"Content-Type":"application/json"},method=method)
 with http.open(req,timeout=15)as response:return json.loads(response.read())
def native_idle(node,port,label):
 with http.open("http://172.16.10."+node+":"+str(port)+"/metrics",timeout=10)as response:b=response.read()
 (r/(label+"_"+node+".metrics")).write_bytes(b)
 v=re.findall(r"^vllm:num_requests_(?:running|waiting)(?:\{[^\n]*\})?\s+([0-9.eE+-]+)",b.decode(),re.M);assert v and all(float(x)==0for x in v)
 return b
def fresh(o,expected,label):
 raw=run(o["host"],["python3","-c",(r/"live_probe.py").read_text()],label,input=json.dumps(dict(owner=o,NPU_count=16)).encode())
 v=json.loads(raw);ids={x["pid"]:x["identity"]for x in expected["owned_targets"]}
 assert v["npu_worker_pids"]==expected["npu_worker_pids"]and all(ids[x["pid"]]==x["identity"]for x in v["owned_targets"]if x["pid"]in v["npu_worker_pids"])
 return v
def retire(proof,log,label):
 identity=proof["host"];assert same_process(identity)and [v.decode()for v in Path("/proc/"+str(identity["pid"])+"/cmdline").read_bytes().split(bytes([0]))if v]==proof["argv"]
 conf=proof["container"]["config"]if "container"in proof else proof["config"]
 assert ref(conf["path"])==conf
 h=request("/healthcheck");assert h["status"]=="ok"and h["request_num"]==0
 snap=request("/control/replicas");assert all(not x["active_requests"]for x in snap["replicas"])
 guard();os.kill(identity["pid"],signal.SIGTERM);end=time.monotonic()+30
 while same_process(identity)and time.monotonic()<end:guard();time.sleep(.1)
 assert not same_process(identity)and not re.search(r":8000\s",subprocess.check_output(["ss","-ltnp"],text=True))
 acks=[json.loads(l)for l in Path(log).read_text().splitlines()if l.startswith('{"event":')]
 assert [x["event"]for x in acks]==["task_acl_init","task_acl_finalize"]and all(x["returncode"]==0for x in acks)
 f=json.loads((state/"response_owners.json.fault").read_text());assert not f["open"]
 atomic_json(r/(label+".json"),dict(at=utc(),exact_owned_host=identity,SDKinit_finalize0=True,fault_journal=f,state125_preserved=True,models_signalled=0))
def start_public(folder,label):
 config,_=checked_config(folder/"service_config.json")
 assert json.loads(config["environment"]["GLM_COMPATIBILITY_MEMBERS"])==dict(structured_output=["D1"],thinking_budget=["D1"],response_lifecycle=["D1"])
 assert not re.search(r":8000\s",subprocess.check_output(["ss","-ltnp"],text=True))
 argv=["/usr/local/python3.12.13/bin/python3",str(plug/"native_acl_lifecycle.py"),"script",str(bundle/"native_engines_service_entry_v14.py"),"--config",str(folder/"service_config.json"),"--host","0.0.0.0","--port","8000"];log=folder/"public.gateway.log"
 shell="ulimit -c 0; export PD_LOCAL_IP=172.16.10.166 PD_NIC=business; source "+str(site/"scripts/pd_common_env.sh")+"; export PYTHONPATH="+str(bundle)+":$PYTHONPATH; export GLM_TOKEN_MEMO_AUDIT_DIR="+str(r/"token_memo_raw")+" GLM_TOKEN_MEMO_TRACE_PATH="+str(r/"token_memo_trace.jsonl")+" GLM_TOKEN_MEMO_STATS_PATH="+str(r/"token_memo_stats.json")+"; exec "+shlex.join(argv)+" > "+shlex.quote(str(log))+" 2>&1"
 run("166",["docker","exec","-d","glm52-single","bash","-c",shell],label+"_start")
 deadline=time.monotonic()+90
 while True:
  guard();top=subprocess.check_output(["docker","top","glm52-single","-eo","pid,ppid,comm,args"],text=True);matches=[int(x.split()[0])for x in top.splitlines()[1:]if "native_engines_service_entry_v14.py --config "+str(folder/"service_config.json")in x];assert len(matches)==1
  pid=matches[0];ident=process_identity(pid);assert [v.decode()for v in Path("/proc/"+str(pid)+"/cmdline").read_bytes().split(bytes([0]))if v]==argv
  try:
   h=request("/healthcheck")
   if h["status"]=="ok"and h["request_num"]==0:break
  except (OSError,ValueError):pass
  assert time.monotonic()<deadline;time.sleep(1)
 snap=request("/control/replicas");assert len(snap["replicas"])==2and all(not x["group_faulted"]and not x["active_requests"]for x in snap["replicas"])
 text=log.read_text();acks=[json.loads(l)for l in text.splitlines()if l.startswith('{"event":')];assert acks[0]["event"]=="task_acl_init"and acks[0]["returncode"]==0
 marker=next(json.loads(l.split("GLM_SERVICE_ENTRY_INSTALLED ",1)[1])for l in text.splitlines()if "GLM_SERVICE_ENTRY_INSTALLED "in l);assert marker["native_domains"]==config["native_domains"]and marker["gateway_version"]=="V14"
 proof=dict(at=utc(),host=ident,argv=argv,config=ref(folder/"service_config.json"),container_marker=marker,placement=snap,SDK_init0=True,retained=True,
  container=dict(host="166",namespace="glm52-single_container",pid=marker["pid"],identity={key:marker[key]for key in["pid","boot_id","start_ticks"]},argv=argv,port=8000,config=ref(folder/"service_config.json"),native_domains=config["native_domains"]),marker=marker)
 atomic_json(folder/"public_service_proof.json",proof);atomic_json(folder/"public_host_owner.json",dict(identity=ident,argv=argv))
 # Observer stays with this exact public process; it can never control a replacement epoch.
 obsdir=folder/"identity_observer";obsdir.mkdir();args=["/usr/bin/python3",str(bundle/"native_identity_observer.py"),"--config",str(folder/"service_config.json"),"--public-owner",str(folder/"public_host_owner.json"),"--output-dir",str(obsdir),"--interval-s","5"]
 with (obsdir/"process.log").open("wb")as f:obs=subprocess.Popen(args,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 atomic_json(obsdir/"process_owner.json",dict(identity=process_identity(obs.pid),argv=args))
 for _ in range(60):
  guard();assert obs.poll()is None
  if (obsdir/"latest.json").exists():
   row=json.loads((obsdir/"latest.json").read_text())
   if all(x["status"]=="healthy"for x in row["groups"]):break
  time.sleep(.5)
 else:raise RuntimeError("HOST observer no healthy initial32 proof")
 return proof
def client(mode,timeout=600):
 shell="ulimit -c 0; export PD_LOCAL_IP=172.16.10.166 PD_NIC=business; source "+str(site/"scripts/pd_common_env.sh")+"; export PYTHONPATH="+str(bundle)+":"+str(r)+":$PYTHONPATH; python3 "+str(plug/"native_acl_lifecycle.py")+" script "+str(r/"client.py")+" "+shlex.quote(mode)
 raw=run("166",["docker","exec","glm52-single","bash","-c",shell],"client_"+mode,timeout=timeout)
 acks=[json.loads(l)for l in raw.decode().splitlines()if l.startswith('{"event":')];assert acks[0]["event"]=="task_acl_init"and acks[-1]["event"]=="task_acl_finalize"and acks[0]["returncode"]==acks[-1]["returncode"]==0
 atomic_json(r/("client_"+mode+"_SDK.json"),dict(at=utc(),SDKinit_finalize0=True,summary=ref(r/("client_"+mode+"_summary.json"))))
