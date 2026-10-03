from pathlib import Path
import sys,json,subprocess,shlex,hashlib,urllib.request,re,time,os,signal
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc,same_process
from owner_guard import start_watchdog,guard
start_watchdog();r=Path(__file__).parent;p=r.parents[1];g=p.parents[2];site=Path("/data/tiankuan/wio/glm52-pd/deploy");plug=site/"plugins/coupled_run129"
http=urllib.request.build_opener(urllib.request.ProxyHandler({}));plans=json.loads((r/"standalone_launch.json").read_text());prior=json.loads((r/"prior_native_members.json").read_text())
def run(node,args,label,input=None,timeout=150):
 guard()
 if node=="167":args=["ssh","-o","BatchMode=yes","root@172.16.10.167",shlex.join(args)]
 z=subprocess.run(args,input=input,capture_output=True,timeout=timeout);(r/(label+".stdout")).write_bytes(z.stdout);(r/(label+".stderr")).write_bytes(z.stderr);atomic_json(r/(label+".receipt.json"),dict(at=utc(),exit_code=z.returncode,stdout_bytes=len(z.stdout),stdout_sha256=hashlib.sha256(z.stdout).hexdigest()));z.check_returncode();return z.stdout
# Fresh exact host namespace identity, argv, config and native idle precede every signal.
public=json.loads((p/"jobs/AUDIT-RUN127-20261003T0005Z/public_service_proof.json").read_text());host=public["host_identity"];service=public["container_owner"]
assert same_process(host)and[v.decode()for v in Path("/proc/"+str(host["pid"])+"/cmdline").read_bytes().split(bytes([0]))if v]==service["argv"]
assert hashlib.sha256(Path(service["config"]["path"]).read_bytes()).hexdigest()==service["config"]["sha256"]
with http.open("http://127.0.0.1:8000/healthcheck",timeout=10)as response:h=json.loads(response.read())
assert h["request_num"]==0
atomic_json(r/"retire_public_before.json",dict(at=utc(),host=host,container=service,health=h))
guard();os.kill(host["pid"],signal.SIGTERM)
deadline=time.monotonic()+45
while same_process(host)and time.monotonic()<deadline:guard();time.sleep(.25)
assert not same_process(host)
log=Path(next(x["log"]["path"]for x in public["SDK_entry_logs"]if x["label"]=="restart"))
acks=[json.loads(x)for x in log.read_text().splitlines()if x.startswith('{"event":')]
assert acks[-1]["event"]=="task_acl_finalize"and acks[-1]["returncode"]==0
fault=json.loads((p/"runs/GLM-RUN-0125/response_owners.json.fault").read_text());assert not fault["open"]
assert not re.search(r":8000\s",subprocess.check_output(["ss","-ltnp"],text=True))
atomic_json(r/"retire_public.json",dict(at=utc(),exact_owned_public_retired=True,SDK_final0=True,models_signalled=0,owner_journal_preserved=True))
for key,v in prior.items():
 o=v["root"];fresh=json.loads(run(o["host"],["python3","-c",(r/"live_probe.py").read_text()],"cleanup_fresh_"+key,input=json.dumps(dict(owner=o,NPU_count=16)).encode()))
 old_ids={x["pid"]:x["identity"]for x in v["owned_targets"]}
 assert all(old_ids[x["pid"]]==x["identity"]for x in fresh["owned_targets"]if x["pid"]in fresh["npu_worker_pids"])
 with http.open("http://172.16.10."+o["host"]+":"+str(o["port"])+"/metrics",timeout=10)as response:b=response.read()
 (r/("cleanup_"+key+".metrics")).write_bytes(b)
 vals=re.findall(r"^vllm:num_requests_(?:running|waiting)(?:\{[^\n]*\})?\s+([0-9.eE+-]+)",b.decode(),re.M);assert vals and all(float(x)==0for x in vals)
 prior[key]=fresh
atomic_json(r/"cleanup_native_members.json",prior)
for pre in[True,False]:
 for key,v in prior.items():
  o=v["root"];run(o["host"],["python3","-c",(r/"stop_prior_cohort.py").read_text()],("ownership_"if pre else"cleanup_")+o["host"],input=json.dumps(dict(owner=o,members=v["owned_targets"],preflight=pre)).encode(),timeout=180)
for key,x in plans.items():
 node=x["host"];raw=run(node,["ss","-ltnp"],"before_sockets_"+node).decode()
 listening={int(a)for a in re.findall(r":([0-9]+)\s",raw)}
 assert not listening&({9081,9900,8000,29958}|set(range(63100,63164))|set(range(62500,62564)))
 # No PD ports are opened by this native standalone engine.
 shell="ulimit -c 0; export PD_LOCAL_IP=172.16.10."+node+" PD_NIC=business; source "+str(site/"scripts/pd_common_env.sh")+"; export PYTHONPATH="+str(plug)+":$PYTHONPATH VLLM_HOST_IP=172.16.10."+node+" VLLM_ENABLE_RESPONSES_API_STORE=1 HCCL_LOGIC_SUPERPOD_ID=0 "+" ".join(k+"="+shlex.quote(v)for k,v in x["environment"].items())+"; exec "+shlex.join(x["argv"])+" > "+shlex.quote(x["log"])+" 2>&1"
 run(node,["docker","exec","-d","glm52-single","bash","-c",shell],"start_"+node)
roots={};deadline=time.monotonic()+2400
while True:
 guard();time.sleep(10)
 for key,x in plans.items():
  template=roots.get(key)or dict(host=x["host"],argv=x["argv"],role="API"if x["api"]else"headless")
  snapshot=json.loads(run(x["host"],["python3","-c",(r/"live_probe.py").read_text()],"startup_members_"+key,input=json.dumps(dict(owner=template,NPU_count=None)).encode()))
  roots[key]=snapshot["root"]
  atomic_json(r/"startup_root_identities.json",roots)
  log=run(x["host"],["tail","-n","150",x["log"]],"startup_tail_"+key).decode(errors="replace")
  assert not any(v in log for v in["OutOfMemoryError","Engine core initialization failed","ValidationError:","RuntimeError:","ValueError:","NameError:","SyntaxError:","ModuleNotFoundError:","AssertionError:"]),"native startup failed on"+x["host"]
 try:
  with http.open("http://172.16.10.166:9081/health",timeout=3)as response:ready=response.status==200
 except Exception:ready=False
 atomic_json(r/"readiness.json",dict(at=utc(),ready=ready,roots=roots));print(json.dumps(dict(ready=ready,roots={k:v["pid"]for k,v in roots.items()})),flush=True)
 if ready:break
 assert time.monotonic()<deadline,"native coupled startup deadline"
members={}
for key,x in plans.items():
 v=json.loads(run(x["host"],["python3","-c",(r/"live_probe.py").read_text()],"ready_members_"+key,input=json.dumps(dict(owner=roots[key],NPU_count=16)).encode()))
 env=v["env"];assert env["GLM_ISSUE_BUDGET_COHORT"]=="GLM-COHORT-0129"and env["VLLM_PP_LAYER_PARTITION"]=="42,36"and env["VLLM_HOST_IP"]=="172.16.10."+x["host"]and env["HCCL_LOGIC_SUPERPOD_ID"]=="0"and env["VLLM_ENABLE_RESPONSES_API_STORE"]=="1"
 members[key]=v
 raw=run(x["host"],["cat",x["log"]],"native_resident_"+key)
 if x["api"]:
  assert b"Graph capturing finished"in raw
  install=[json.loads(l.split("GLM_ISSUE_BUDGET_INSTALLED ",1)[1])for l in raw.decode(errors="replace").splitlines()if"GLM_ISSUE_BUDGET_INSTALLED "in l]
  assert len(install)==1and install[0]["native_base"]=="AsyncScheduler"and install[0]["allocated_scheduler_max"]==4096
atomic_json(r/"standalone_root_identities.json",roots);atomic_json(r/"standalone_native_members.json",members)
from standalone_service_config import render
config=render(r,p/"runs/GLM-RUN-0125");atomic_json(r/"service_config.json",config)
# Compatibility metadata for the existing direct semantic/pilot driver; one API only.
api=roots["node0"];owner=dict(api,rank=0,native_dp_rank=0,native_dp_size=1,logical_alias="D1",port=9081)
atomic_json(r/"adopted_model_identities.json",dict(D1=owner))
atomic_json(r/"deployment_summary.json",dict(at=utc(),ready=True,geometry=config["geometry"],native_domain=config["native_domains"],roots=roots,NPU_count=32,API_count=1,PD_helpers=0,kv_connector=None,cohort="GLM-COHORT-0129",PP_partition=[42,36],MTP3=True,Graph32=True,SDK_model_init_active=True,STORE_replication=False,prior_models_retired=True,public_rebootstrap_pending=True))
print("coupled129 native engine ready",flush=True)
