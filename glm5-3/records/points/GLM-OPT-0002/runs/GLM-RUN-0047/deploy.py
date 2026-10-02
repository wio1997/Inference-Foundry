import json,subprocess,sys,time,urllib.request,re,shlex,hashlib,concurrent.futures
from pathlib import Path
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc,same_process
from owner_guard import start_watchdog,guard
start_watchdog()
r=Path(__file__).parent;g=r.parents[4];site=Path("/data/tiankuan/wio/glm52-pd/deploy");plug=site/"plugins/pd_dual_run47";events=[]
http=urllib.request.build_opener(urllib.request.ProxyHandler({}))
def event(kind,**v):
 events.append(dict(at=utc(),event=kind,**v));atomic_json(r/"deployment_events.json",events);print(json.dumps(events[-1]),flush=True)
def cmd(node,args,timeout=60):
 if node=="167":args=["ssh","-o","BatchMode=yes","root@172.16.10.167",shlex.join(args)]
 p=subprocess.run(args,capture_output=True,timeout=timeout);event("command",node=node,argv=args,returncode=p.returncode,stdout_tail=p.stdout.decode(errors="replace")[-1800:],stderr_tail=p.stderr.decode(errors="replace")[-1000:]);p.check_returncode();return p.stdout
def copy(node,src,dst):
 if node=="166":dst.write_bytes(src.read_bytes())
 else:subprocess.run(["scp","-q",str(src),"root@172.16.10.167:"+str(dst)],check=True,capture_output=True)
 assert cmd(node,["sha256sum",str(dst)]).decode().split()[0]==hashlib.sha256(src.read_bytes()).hexdigest()
old=r.parent/"GLM-RUN-0046";s=json.loads((old/"state.json").read_text());assert s["status"]=="completed"and not same_process(s["owner"])
a=json.loads((old/"reduction_brief.json").read_text());assert a["measurement_valid"]and a["functional_acceptance"]and a["new_effective_output_tokens"]==320
job=r.parents[1]/"jobs/PD2-ACL-LIFETIME-CONFIG-20261002T0223Z";a=json.loads((job/"reduction.json").read_text());assert a["normal_full_config_exit"]and len(a["rows"])==4 and all(x["returncode"]==0 for x in a["rows"])
assert a["wrapper"]["sha256"]==hashlib.sha256((g/"runtime/native_acl_lifecycle.py").read_bytes()).hexdigest()
prior=json.loads((old/"adopted_model_identities.json").read_text());atomic_json(r/"prior_model_owners.json",prior)
# One controller verifies all old physical members/idle/source before any task signal.
for node in ["166","167"]:
 guard();cmd(node,["mkdir","-p",str(plug)])
 for name in ["native_acl_lifecycle.py","atomic_mq_bind.py","atomic_mq_worker.py","coupled_dp_metadata_v2.py","coupled_atomic_mq_worker.py","glm_tool_contract.py"]:
  copy(node,g/"runtime"/name,plug/name)
 copy(node,r/"stop_prior_cohort.py",plug/"stop_prior_cohort.py")
 assert cmd(node,["sha256sum",str(site/"scripts/pd_common_env.sh")]).decode().split()[0]=="230e95e2ce618f2e4afb7adbc12afb5c23e588fbc6577a0b91246070edcca270"
 index=json.loads((r.parents[1]/"jobs/PD2-CPU-DIAGNOSTIC-20261002T0158Z/source_index.json").read_text())
 for entry in index:
  assert cmd(node,["docker","exec","glm52-single","sha256sum",entry["native_path"]]).decode().split()[0]==entry["sha256"],"installednative source changed"
 code="import sys,pathlib;sys.path.insert(0,"+repr(str(plug))+");from coupled_dp_metadata_v2 import make_sources;make_sources(pathlib.Path('/vllm-workspace/vllm-ascend/vllm_ascend/worker/model_runner_v1.py').read_text());print('native_metadata_source_guard_passed')"
 cmd(node,["docker","exec","glm52-single","python3","-c",code])
 assert cmd(node,["docker","exec","glm52-single","sha256sum","/vllm-workspace/vllm/vllm/distributed/device_communicators/shm_broadcast.py"]).decode().split()[0]=="a78bbfb8f750812049646a1224ccf1f12ca8e49f8765481cefb3ead93dff82a3"
 owners=[x for x in prior.values()if x["host"]==node]
 raw=cmd(node,["env","GLM_EXPECTED_COHORT="+json.dumps(owners),"GLM_CLEANUP_PREFLIGHT_ONLY=1","python3",str(plug/"stop_prior_cohort.py")])
 (r/("prior_"+node+".owner_preflight.json")).write_bytes(raw)
 shell="ulimit -c 0; export PD_LOCAL_IP=172.16.10."+node+" PD_NIC=business; source "+str(site/"scripts/pd_common_env.sh")+"; export PYTHONPATH="+str(plug)+":$PYTHONPATH; python3 "+str(plug/"native_acl_lifecycle.py")+" cli serve --help"
 helpout=cmd(node,["docker","exec","glm52-single","bash","-c",shell],100);(r/("native_CLI_help_"+node+".stdout")).write_bytes(helpout)
 ack=[json.loads(l)for l in helpout.decode().splitlines()if l.startswith('{"event": "task_acl_')]
 assert len(ack)==2 and all(x["returncode"]==0 for x in ack)
 for key,owner in prior.items():
  if owner["host"]==node:(r/("prior_"+node+".native.log")).write_bytes(cmd(node,["cat",str(site/("logs/TP_run46_"+str(0 if node=="166"else 1)+".log"))]))
with http.open("http://172.16.10.166:9081/metrics",timeout=10)as reply:metrics=reply.read()
(r/"prior_idle.metrics").write_bytes(metrics)
for key in ["num_requests_running","num_requests_waiting"]:
 vals=re.findall(r"^vllm:"+key+r"(?:\{[^\n]*\})?\s+([0-9.eE+-]+)",metrics.decode(),re.M);assert vals and all(float(x)==0 for x in vals)
guard();event("old46_exact_cohort_all_preflight_idle_ACK_CLI_validated",native_requests=0,new_engines=0)
def stop(node):
 guard();owners=[x for x in prior.values()if x["host"]==node]
 return cmd(node,["env","GLM_EXPECTED_COHORT="+json.dumps(owners),"python3",str(plug/"stop_prior_cohort.py")],100)
with concurrent.futures.ThreadPoolExecutor(max_workers=2)as pool:
 fs=[pool.submit(stop,node)for node in ["166","167"]]
 for f in fs:f.result()
guard();event("exact_joint46_cohort_removed")
for node in ["166","167"]:
 top=cmd(node,["docker","top","glm52-single","-eo","pid,ppid,comm,args"]).decode();(r/("before_"+node+".processes")).write_text(top)
 assert not any("/bin/vllm serve "in l or "native_acl_lifecycle.py cli serve "in l or"VLLM::"in l or"bishengir-compile "in l for l in top.splitlines())
 ports=cmd(node,["ss","-ltnp"]).decode();(r/("before_"+node+".sockets")).write_text(ports)
 assert not re.search(r":(?:9081|9082|9900|9901|8000|8002|35000|35020|35030|35100|35120|35130)\s",ports)
planned=json.loads((r/"planned_launch.json").read_text());owners={}
for role in ["P","D"]:
 # P first, measure resident cohort before adding the second role. Same64NPU-visible process set, no extra devices.
 for x in [z for z in planned if z["role"]==role]:
  guard();node=x["node"];shell="ulimit -c 0; export PD_LOCAL_IP=172.16.10."+node+" PD_NIC=business; source "+str(site/"scripts/pd_common_env.sh")+"; export PYTHONPATH="+str(plug)+":$PYTHONPATH; export VLLM_HOST_IP=172.16.10."+node+" VLLM_ENABLE_RESPONSES_API_STORE=1 HCCL_LOGIC_SUPERPOD_ID="+str(x["rank"])+"; exec "+shlex.join(x["argv"])+" > "+shlex.quote(x["log"])+" 2>&1"
  cmd(node,["docker","exec","-d","glm52-single","bash","-c",shell]);event("native_role_started",role=role,node=node,rank=x["rank"],port=x["port"])
 deadline=time.monotonic()+1800
 while True:
  guard();ready={}
  for x in [z for z in planned if z["role"]==role]:
   node=x["node"];key=role+str(x["rank"]);top=cmd(node,["docker","top","glm52-single","-eo","pid,ppid,comm,args"]).decode()
   rows=[l for l in top.splitlines()if "native_acl_lifecycle.py cli serve "in l and"--port "+str(x["port"])+" "in l]
   if len(rows)!=1:raise RuntimeError("native "+key+" taskentry exited/ambiguous; preserve raw")
   pid=int(rows[0].split()[0]);code="import pathlib,json;p=pathlib.Path('/proc/"+str(pid)+"');s=(p/'stat').read_text();print(json.dumps({'boot_id':pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'start_ticks':s[s.rfind(')')+2:].split()[19],'argv':[x.decode()for x in(p/'cmdline').read_bytes().split(bytes([0]))if x]}))"
   ident=json.loads(cmd(node,["python3","-c",code]));assert ident["argv"]==x["argv"],"nativeCLIargv changed"
   o={"host":node,"pid":pid,"identity":{k:ident[k]for k in ["boot_id","start_ticks"]},"argv":ident["argv"],"role":role,"rank":x["rank"],"port":x["port"]}
   if key in owners:assert owners[key]==o,"actualnativeowner changed"
   owners[key]=o;atomic_json(r/"startup_model_identities.json",owners)
   tail=cmd(node,["tail","-n","130",x["log"]]).decode(errors="replace")
   if any(t in tail for t in ["OutOfMemoryError","Engine core initialization failed","corrupted size vs. prev_size","ValidationError:","ModuleNotFoundError:","RuntimeError:","SyntaxError:","NameError:","native control source changed:","native metadata substitution mismatch:"]):raise RuntimeError("native "+key+" startup failure, no replay")
   try:ready[key]=http.open("http://172.16.10."+node+":"+str(x["port"])+"/health",timeout=3).status==200
   except Exception:ready[key]=False
  event("native_role_readiness",role=role,ready=ready,owners=owners)
  if all(ready.values()):break
  if time.monotonic()>deadline:raise RuntimeError("native "+role+" deadline")
  time.sleep(15)
 for x in [z for z in planned if z["role"]==role]:
  log=cmd(x["node"],["cat",x["log"]],100);(r/("resident_"+role+"_"+x["node"]+".log")).write_bytes(log)
  lines=log.decode(errors="replace").splitlines()
  receipts=[l for l in lines if "GLM_DP_METADATA_INSTALLED "in l]
  assert len(receipts)==16,"nativecombinedmetadatainstallation missing"
  atomic_json(r/("installed_"+role+"_"+x["node"]+".json"),receipts)
  for endpoint in ["health","metrics"]:
   data=http.open("http://172.16.10."+x["node"]+":"+str(x["port"])+"/"+endpoint,timeout=10).read();(r/(role+"_"+x["node"]+".resident."+endpoint)).write_bytes(data)
 event("native_role_resident_profile_captured",role=role,second_role_pending=role=="P")
atomic_json(r/"adopted_model_identities.json",owners)
groups=[]
for role in ["P","D"]:
 members={k:v for k,v in owners.items()if v["role"]==role};epoch=hashlib.sha256(json.dumps(members,sort_keys=True).encode()).hexdigest();groups.append({"id":"PD-"+role+"-DP2-TP16-DCP16-EP32-Run47","epoch":epoch,"members":list(members)})
atomic_json(r/"execution_groups.json",groups);event("native_dual_resident_PD_ready",owners=owners,groups=groups)
