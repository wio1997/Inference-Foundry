from pathlib import Path
import sys,json,subprocess,shlex,hashlib,urllib.request,re,ast
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc,same_process
from owner_guard import start_watchdog,guard
start_watchdog();r=Path(__file__).parent;p=r.parents[1];g=p.parents[2];site=Path("/data/tiankuan/wio/glm52-pd/deploy");plug=site/"plugins/coupled_run130"
http=urllib.request.build_opener(urllib.request.ProxyHandler({}));old=r.parent/"GLM-RUN-0128"
state=json.loads((old/"state.json").read_text());assert state["status"]=="completed"and not same_process(state["owner"])
proof=p/"jobs/AUDIT-RUN128-20261003T0020Z/reduction.json"
assert hashlib.sha256(proof.read_bytes()).hexdigest()=="ef01b4e3941ee954ef8aee8d0e008eb96c51cf5b260ef4f6e7ec23bd746497a8"
assert json.loads(proof.read_text())["complete_outputs"]==245760
for name,sha in[("CPU-TP16PP2-2NODE-20261003T0040Z","54c12b9525998609b644034c3fbe946abb1c6e0d0e04acd508da6cb25c29fa57"),("CPU-TP16PP2-HEADLESS-20261003T0109Z","c66f74695d1882889cfc04c99d9866f74029cd4d413435c58a5ec4bfd4504609"),("STANDALONE-SERVICE-CPU-20261003T0113Z","8097db6ffd2184765622006e6ab9654fd205a33f9e68e3e2f622e09fe211c32a")]:
 assert hashlib.sha256((p/"jobs"/name/"reduction.json").read_bytes()).hexdigest()==sha
def run(node,args,label,input=None,timeout=150):
 guard()
 if node=="167":args=["ssh","-o","BatchMode=yes","root@172.16.10.167",shlex.join(args)]
 z=subprocess.run(args,input=input,capture_output=True,timeout=timeout);(r/(label+".stdout")).write_bytes(z.stdout);(r/(label+".stderr")).write_bytes(z.stderr);z.check_returncode();return z.stdout
plans=json.loads((r/"standalone_launch.json").read_text())
for key,x in plans.items():
 node=x["host"];run(node,["mkdir","-p",str(plug)],"plugin_dir_"+node)
 files=[g/"runtime"/n for n in["native_acl_lifecycle.py","atomic_mq_bind.py","atomic_mq_worker.py","coupled_dp_metadata_v2.py","coupled_atomic_mq_worker.py","sfa_workspace_guard.py","sfa_workspace_worker.py","glm_tool_contract.py","issue_budget_scheduler_v3.py"]]+[r/"api_config_probe.py",r/"issue_budget_policy.json"]
 for i,f in enumerate(files):
  target=plug/f.name;run(node,["python3","-c","import pathlib,sys;pathlib.Path("+repr(str(target))+").write_bytes(sys.stdin.buffer.read())"],"copy_"+node+"_"+str(i),input=f.read_bytes())
  assert run(node,["sha256sum",str(target)],"copy_sha_"+node+"_"+str(i)).decode().split()[0]==hashlib.sha256(f.read_bytes()).hexdigest()
 for f in ["/vllm-workspace/vllm/vllm/entrypoints/cli/serve.py","/vllm-workspace/vllm/vllm/v1/executor/multiproc_executor.py","/vllm-workspace/vllm/vllm/distributed/parallel_state.py","/vllm-workspace/vllm-ascend/vllm_ascend/worker/worker.py"]:
  expected=json.loads((p/"jobs/CPU-TP16PP2-HEADLESS-20261003T0109Z/reduction.json").read_text())["proof"][0]["native_CLI_branch"]["native_sources"][f]
  assert run(node,["docker","exec","glm52-single","sha256sum",f],"native_sha_"+node+"_"+Path(f).name).decode().split()[0]==expected
 shell="ulimit -c 0; export PD_LOCAL_IP=172.16.10."+node+" PD_NIC=business; source "+str(site/"scripts/pd_common_env.sh")+"; export PYTHONPATH="+str(plug)+":$PYTHONPATH VLLM_HOST_IP=172.16.10."+node+" "+" ".join(k+"="+shlex.quote(v)for k,v in x["environment"].items())+"; python3 "+str(plug/"native_acl_lifecycle.py")+" script "+str(plug/"api_config_probe.py")+" "+shlex.quote(json.dumps(x["argv"]))
 raw=run(node,["docker","exec","glm52-single","bash","-c",shell],"CPU_config_"+node,timeout=180)
 acks=[json.loads(l)for l in raw.decode().splitlines()if l.startswith('{"event":')]
 assert acks[0]["event"]=="task_acl_init"and acks[-1]["event"]=="task_acl_finalize"and acks[0]["returncode"]==acks[-1]["returncode"]==0
 v=next(x for x in acks if x["event"]=="full_native_CLI_API_config_valid");assert v["world_size"]==32and v["local_world_size"]==16and v["node_rank"]==x["node_rank"]
owners=json.loads((old/"adopted_model_identities.json").read_text());previous=json.loads((old/"native_member_identities.json").read_text());fresh={}
for key,o in owners.items():
 raw=run(o["host"],["python3","-c",(r/"live_probe.py").read_text()],"before_native_"+key,input=json.dumps(dict(owner=o,NPU_count=16)).encode())
 v=json.loads(raw);assert v["npu_worker_pids"]==previous[key]["npu_worker_pids"]
 prior={x["pid"]:x["identity"]for x in previous[key]["owned_targets"]}
 assert all(prior[x["pid"]]==x["identity"]for x in v["owned_targets"]if x["pid"]in v["npu_worker_pids"])
 with http.open("http://172.16.10."+o["host"]+":"+str(o["port"])+"/metrics",timeout=10)as response:b=response.read()
 (r/("before_"+key+".metrics")).write_bytes(b)
 values=re.findall(r"^vllm:num_requests_(?:running|waiting)(?:\{[^\n]*\})?\s+([0-9.eE+-]+)",b.decode(),re.M);assert values and all(float(x)==0for x in values)
 fresh[key]=v
atomic_json(r/"prior_native_members.json",fresh)
with http.open("http://127.0.0.1:8000/healthcheck",timeout=10)as response:a=json.loads(response.read())
assert a["status"]=="ok"and a["request_num"]==0
with http.open("http://127.0.0.1:8000/control/replicas",timeout=10)as response:placement=json.loads(response.read())
assert len(placement["replicas"])==2and all(not x["temporarily_unhealthy"]and not x["group_faulted"]and not x["draining"]and x["active_requests"]==0for x in placement["replicas"])
atomic_json(r/"prior_public_health.json",a)
atomic_json(r/"prepared_state.json",dict(at=utc(),fresh_API2_NPU32=True,idle=True,native_CPU_both_nodes=True,SDK0=True,models_started=0,signals=0))
print("coupled130 preflight valid")
