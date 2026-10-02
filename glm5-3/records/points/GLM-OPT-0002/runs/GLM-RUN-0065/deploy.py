from pathlib import Path
import json,sys,subprocess,shlex,hashlib,re,time,urllib.request,concurrent.futures
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc,same_process
from owner_guard import start_watchdog,guard
start_watchdog();r=Path(__file__).parent;g=r.parents[4];p=r.parents[1];site=Path("/data/tiankuan/wio/glm52-pd/deploy");plug=site/"plugins/dp_run65";planned=json.loads((r/"planned_launch.json").read_text());old=r.parent/"GLM-RUN-0064"
s=json.loads((old/"state.json").read_text());assert s["status"]=="completed"and not same_process(s["owner"])
a=json.loads((old/"reduction_brief.json").read_text());assert a["functional_acceptance"]and a["effective_public_output_tokens"]==128 and all(x["native_cold_counters_confirmed"]for x in a["requests"])
prior=json.loads((old/"adopted_model_identities.json").read_text());http=urllib.request.build_opener(urllib.request.ProxyHandler({}));n=0
def cmd(node,args,timeout=100,input=None,label=None):
 global n
 guard();n+=1
 if node=="167":args=["ssh","-o","BatchMode=yes","root@172.16.10.167",shlex.join(args)]
 z=subprocess.run(args,input=input,capture_output=True,timeout=timeout);stem=r/(label or"command_"+str(n)+"_"+node);stem.with_suffix(".stdout").write_bytes(z.stdout);stem.with_suffix(".stderr").write_bytes(z.stderr);atomic_json(stem.with_suffix(".receipt.json"),dict(at=utc(),exit_code=z.returncode,stdout_bytes=len(z.stdout),stdout_sha256=hashlib.sha256(z.stdout).hexdigest(),stderr_bytes=len(z.stderr)));z.check_returncode();return z.stdout
def copy(node,f):
 target=plug/f.name
 if node=="166":target.write_bytes(f.read_bytes())
 else:subprocess.run(["scp","-q",str(f),"root@172.16.10.167:"+str(target)],capture_output=True,check=True)
 assert cmd(node,["sha256sum",str(target)]).decode().split()[0]==hashlib.sha256(f.read_bytes()).hexdigest()
source_index=json.loads((p/"jobs/PD2-CPU-DIAGNOSTIC-20261002T0158Z/source_index.json").read_text());SFA=json.loads((p/"jobs/SFA58-RUNTIME-FAILURE-DETAIL-20261002T0801Z/reduction.json").read_text())["installed_sources"]
for node in["166","167"]:
 cmd(node,["mkdir","-p",str(plug)])
 for f in ["native_acl_lifecycle.py","atomic_mq_bind.py","atomic_mq_worker.py","coupled_dp_metadata_v2.py","coupled_atomic_mq_worker.py","sfa_workspace_guard.py","sfa_workspace_worker.py","glm_tool_contract.py"]:copy(node,g/"runtime"/f)
 copy(node,r/"api_config_probe.py")
 for x in source_index+SFA:
  assert cmd(node,["docker","exec","glm52-single","sha256sum",x["native_path"]]).decode().split()[0]==x["sha256"],"native sourcechanged"
 assert cmd(node,["sha256sum",str(site/"scripts/pd_common_env.sh")]).decode().split()[0]=="230e95e2ce618f2e4afb7adbc12afb5c23e588fbc6577a0b91246070edcca270"
 x=next(v for v in planned if v["node"]==node)
 shell="ulimit -c 0; export PD_LOCAL_IP=172.16.10."+node+" PD_NIC=business; source "+str(site/"scripts/pd_common_env.sh")+"; export PYTHONPATH="+str(plug)+":$PYTHONPATH VLLM_HOST_IP=172.16.10."+node+" "+" ".join(k+"="+shlex.quote(v)for k,v in x["env"].items())+"; python3 "+str(plug/"native_acl_lifecycle.py")+" script "+str(plug/"api_config_probe.py")+" "+shlex.quote(json.dumps(x["argv"]))
 raw=cmd(node,["docker","exec","glm52-single","bash","-c",shell],180,label="CPU_fullCLI_"+node);acks=[json.loads(l)for l in raw.decode().splitlines()if l.startswith('{"event":')];assert len(acks)==3 and acks[0]["event"]=="task_acl_init"and acks[-1]["event"]=="task_acl_finalize"and acks[0]["returncode"]==acks[-1]["returncode"]==0 and acks[1]["event"]=="full_native_CLI_API_config_valid"
 code="import sys,pathlib;sys.path.insert(0,"+repr(str(plug))+");from coupled_dp_metadata_v2 import make_sources;make_sources(pathlib.Path('/vllm-workspace/vllm-ascend/vllm_ascend/worker/model_runner_v1.py').read_text());print('native_metadata_guard_passed')"
 cmd(node,["docker","exec","glm52-single","python3","-c",code],label="CPU_metadata_"+node)
 for ep in["health","metrics"]:
  with http.open("http://172.16.10."+node+":"+str(prior["D"+str(x["rank"])]["port"])+"/"+ep,timeout=10)as z:b=z.read();assert z.status==200;(r/("prior_"+node+"."+ep)).write_bytes(b)
  if ep=="metrics":assert all(float(v)==0 for v in re.findall(r"^vllm:num_requests_(?:running|waiting)(?:\{[^\n]*\})?\s+([0-9.eE+-]+)",b.decode(),re.M))
def cleanup(node,pre):
 a=json.loads((old/("preflight_"+node+".json")).read_text())[0];o=next(v for v in prior.values()if v["host"]==node);payload=json.dumps(dict(owner=o,members=a["retained_D_targets"],preflight=pre)).encode();return cmd(node,["python3","-c",(r/"stop_prior_cohort.py").read_text()],180,input=payload,label=("ownership_"if pre else"cleanup_")+node)
for node in["166","167"]:cleanup(node,True)
# Sequential scope allows verified peer exits after firstAPI TERM; no unattributed process may be signalled.
for node in["166","167"]:cleanup(node,False)
for node in["166","167"]:
 raw=cmd(node,["ss","-ltnp"]).decode();listening={int(x)for x in re.findall(r":([0-9]+)\s",raw)};static=set(range(62100,62132))|set(range(62300,62364))|set(range(62500,62564))|set(range(63100,63164))|{9900,9901}
 assert not listening&static,"newDnamespaceoccupied"
 (r/("before_"+node+".sockets")).write_text(raw)
for x in planned:
 node=x["node"];shell="ulimit -c 0; export PD_LOCAL_IP=172.16.10."+node+" PD_NIC=business; source "+str(site/"scripts/pd_common_env.sh")+"; export PYTHONPATH="+str(plug)+":$PYTHONPATH; export VLLM_HOST_IP=172.16.10."+node+" VLLM_ENABLE_RESPONSES_API_STORE=1 HCCL_LOGIC_SUPERPOD_ID="+str(x["rank"])+" "+" ".join(k+"="+shlex.quote(v)for k,v in x["env"].items())+"; exec "+shlex.join(x["argv"])+" > "+shlex.quote(x["log"])+" 2>&1"
 cmd(node,["docker","exec","-d","glm52-single","bash","-c",shell],label="start_"+node)
owners={};deadline=time.monotonic()+2400
while True:
 ready=[]
 for x in planned:
  node=x["node"];top=cmd(node,["docker","top","glm52-single","-eo","pid,ppid,comm,args"]).decode();found=[l for l in top.splitlines()if"native_acl_lifecycle.py cli serve "in l and"--port "+str(x["port"])+" "in l];assert len(found)==1,"nativeAPI exitedorambiguous"
  pid=int(found[0].split()[0]);code="import pathlib,json;p=pathlib.Path('/proc/"+str(pid)+"');s=(p/'stat').read_text();print(json.dumps(dict(boot_id=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip(),start_ticks=s[s.rfind(')')+2:].split()[19],argv=[v.decode()for v in(p/'cmdline').read_bytes().split(bytes([0]))if v])))"
  a=json.loads(cmd(node,["python3","-c",code]));assert a["argv"]==x["argv"];key="D"+str(x["rank"]);o=dict(host=node,pid=pid,identity={k:a[k]for k in["boot_id","start_ticks"]},argv=a["argv"],role="D",rank=x["rank"],port=x["port"])
  if key in owners:assert owners[key]==o
  owners[key]=o;atomic_json(r/"startup_model_identities.json",owners)
  tail=cmd(node,["tail","-n","130",x["log"]]).decode(errors="replace");assert not any(v in tail for v in["OutOfMemoryError","Engine core initialization failed","ValidationError:","RuntimeError:","ValueError:","NameError:","SyntaxError:","ModuleNotFoundError:"]),"native startup failed"
  try:ready.append(http.open("http://172.16.10."+node+":"+str(x["port"])+"/health",timeout=3).status==200)
  except Exception:ready.append(False)
 atomic_json(r/"readiness.json",dict(at=utc(),ready=ready,owners=owners));print(json.dumps({"ready":ready}),flush=True)
 if all(ready):break
 if time.monotonic()>deadline:raise RuntimeError("nativecohortdeadline")
 time.sleep(15)
for x in planned:
 b=cmd(x["node"],["cat",x["log"]],label="native_resident_"+x["node"]);ls=b.decode(errors="replace").splitlines();assert len([l for l in ls if"GLM_DP_METADATA_INSTALLED "in l])==16
 skip=[json.loads(l.split("GLM_UNUSED_SFA_WORKSPACE_SKIPPED ",1)[1])for l in ls if"GLM_UNUSED_SFA_WORKSPACE_SKIPPED "in l];assert len(skip)==32 and len({v["pid"]for v in skip})==16 and all(v["actual_elements"]==0 and v["native_math_changes"]==0 for v in skip)
 assert any("Graph capturing finished"in l for l in ls)
 (r/("resident_"+x["node"]+".npu-smi")).write_bytes(cmd(x["node"],["npu-smi","info"]))
atomic_json(r/"adopted_model_identities.json",owners);atomic_json(r/"deployment_summary.json",dict(at=utc(),nativeD_only_DP2TP16DCP16EP32=True,new_models=2,actual_API_owners=owners,prefill_batch=16384,max_num_seqs=8,kv_bytes=13743895347,HCCL_MB=4096,K5_capture_sizes=[6,12,24,48],no_oldqueue=True));print("newD65nativecohortready",flush=True)
