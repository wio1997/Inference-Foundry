from pathlib import Path
import json,sys,subprocess,shlex,hashlib,re,time,urllib.request,concurrent.futures
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc,same_process
from owner_guard import start_watchdog,guard
start_watchdog();r=Path(__file__).parent;g=r.parents[4];p=r.parents[1];site=Path("/data/tiankuan/wio/glm52-pd/deploy");plug=site/"plugins/dp_run123";planned=json.loads((r/"planned_launch.json").read_text());old=r.parent/"GLM-RUN-0121";last=r.parent/"GLM-RUN-0122"
s=json.loads((last/"state.json").read_text());assert s["status"]=="completed"and not same_process(s["owner"])
assert hashlib.sha256((last/"reduction_brief.json").read_bytes()).hexdigest()=="852ee12ddb9ca898733125ae69514cbd79ef99a7fe9322021e1afeab74f79b08"
audit=json.loads((last/"reduction_brief.json").read_text());assert audit["functional_acceptance"]and audit["effective_public_output_tokens"]==7552
assert hashlib.sha256((p/"jobs/DSA-CP-DEPENDENCY-CPU-20261002T1224Z-v2/reduction.json").read_bytes()).hexdigest()=="99c53083c93d774cb67dbc5cd5f29f312a96d20edc52015220c8c7ce883f75c0"
assert hashlib.sha256((g/"runtime/issue_budget_scheduler_v3.py").read_bytes()).hexdigest()=="4ac2c4f76bbbf778e3919090d18338cafe0a19061bc903961429ebd3707f531c"
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
for node in["167"]:
 cmd(node,["mkdir","-p",str(plug)])
 for f in ["native_acl_lifecycle.py","atomic_mq_bind.py","atomic_mq_worker.py","coupled_dp_metadata_v2.py","coupled_atomic_mq_worker.py","sfa_workspace_guard.py","sfa_workspace_worker.py","glm_tool_contract.py","issue_budget_scheduler_v3.py"]:copy(node,g/"runtime"/f)
 copy(node,r/"api_config_probe.py")
 copy(node,r/"issue_budget_policy.json")
 for x in source_index+SFA+[dict(native_path=v["path"],sha256=v["sha256"])for v in json.loads((p/"jobs/DSA-CP-DEPENDENCY-CPU-20261002T1224Z-v2/reduction.json").read_text())["sources"]]:
  assert cmd(node,["docker","exec","glm52-single","sha256sum",x["native_path"]]).decode().split()[0]==x["sha256"],"native sourcechanged"
 assert cmd(node,["sha256sum",str(site/"scripts/pd_common_env.sh")]).decode().split()[0]=="230e95e2ce618f2e4afb7adbc12afb5c23e588fbc6577a0b91246070edcca270"
 x=next(v for v in planned if v["node"]==node)
 shell="ulimit -c 0; export PD_LOCAL_IP=172.16.10."+node+" PD_NIC=business; source "+str(site/"scripts/pd_common_env.sh")+"; export PYTHONPATH="+str(plug)+":$PYTHONPATH VLLM_HOST_IP=172.16.10."+node+" "+" ".join(k+"="+shlex.quote(v)for k,v in x["env"].items())+"; python3 "+str(plug/"native_acl_lifecycle.py")+" script "+str(plug/"api_config_probe.py")+" "+shlex.quote(json.dumps(x["argv"]))
 raw=cmd(node,["docker","exec","glm52-single","bash","-c",shell],180,label="CPU_fullCLI_"+node);acks=[json.loads(l)for l in raw.decode().splitlines()if l.startswith('{"event":')];assert len(acks)==3 and acks[0]["event"]=="task_acl_init"and acks[-1]["event"]=="task_acl_finalize"and acks[0]["returncode"]==acks[-1]["returncode"]==0 and acks[1]["event"]=="full_native_CLI_API_config_valid"
 code="import sys,pathlib;sys.path.insert(0,"+repr(str(plug))+");from coupled_dp_metadata_v2 import make_sources;make_sources(pathlib.Path('/vllm-workspace/vllm-ascend/vllm_ascend/worker/model_runner_v1.py').read_text());print('native_metadata_guard_passed')"
 cmd(node,["docker","exec","glm52-single","python3","-c",code],label="CPU_metadata_"+node)
previous_members=json.loads((p/"jobs/RECONCILE-LIVE121-P89-20261002T2310Z/native_member_identities.json").read_text());refreshed={}
for key,o in prior.items():
 code="import pathlib,json,subprocess,re,sys\no=json.load(sys.stdin);boot=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()\ndef proc(pid):\n try:\n  p=pathlib.Path('/proc/'+str(pid));s=(p/'stat').read_text();v=s[s.rfind(')')+2:].split();return dict(identity=dict(boot_id=boot,start_ticks=v[19]),state=v[0])\n except FileNotFoundError:return None\np=proc(o['pid']);alive=bool(p and p['identity']==o['identity']and p['state']not in['Z','X'])\nif not alive:print(json.dumps(dict(API_alive=False)));raise SystemExit(0)\nassert [v.decode()for v in pathlib.Path('/proc/'+str(o['pid'])+'/cmdline').read_bytes().split(bytes([0]))if v]==o['argv']\nt=subprocess.check_output(['docker','top','glm52-single','-eo','pid,ppid,stat,comm,args'],text=True);rows={}\nfor l in t.splitlines()[1:]:\n x=l.split(None,4)\n if len(x)==5:rows[int(x[0])]=(int(x[1]),x[2],x[3],x[4])\nids={o['pid']}\nwhile True:\n more=ids|{pid for pid,x in rows.items()if x[0]in ids}\n if more==ids:break\n ids=more\ntargets=[dict(pid=pid,identity=proc(pid)['identity'],state=proc(pid)['state'],comm=rows[pid][2])for pid in sorted(ids)if proc(pid)and proc(pid)['state']not in['Z','X']]\nunknown=[pid for pid,x in rows.items()if x[1][0]not in['Z','X']and('/bin/vllm serve 'in x[3]or'native_acl_lifecycle.py cli serve 'in x[3]or x[2].startswith('VLLM')or'bishengir-compile 'in x[3])and pid not in ids]\nassert not unknown,unknown\nraw=subprocess.check_output(['npu-smi','info'],text=True,timeout=60);npus={int(v)for v in re.findall(r'^\\|\\s+\\d+\\s+\\d+\\s+\\|\\s+(\\d+)\\s+\\|\\s+VLLMWorker',raw,re.M)};assert len(npus)==16 and npus<=ids\ne=dict(v.decode().split('=',1)for v in pathlib.Path('/proc/'+str(o['pid'])+'/environ').read_bytes().split(bytes([0]))if b'='in v)\nprint(json.dumps(dict(API_alive=True,API_owner=o,owned_targets=targets,npu_worker_pids=sorted(npus),npu_smi=raw,env={k:e.get(k)for k in ['HCCL_BUFFSIZE','HCCL_NPU_SOCKET_PORT_RANGE','HCCL_HOST_SOCKET_PORT_RANGE','VLLM_ENABLE_RESPONSES_API_STORE','GLM_ISSUE_BUDGET_POLICY','GLM_ISSUE_BUDGET_COHORT','GLM_ISSUE_BUDGET_DEFAULT']})))\n"
 raw=cmd(o["host"],["python3","-c",code],input=json.dumps(o).encode(),label="prior_members_"+o["host"]);refreshed[key]=json.loads(raw);assert refreshed[key]["API_alive"]and refreshed[key]["API_owner"]==o and refreshed[key]["npu_worker_pids"]==previous_members[key]["npu_worker_pids"]
 original={v["pid"]:v["identity"]for v in previous_members[key]["owned_targets"]}
 assert all(original[v["pid"]]==v["identity"]for v in refreshed[key]["owned_targets"]if v["pid"]in refreshed[key]["npu_worker_pids"])
for key,o in prior.items():
 raw=cmd(o["host"],["cat",str(Path(o["argv"][1]).parent/"issue_budget_policy.json")])
 expected=dict(policy=json.loads((old/"issue_budget_policy.json").read_text()));expected=expected["policy"];expected["cohort_id"]="GLM-COHORT-0089"if o["host"]=="166"else"GLM-COHORT-0121";expected["serial"]=3 if o["host"]=="166"else 1;assert json.loads(raw)==expected
 if o["host"]in["166","167"]:
  with http.open("http://172.16.10."+o["host"]+":"+str(o["port"])+"/metrics",timeout=10)as res:b=res.read();assert res.status==200
  (r/("retained_P_idle_"+key+".metrics")).write_bytes(b)
  counts=[float(v)for v in re.findall(r"^vllm:num_requests_(?:running|waiting)(?:\{[^\n]*\})?\s+([0-9.eE+-]+)",b.decode(),re.M)]
  assert len(counts)>=2 and all(v==0 for v in counts)
atomic_json(r/"pre_cleanup_native_member_identities.json",refreshed)
def cleanup(node,pre):
 o=next(v for v in prior.values()if v["host"]==node);a=refreshed["D"+str(o["rank"])];assert a["API_owner"]==o and a["API_alive"];payload=json.dumps(dict(owner=o,members=a["owned_targets"],preflight=pre)).encode();return cmd(node,["python3","-c",(r/"stop_prior_cohort.py").read_text()],180,input=payload,label=("ownership_"if pre else"cleanup_")+node)
for node in["167"]:cleanup(node,True)
# Sequential scope allows verified peer exits after firstAPI TERM; no unattributed process may be signalled.
for node in["167"]:cleanup(node,False)
for node in["167"]:
 raw=cmd(node,["ss","-ltnp"]).decode();listening={int(x)for x in re.findall(r":([0-9]+)\s",raw)};static=set(range(62100,62132))|set(range(62300,62364))|set(range(62500,62564))|set(range(63100,63164))|{9081,9900,9901}|set(range(29200,29232))
 assert not listening&static,"newDnamespaceoccupied"
 (r/("before_"+node+".sockets")).write_text(raw)
 low,high=map(int,cmd(node,["cat","/proc/sys/net/ipv4/ip_local_port_range"]).decode().split());assert 29231<low
 assert next(x for x in planned if x["node"]=="167")["kv_port_span"]==[]
for x in planned:
 if x["node"]!="167":continue
 node=x["node"];shell="ulimit -c 0; export PD_LOCAL_IP=172.16.10."+node+" PD_NIC=business; source "+str(site/"scripts/pd_common_env.sh")+"; export PYTHONPATH="+str(plug)+":$PYTHONPATH; export VLLM_HOST_IP=172.16.10."+node+" VLLM_ENABLE_RESPONSES_API_STORE=1 HCCL_LOGIC_SUPERPOD_ID="+str(x["rank"])+" "+" ".join(k+"="+shlex.quote(v)for k,v in x["env"].items())+"; exec "+shlex.join(x["argv"])+" > "+shlex.quote(x["log"])+" 2>&1"
 cmd(node,["docker","exec","-d","glm52-single","bash","-c",shell],label="start_"+node)
owners={};deadline=time.monotonic()+2400
while True:
 ready=[]
 for x in planned:
  node=x["node"];top=cmd(node,["docker","top","glm52-single","-eo","pid,ppid,comm,args"]).decode();found=[l for l in top.splitlines()if"native_acl_lifecycle.py cli serve "in l and"--port "+str(x["port"])+" "in l];assert len(found)==1,"nativeAPI exitedorambiguous"
  pid=int(found[0].split()[0]);code="import pathlib,json;p=pathlib.Path('/proc/"+str(pid)+"');s=(p/'stat').read_text();print(json.dumps(dict(boot_id=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip(),start_ticks=s[s.rfind(')')+2:].split()[19],argv=[v.decode()for v in(p/'cmdline').read_bytes().split(bytes([0]))if v])))"
  a=json.loads(cmd(node,["python3","-c",code]));assert a["argv"]==x["argv"];key="D"+str(x["rank"]);o=dict(host=node,pid=pid,identity={k:a[k]for k in["boot_id","start_ticks"]},argv=a["argv"],role=x["role"],rank=x["rank"],native_dp_rank=0,native_dp_size=1,native_ep_size=16 if node=="166"else 8,logical_alias="D"+str(x["rank"]),port=x["port"])
  if key=="D1":o.update(native_tp_size=8,native_dcp_size=8,native_pp_size=2,pp_layer_partition=[42,36],expert_parallel_enabled=False);o["native_ep_group_world_size"]=o.pop("native_ep_size")
  if key=="D0":assert o==prior[key],"retained P89 exact API changed"
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
 b=cmd(x["node"],["cat",x["log"]],label="native_resident_"+x["node"]);ls=b.decode(errors="replace").splitlines();assert len([l for l in ls if"GLM_DP_METADATA_INSTALLED "in l])==0
 skip=[json.loads(l.split("GLM_UNUSED_SFA_WORKSPACE_SKIPPED ",1)[1])for l in ls if"GLM_UNUSED_SFA_WORKSPACE_SKIPPED "in l];assert len(skip)==0,"DSADCP native workspace must not be declared skipped by old whitelist"
 assert any("Graph capturing finished"in l for l in ls)
 installed=[json.loads(l.split("GLM_ISSUE_BUDGET_INSTALLED ",1)[1])for l in ls if"GLM_ISSUE_BUDGET_INSTALLED "in l];assert len(installed)==1 and installed[0]["native_base"]=="AsyncScheduler"and installed[0]["allocated_scheduler_max"]==4096 and installed[0]["default_budget"]==4096
 (r/("resident_"+x["node"]+".npu-smi")).write_bytes(cmd(x["node"],["npu-smi","info"]))
atomic_json(r/"adopted_model_identities.json",owners)
members={}
for key,o in owners.items():
 code="import pathlib,json,subprocess,re,sys\no=json.load(sys.stdin);boot=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()\ndef proc(pid):\n try:\n  p=pathlib.Path('/proc/'+str(pid));s=(p/'stat').read_text();v=s[s.rfind(')')+2:].split();return dict(identity=dict(boot_id=boot,start_ticks=v[19]),state=v[0])\n except FileNotFoundError:return None\np=proc(o['pid']);alive=bool(p and p['identity']==o['identity']and p['state']not in['Z','X'])\nif not alive:print(json.dumps(dict(API_alive=False)));raise SystemExit(0)\nassert [v.decode()for v in pathlib.Path('/proc/'+str(o['pid'])+'/cmdline').read_bytes().split(bytes([0]))if v]==o['argv']\nt=subprocess.check_output(['docker','top','glm52-single','-eo','pid,ppid,stat,comm,args'],text=True);rows={}\nfor l in t.splitlines()[1:]:\n x=l.split(None,4)\n if len(x)==5:rows[int(x[0])]=(int(x[1]),x[2],x[3],x[4])\nids={o['pid']}\nwhile True:\n more=ids|{pid for pid,x in rows.items()if x[0]in ids}\n if more==ids:break\n ids=more\ntargets=[dict(pid=pid,identity=proc(pid)['identity'],state=proc(pid)['state'],comm=rows[pid][2])for pid in sorted(ids)if proc(pid)and proc(pid)['state']not in['Z','X']]\nunknown=[pid for pid,x in rows.items()if x[1][0]not in['Z','X']and('/bin/vllm serve 'in x[3]or'native_acl_lifecycle.py cli serve 'in x[3]or x[2].startswith('VLLM')or'bishengir-compile 'in x[3])and pid not in ids]\nassert not unknown,unknown\nraw=subprocess.check_output(['npu-smi','info'],text=True,timeout=60);npus={int(v)for v in re.findall(r'^\\|\\s+\\d+\\s+\\d+\\s+\\|\\s+(\\d+)\\s+\\|\\s+VLLMWorker',raw,re.M)};assert len(npus)==16 and npus<=ids\ne=dict(v.decode().split('=',1)for v in pathlib.Path('/proc/'+str(o['pid'])+'/environ').read_bytes().split(bytes([0]))if b'='in v)\nprint(json.dumps(dict(API_alive=True,API_owner=o,owned_targets=targets,npu_worker_pids=sorted(npus),npu_smi=raw,env={k:e.get(k)for k in ['HCCL_BUFFSIZE','HCCL_NPU_SOCKET_PORT_RANGE','HCCL_HOST_SOCKET_PORT_RANGE','VLLM_ENABLE_RESPONSES_API_STORE','GLM_ISSUE_BUDGET_POLICY','GLM_ISSUE_BUDGET_COHORT','GLM_ISSUE_BUDGET_DEFAULT']})))\n"
 raw=cmd(o["host"],["python3","-c",code],input=json.dumps(o).encode(),label="resident_members_"+o["host"]);members[key]=json.loads(raw);assert members[key]["API_alive"]and members[key]["env"]["GLM_ISSUE_BUDGET_COHORT"]==("GLM-COHORT-0089"if o["host"]=="166"else"GLM-COHORT-0123")
atomic_json(r/"native_member_identities.json",members)
atomic_json(r/"deployment_summary.json",dict(at=utc(),native_domain_geometry={"P":dict(TP=16,DCP=16,EP=16,PP=1),"D":dict(TP=8,DCP=8,expert_parallel_enabled=False,native_comm="AllGather",PP=2,layer_partition=[42,36])},logical_API_aliases={"D0":"P166","D1":"D167"},native_DP=1,native_TP={"D0":16,"D1":8},native_DCP={"D0":16,"D1":8},expert_parallel_enabled={"D0":True,"D1":False},native_ep_group_world_size={"D0":16,"D1":8},native_PP={"D0":1,"D1":2},kv_connector={"D0":"MooncakeConnectorV1","D1":None},new_models=1,retained_native_P89=True,D_KV_port_span=[],mixed_native_cohorts={"D0":"GLM-COHORT-0089","D1":"GLM-COHORT-0123"},actual_API_owners=owners,prefill_batch=4096,max_num_seqs=8,kv_bytes=3221225472,HCCL_MB=768,native_MTP_tokens={"D0":5,"D1":3},capture_sizes={"D0":[6,12,24,48],"D1":[4,8,16,32]},DSA_CP=False,native_SP_MoE={"D0":False,"D1":False},D_profiling_mode="dynamic",profile_not_started=True,shared_expert_overlap=False,inherited_MLA_workspace_allocation="native retained",coupled_DP_metadata_installed=False,issue_budget_default=4096,policy_prefill_threshold=1024,policy_cadence=1,native_global_prefill_interval=2,native_async_preserved=True,no_oldqueue=True));print("nativeP_EP16_D_local_PP2_TP8_EPdisabled_AllGather_noKV_ready",flush=True)
