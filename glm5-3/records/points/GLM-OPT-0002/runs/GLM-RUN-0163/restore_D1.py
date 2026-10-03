from common import *
folder=r/"restored";plans=json.loads((r/"standalone_launch.json").read_text());roots=json.loads((r/"standalone_root_identities.json").read_text());members=json.loads((r/"standalone_native_members.json").read_text())
fresh(roots["node0"],members["node0"],"before_restore_D0");native_idle("166",9081,"before_restore")
raw=run("167",["ss","-ltnp"],"D1_before_restore_sockets").decode();listening={int(a)for a in re.findall(r":([0-9]+)\s",raw)}
assert not listening&({9081,9900,8000,29958}|set(range(63100,63164))|set(range(62500,62564)))
x=json.loads((r/"candidate_plan.json").read_text());plans["node1"]=x;node="167";native_plug=site/"plugins/local_pp163"
# New task-private plugin copies; allocated/issue budget8192, threshold4096; native math unchanged.
assert run(node,["test","!","-e",x["log"]],"new_D1_log_absent")==b""
shell="ulimit -c 0; export PD_LOCAL_IP=172.16.10."+node+" PD_NIC=business; source "+str(site/"scripts/pd_common_env.sh")+"; export PYTHONPATH="+str(native_plug)+":$PYTHONPATH VLLM_HOST_IP=172.16.10."+node+" VLLM_ENABLE_RESPONSES_API_STORE=1 "+" ".join(k+"="+shlex.quote(v)for k,v in x["environment"].items())+"; exec "+shlex.join(x["argv"])+" > "+shlex.quote(x["log"])+" 2>&1"
run(node,["docker","exec","-d","glm52-single","bash","-c",shell],"D1_restore_start")
root=None;deadline=time.monotonic()+2400
while True:
 guard();time.sleep(10);template=root or dict(host=node,argv=x["argv"],role="API")
 v=json.loads(run(node,["python3","-c",(r/"live_probe.py").read_text()],"D1_startup_members",input=json.dumps(dict(owner=template,NPU_count=None)).encode()));root=v["root"];atomic_json(folder/"startup_root.json",root)
 text=run(node,["tail","-n","150",x["log"]],"D1_startup_tail").decode(errors="replace")
 assert not any(t in text for t in ["OutOfMemoryError","Engine core initialization failed","ValidationError:","RuntimeError:","ValueError:","NameError:","SyntaxError:","ModuleNotFoundError:","AssertionError:"]),"D1 native startup failed"
 ready=False
 try:
  with http.open("http://172.16.10.167:9900/health",timeout=3)as res:ready=res.status==200
 except Exception:pass
 atomic_json(folder/"readiness.json",dict(at=utc(),ready=ready,root=root));print(json.dumps(dict(D1_ready=ready,root=root["pid"])),flush=True)
 if ready:break
 assert time.monotonic()<deadline
new=json.loads(run(node,["python3","-c",(r/"live_probe.py").read_text()],"D1_ready_members",input=json.dumps(dict(owner=root,NPU_count=16)).encode()))
env=new["env"];assert env["GLM_ISSUE_BUDGET_COHORT"]=="GLM-COHORT-0137"and env["VLLM_PP_LAYER_PARTITION"]=="38,40"and env["VLLM_HOST_IP"]=="172.16.10.167"and env["HCCL_LOGIC_SUPERPOD_ID"]=="2"and env["VLLM_ENABLE_RESPONSES_API_STORE"]=="1"
raw=run(node,["cat",x["log"]],"D1_native_resident");assert b"Graph capturing finished"in raw
install=[json.loads(l.split("GLM_ISSUE_BUDGET_INSTALLED ",1)[1])for l in raw.decode(errors="replace").splitlines()if "GLM_ISSUE_BUDGET_INSTALLED "in l];assert len(install)==1and install[0]["native_base"]=="AsyncScheduler"and install[0]["allocated_scheduler_max"]==8192
roots["node1"]=root;members["node1"]=new
atomic_json(folder/"standalone_launch.json",plans);atomic_json(folder/"standalone_root_identities.json",roots);atomic_json(folder/"standalone_native_members.json",members)
material=json.loads((r/"native_engines_resident.json").read_text())
for e in material["engines"]:
 if e["replica_id"]=="D1":
  e["plans"]=dict(node0=x);e["roots"]=dict(node0=root);e["members"]=dict(node0=new);e.pop("routing_hint",None)
atomic_json(folder/"native_engines_resident.json",material);config=render(folder,state);atomic_json(folder/"service_config.json",config)
oldconfig,_=checked_config(r/"service_config.json");before={x["id"]:x["epoch"]for x in oldconfig["native_domains"]};after={x["id"]:x["epoch"]for x in config["native_domains"]}
assert after["local-166"]==before["local-166"]and after["local-167"]!=before["local-167"]
owners=json.loads((r/"adopted_model_identities.json").read_text());owners["D1"]=dict(root,rank=0,native_dp_rank=0,native_dp_size=1,logical_alias="D1",port=9900);atomic_json(folder/"adopted_model_identities.json",owners)
fresh(roots["node0"],members["node0"],"after_restore_D0");native_idle("166",9081,"after_restore");native_idle("167",9900,"after_restore")
atomic_json(folder/"deployment_summary.json",dict(at=utc(),ready=True,native_domains=config["native_domains"],D0_epoch_unchanged=True,D1_new_epoch=True,actual_native_workers=32,PDhelpers=0,STORE_replication=False,request_replay=False,native_operator_math_unchanged=True,D1_native_allocated_budget=8192,D1_issue_budget=8192,D1_prefill_threshold=4096,D0_native_budget=4096))
