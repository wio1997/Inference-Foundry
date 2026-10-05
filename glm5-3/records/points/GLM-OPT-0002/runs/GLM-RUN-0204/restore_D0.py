from common import *
folder=r/"restored";plans=json.loads((r/"standalone_launch.json").read_text());roots=json.loads((r/"standalone_root_identities.json").read_text());members=json.loads((r/"standalone_native_members.json").read_text())
fresh(roots["node1"],members["node1"],"before_restore_D1");native_idle("167",9900,"before_restore")
raw=run("166",["ss","-ltnp"],"D0_before_restore_sockets").decode();listening={int(a)for a in re.findall(r":([0-9]+)\s",raw)}
assert not listening&({9900,9081,29972}|set(range(63100,63164))|set(range(62500,62564)))
x=json.loads((r/"candidate_plan.json").read_text());plans["node0"]=x;node="166";native_plug=site/"plugins/local_pp204"
# New task-private plugin copies; allocated/issue budget8192, threshold1024; native math unchanged.
assert run(node,["test","!","-e",x["log"]],"new_D0_log_absent")==b""
shell="ulimit -c 0; export PD_LOCAL_IP=172.16.10."+node+" PD_NIC=business; source "+str(site/"scripts/pd_common_env.sh")+"; export PYTHONPATH="+str(native_plug)+":$PYTHONPATH VLLM_HOST_IP=172.16.10."+node+" VLLM_ENABLE_RESPONSES_API_STORE=1 "+" ".join(k+"="+shlex.quote(v)for k,v in x["environment"].items())+"; exec "+shlex.join(x["argv"])+" > "+shlex.quote(x["log"])+" 2>&1"
run(node,["docker","exec","-d","glm52-single","bash","-c",shell],"D0_restore_start")
root=None;deadline=time.monotonic()+2400
while True:
 guard();time.sleep(10);template=root or dict(host=node,argv=x["argv"],role="API")
 v=json.loads(run(node,["python3","-c",(r/"live_probe.py").read_text()],"D0_startup_members",input=json.dumps(dict(owner=template,NPU_count=None)).encode()));root=v["root"];atomic_json(folder/"startup_root.json",root)
 text=run(node,["tail","-n","150",x["log"]],"D0_startup_tail").decode(errors="replace")
 assert not any(t in text for t in ["OutOfMemoryError","Engine core initialization failed","ValidationError:","RuntimeError:","ValueError:","NameError:","SyntaxError:","ModuleNotFoundError:","AssertionError:"]),"D0 native startup failed"
 ready=False
 try:
  with http.open("http://172.16.10.166:9081/health",timeout=3)as res:ready=res.status==200
 except Exception:pass
 atomic_json(folder/"readiness.json",dict(at=utc(),ready=ready,root=root));print(json.dumps(dict(D0_ready=ready,root=root["pid"])),flush=True)
 if ready:break
 assert time.monotonic()<deadline
new=json.loads(run(node,["python3","-c",(r/"live_probe.py").read_text()],"D0_ready_members",input=json.dumps(dict(owner=root,NPU_count=16)).encode()))
env=new["env"];assert env["GLM_ISSUE_BUDGET_COHORT"]=="GLM-COHORT-0204"and env["VLLM_PP_LAYER_PARTITION"]=="42,36"and env["VLLM_HOST_IP"]=="172.16.10.166"and env["HCCL_LOGIC_SUPERPOD_ID"]=="1"and env["VLLM_ENABLE_RESPONSES_API_STORE"]=="1"
raw=run(node,["cat",x["log"]],"D0_native_resident");assert b"Graph capturing finished"in raw
markers=[json.loads(l.split("GLM_PP_EMPTY_TOKEN_GUARD_INSTALLED ",1)[1])for l in raw.decode(errors="replace").splitlines()if "GLM_PP_EMPTY_TOKEN_GUARD_INSTALLED "in l];assert len(markers)==16 and all(x==json.loads((p/"runs/GLM-RUN-0190/restored/PP_empty_guard_workers.json").read_text())["markers"][0]for x in markers)
comp=json.loads(new["root"]["argv"][new["root"]["argv"].index("--compilation-config")+1])
assert comp==dict(cudagraph_mode="FULL_DECODE_ONLY",cudagraph_capture_sizes=[4,8,12,16,20,24,28,32],max_cudagraph_capture_size=32)
assert b"'cudagraph_capture_sizes': [4, 8, 12, 16, 20, 24, 28, 32]"in raw and b"8/8"in raw
graph_lines=[l for l in raw.decode(errors="replace").splitlines()if"Graph capturing finished"in l or"GPU KV cache size"in l or"Capturing CUDA graphs"in l]
atomic_json(folder/"native_dense_capture_proof.json",dict(at=utc(),native_config=comp,capture_count8_native_progress=True,source=ref(r/"D0_native_resident.stdout"),capture_and_KV_lines=graph_lines,actual_runtime_batch_dispatch_unknown=True,operator_math_changes=0))
atomic_json(folder/"PP_empty_guard_workers.json",dict(at=utc(),workers=16,markers=markers,operator_math_edits=0))
install=[json.loads(l.split("GLM_ISSUE_BUDGET_INSTALLED ",1)[1])for l in raw.decode(errors="replace").splitlines()if "GLM_ISSUE_BUDGET_INSTALLED "in l];assert len(install)==1and install[0]["allocated_scheduler_max"]==8192
local=[json.loads(l.split("GLM_LOCAL_PREFILL_CADENCE_INSTALLED ",1)[1])for l in raw.decode(errors="replace").splitlines()if "GLM_LOCAL_PREFILL_CADENCE_INSTALLED "in l]
assert len(local)==1and local[0]["native_scheduler"]=="AsyncScheduler"and local[0]["geometry"]==dict(DP=1,TP=8,PP=2,DCP=8,no_EP=True,no_KV_transfer=True,authority="single native scheduler",sequence_is_not_physical_GPU_steps=True)
assert local[0]["native_DP1_throttle_always_false"]and local[0]["native_math_changes"]==0and local[0]["eligibility_conservation"]and local[0]["source_control"]=="issue_budget_scheduler_v5"
assert json.loads(new["root"]["argv"][new["root"]["argv"].index("--speculative-config")+1])["num_speculative_tokens"]==3
atomic_json(folder/"local_cadence_installed.json",dict(at=utc(),markers=local,actual_APPLIED_still_pending=True))
roots["node0"]=root;members["node0"]=new
atomic_json(folder/"standalone_launch.json",plans);atomic_json(folder/"standalone_root_identities.json",roots);atomic_json(folder/"standalone_native_members.json",members)
material=json.loads((r/"native_engines_resident.json").read_text())
for e in material["engines"]:
 if e["replica_id"]=="D0":
  e["id"]="local-204-166";e["plans"]=dict(node0=x);e["roots"]=dict(node0=root);e["members"]=dict(node0=new);e.pop("routing_hint",None)
atomic_json(folder/"native_engines_resident.json",material);config=render(folder,state);atomic_json(folder/"service_config.json",config)
oldconfig,_=checked_config(r/"service_config.json");before={x["id"]:x["epoch"]for x in oldconfig["native_domains"]};after={x["id"]:x["epoch"]for x in config["native_domains"]}
assert after["local-200-167"]==before["local-200-167"]and after["local-204-166"]!=before["local-202-166"]
owners=json.loads((r/"adopted_model_identities.json").read_text());owners["D0"]=dict(root,rank=0,native_dp_rank=0,native_dp_size=1,logical_alias="D0",port=9081);atomic_json(folder/"adopted_model_identities.json",owners)
fresh(roots["node1"],members["node1"],"after_restore_D1");native_idle("167",9900,"after_restore");native_idle("166",9081,"after_restore")
atomic_json(folder/"deployment_summary.json",dict(at=utc(),ready=True,native_domains=config["native_domains"],D1_epoch_unchanged=True,D0_new_epoch=True,actual_native_workers=32,PDhelpers=0,STORE_replication=False,request_replay=False,native_operator_math_unchanged=True,D0_native_allocated_budget=8192,D0_issue_budget=8192,D0_prefill_threshold=1024,D1_native_budget=8192))
