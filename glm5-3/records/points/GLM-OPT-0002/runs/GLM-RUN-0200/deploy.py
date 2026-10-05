from common import *
folder=r/"restored";plans=json.loads((r/"candidate_plans.json").read_text());native_plug=site/"plugins/local_pp200"
for k,x in plans.items():
 node=x["host"];raw=run(node,["ss","-ltnp"],"before_sockets_"+node).decode();listening={int(a)for a in re.findall(r":([0-9]+)\s",raw)}
 assert not listening&({9081,9900,29968}|set(range(63100,63164))|set(range(62500,62564)))
 run(node,["test","!","-e",x["log"]],"new_log_absent_"+node)
 shell="ulimit -c 0; export PD_LOCAL_IP=172.16.10."+node+" PD_NIC=business; source "+str(site/"scripts/pd_common_env.sh")+"; export PYTHONPATH="+str(native_plug)+":$PYTHONPATH VLLM_HOST_IP=172.16.10."+node+" VLLM_ENABLE_RESPONSES_API_STORE=1 "+" ".join(a+"="+shlex.quote(b)for a,b in x["environment"].items())+"; exec "+shlex.join(x["argv"])+" > "+shlex.quote(x["log"])+" 2>&1"
 run(node,["docker","exec","-d","glm52-single","bash","-c",shell],"start_"+node)
roots={};deadline=time.monotonic()+2400
while True:
 guard();time.sleep(10)
 for k,x in plans.items():
  template=roots.get(k)or dict(host=x["host"],argv=x["argv"],role="API"if x["api"]else"headless")
  v=json.loads(run(x["host"],["python3","-c",(r/"live_probe.py").read_text()],"startup_"+k,input=json.dumps(dict(owner=template,NPU_count=None)).encode()));roots[k]=v["root"];atomic_json(folder/"startup_root_identities.json",roots)
  raw=run(x["host"],["tail","-n","180",x["log"]],"startup_tail_"+k).decode(errors="replace")
  assert not any(t in raw for t in["OutOfMemoryError","Engine core initialization failed","ValidationError:","RuntimeError:","ValueError:","NameError:","SyntaxError:","ModuleNotFoundError:","AssertionError:"]),"native startupfailed "+x["host"]
 ready=False
 try:
  ready=all(http.open("http://172.16.10."+x["host"]+":"+str(9081 if x["host"]=="166"else 9900)+"/health",timeout=3).status==200for x in plans.values())
 except Exception:pass
 atomic_json(folder/"readiness.json",dict(at=utc(),ready=ready,roots=roots));print(json.dumps(dict(ready=ready,roots={k:v["pid"]for k,v in roots.items()})),flush=True)
 if ready:break
 assert time.monotonic()<deadline
members={};guards=[]
for k,x in plans.items():
 v=json.loads(run(x["host"],["python3","-c",(r/"live_probe.py").read_text()],"ready_"+k,input=json.dumps(dict(owner=roots[k],NPU_count=16)).encode()));members[k]=v
 env=v["env"];assert env["GLM_ISSUE_BUDGET_COHORT"]=="GLM-COHORT-0200"and env["VLLM_PP_LAYER_PARTITION"]=="22,20,20,16"and env["HCCL_LOGIC_SUPERPOD_ID"]==("1"if x["host"]=="166"else"2")and env["VLLM_ENABLE_RESPONSES_API_STORE"]=="1"
 raw=run(x["host"],["cat",x["log"]],"native_resident_"+k);text=raw.decode(errors="replace")
 rows=[json.loads(l.split("GLM_PP_EMPTY_TOKEN_GUARD_INSTALLED ",1)[1])for l in text.splitlines()if"GLM_PP_EMPTY_TOKEN_GUARD_INSTALLED "in l];assert len(rows)==16;guards+=rows
 if x["api"]:
  assert b"Graph capturing finished"in raw
  rows=[json.loads(l.split("GLM_LOCAL_PREFILL_CADENCE_INSTALLED ",1)[1])for l in text.splitlines()if"GLM_LOCAL_PREFILL_CADENCE_INSTALLED "in l]
  assert len(rows)==1and rows[0]["geometry"]==dict(DP=1,TP=4,PP=4,DCP=4,no_EP=True,no_KV_transfer=True,authority="single native scheduler",sequence_is_not_physical_GPU_steps=True)and rows[0]["source_control"]=="issue_budget_scheduler_v5"
  atomic_json(folder/"local_cadence_installed.json",rows)
assert len(guards)==32and all(v==json.loads((p/"runs/GLM-RUN-0190/restored/PP_empty_guard_workers.json").read_text())["markers"][0]for v in guards)
atomic_json(folder/"PP_empty_guard_workers.json",dict(workers=32,markers=guards,operator_math_edits=0))
for n,v in[("standalone_launch.json",plans),("standalone_root_identities.json",roots),("standalone_native_members.json",members)]:atomic_json(folder/n,v)

material=dict(placement=dict(kind="shape_split",input_threshold_bytes=32768,prefill_members=["D1"],decode_members=["D0"]),engines=[dict(id="local-200-"+x["host"],replica_id="D0"if x["host"]=="166"else"D1",plans={k:x},roots={k:roots[k]},members={k:members[k]})for k,x in plans.items()])
atomic_json(folder/"native_engines_resident.json",material);config=render(folder,state);atomic_json(folder/"service_config.json",config)
atomic_json(folder/"adopted_model_identities.json",{("D0"if o["host"]=="166"else"D1"):dict(o,rank=0 if o["host"]=="166"else 1,native_dp_rank=0,native_dp_size=1,logical_alias="D0"if o["host"]=="166"else"D1",port=9081 if o["host"]=="166"else 9900)for k,o in roots.items()})
for node,port in[("166",9081),("167",9900)]:native_idle(node,port,"after_deploy")
atomic_json(folder/"deployment_summary.json",dict(at=utc(),ready=True,native_domains=config["native_domains"],NPU_workers=32,frontend_count=2,headless_count=0,PDhelpers=0,STORE_replication=False,request_replay=False,native_operator_math_changes=0,geometry=dict(TP=4,PP=4,DCP=4,DP=1,nnodes=1,world16_each=True),capacity_claim=False,shape_split=config["placement"]))
