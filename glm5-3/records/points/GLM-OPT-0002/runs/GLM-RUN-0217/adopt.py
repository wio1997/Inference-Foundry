from common import *
folder=r/"restored";plans=json.loads((r/"standalone_launch.json").read_text());roots=json.loads((r/"standalone_root_identities.json").read_text());members=json.loads((r/"standalone_native_members.json").read_text())
x=plans["node0"];node="166";root=roots["node0"];new=fresh(root,members["node0"],"D0_adopted_members")
fresh(roots["node1"],members["node1"],"before_adopt_D1");native_idle("167",9900,"before_adopt")
env=new["env"];assert env["VLLM_USE_V2_MODEL_RUNNER"]=="1";assert env["GLM_ISSUE_BUDGET_COHORT"]=="GLM-COHORT-0216"and env["VLLM_PP_LAYER_PARTITION"]=="42,36"and env["VLLM_HOST_IP"]=="172.16.10.166"and env["HCCL_LOGIC_SUPERPOD_ID"]=="1"and env["VLLM_ENABLE_RESPONSES_API_STORE"]=="1"
raw=run(node,["cat",x["log"]],"D0_native_resident");assert b"Graph capturing finished"in raw and b"npu model runner v2 is in developing"in raw
markers=[json.loads(l.split("GLM_PP_EMPTY_TOKEN_GUARD_INSTALLED ",1)[1])for l in raw.decode(errors="replace").splitlines()if "GLM_PP_EMPTY_TOKEN_GUARD_INSTALLED "in l];assert len(markers)==16 and all(x==json.loads((p/"runs/GLM-RUN-0190/restored/PP_empty_guard_workers.json").read_text())["markers"][0]for x in markers)
comp=json.loads(new["root"]["argv"][new["root"]["argv"].index("--compilation-config")+1])
assert comp==dict(cudagraph_mode="FULL_DECODE_ONLY",cudagraph_capture_sizes=[3,6,9,12,15,18,21,24],max_cudagraph_capture_size=24)
assert b"'cudagraph_capture_sizes': [3, 6, 9, 12, 15, 18, 21, 24]"in raw and b"8/8"in raw
graph_lines=[l for l in raw.decode(errors="replace").splitlines()if"Graph capturing finished"in l or"GPU KV cache size"in l or"Capturing CUDA graphs"in l]
atomic_json(folder/"native_dense_capture_proof.json",dict(at=utc(),native_config=comp,capture_count8_native_progress=True,source=ref(r/"D0_native_resident.stdout"),capture_and_KV_lines=graph_lines,actual_runtime_batch_dispatch_unknown=True,operator_math_changes=0,prototype_V2=True,full_request_equivalence_unknown=True))
atomic_json(folder/"PP_empty_guard_workers.json",dict(at=utc(),workers=16,markers=markers,operator_math_edits=0,V1_guard_source_imported=True,V2_empty_metadata_execution_proof=False))
install=[json.loads(l.split("GLM_ISSUE_BUDGET_INSTALLED ",1)[1])for l in raw.decode(errors="replace").splitlines()if "GLM_ISSUE_BUDGET_INSTALLED "in l];assert len(install)==1and install[0]["allocated_scheduler_max"]==8192
local=[json.loads(l.split("GLM_LOCAL_PREFILL_CADENCE_INSTALLED ",1)[1])for l in raw.decode(errors="replace").splitlines()if "GLM_LOCAL_PREFILL_CADENCE_INSTALLED "in l]
assert len(local)==1and local[0]["native_scheduler"]=="AsyncScheduler"and local[0]["geometry"]==dict(DP=1,TP=8,PP=2,DCP=8,no_EP=True,no_KV_transfer=True,authority="single native scheduler",sequence_is_not_physical_GPU_steps=True)
assert local[0]["native_DP1_throttle_always_false"]and local[0]["native_math_changes"]==0and local[0]["eligibility_conservation"]and local[0]["source_control"]=="issue_budget_scheduler_v5"
assert json.loads(new["root"]["argv"][new["root"]["argv"].index("--speculative-config")+1])["num_speculative_tokens"]==2
atomic_json(folder/"local_cadence_installed.json",dict(at=utc(),markers=local,actual_APPLIED_still_pending=True))
roots["node0"]=root;members["node0"]=new
atomic_json(folder/"standalone_launch.json",plans);atomic_json(folder/"standalone_root_identities.json",roots);atomic_json(folder/"standalone_native_members.json",members)
material=json.loads((r/"native_engines_resident.json").read_text())
for e in material["engines"]:
 if e["replica_id"]=="D0":
  e["id"]="local-216-166";e["plans"]=dict(node0=x);e["roots"]=dict(node0=root);e["members"]=dict(node0=new);e.pop("routing_hint",None)
atomic_json(folder/"native_engines_resident.json",material);config=render(folder,state);atomic_json(folder/"service_config.json",config)
oldconfig,_=checked_config(r/"service_config.json");before={x["id"]:x["epoch"]for x in oldconfig["native_domains"]};after={x["id"]:x["epoch"]for x in config["native_domains"]}
assert after["local-211-167"]==before["local-211-167"]and after["local-216-166"]!=before["local-214-166"]
owners=json.loads((r/"adopted_model_identities.json").read_text());owners["D0"]=dict(root,rank=0,native_dp_rank=0,native_dp_size=1,logical_alias="D0",port=9081);atomic_json(folder/"adopted_model_identities.json",owners)
fresh(roots["node1"],members["node1"],"after_restore_D1");native_idle("167",9900,"after_restore");native_idle("166",9081,"after_restore")
atomic_json(folder/"deployment_summary.json",dict(at=utc(),ready=True,native_domains=config["native_domains"],D1_epoch_unchanged=True,D0_new_epoch=True,actual_native_workers=32,PDhelpers=0,STORE_replication=False,request_replay=False,native_operator_math_unchanged=True,prototype_V2=True,V2_thinking_token_budget_gap_open=True,full_native_request_equivalence=False,D0_native_allocated_budget=8192,D0_issue_budget=8192,D0_prefill_threshold=1024,D1_native_budget=8192))
