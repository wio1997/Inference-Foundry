from common import *
prev=json.loads((p/"runs/GLM-RUN-0203/state.json").read_text());assert prev["status"]=="completed"and not same_process(prev["owner"])
audit=p/"jobs/AUDIT-RUN203-20261005TPOSTMIXED/reduction.json";assert ref(audit)["sha256"]=="5751c86478b5f2df7b1b3041d25307476749ab628fa0c9adbefdb66a0d94dfc2"
cpu=p/"jobs/DENSE-GRAPH-PP2-FULLCLI-CPU-20261005T0625Z/reduction.json";assert ref(cpu)["sha256"]=="7088ee7f24a9fd990919dae6650faf5e7bba870e78284834cd17060e59f52609"
v=json.loads(cpu.read_text());assert v["CPU_config_VALID"]and v["SDKinit_finalize0"]and v["native_dense_Graph_CPU"]["full_native_config"]and v["current_native32_same_idle_counters"]
assert all(v[k]==0for k in["new_workers","new_models","new_inference","new_NPU_tensors"])
assert ref(r/"candidate_plan.json")["sha256"]==v["candidate_plan"]["sha256"]and ref(r/"installed_sources.json")["sha256"]==v["private_sources"]["sha256"]
for x in json.loads((r/"installed_sources.json").read_text()):assert ref(x["path"])==x
roots=json.loads((r/"standalone_root_identities.json").read_text());members=json.loads((r/"standalone_native_members.json").read_text())
for k,o in roots.items():fresh(o,members[k],"prepare_"+k);native_idle(o["host"],9081 if o["host"]=="166"else 9900,"before")
proof=json.loads((r/"public_service_proof.json").read_text());assert same_process(proof["host"])and ref(proof["container"]["config"]["path"])==proof["container"]["config"]
assert [v.decode()for v in Path("/proc/"+str(proof["host"]["pid"])+"/cmdline").read_bytes().split(bytes([0]))if v]==proof["argv"]
atomic_json(r/"old_public_service_proof.json",proof)
snap=request("/control/replicas");assert all(not x["group_faulted"]and not x["active_requests"]and not x["draining"]for x in snap["replicas"])
for node,cohort in[("166","0202"),("167","0200")]:
 expected=dict(schema_version=1,cohort_id="GLM-COHORT-"+cohort,budget_tokens=8192,prefill_threshold_tokens=1024,prefill_cadence=1,serial=1)
 assert json.loads(run(node,["cat",str(site/("plugins/local_pp"+cohort.lstrip("0")+"/issue_budget_policy.json"))],"prepare_policy_"+node))==expected
# Only legal native capture-list keys differ, after normalization of new private paths/master port/log/cohort.
a=json.loads((r/"standalone_launch.json").read_text())["node0"];b=json.loads((r/"candidate_plan.json").read_text())
ac=json.loads(a["argv"][a["argv"].index("--compilation-config")+1]);bc=json.loads(b["argv"][b["argv"].index("--compilation-config")+1])
assert ac["cudagraph_capture_sizes"]==[4,8,16,32]and bc["cudagraph_capture_sizes"]==[4,8,12,16,20,24,28,32]
ac.pop("cudagraph_capture_sizes");bc.pop("cudagraph_capture_sizes");assert ac==bc
an=json.dumps(a).replace("local_pp202","local_pp204").replace("GLM-COHORT-0202","GLM-COHORT-0204").replace("29970","29972")
aa=json.loads(an);aa["argv"][aa["argv"].index("--compilation-config")+1]=b["argv"][b["argv"].index("--compilation-config")+1];assert aa==b
client("before")
atomic_json(r/"prepared_state.json",dict(at=utc(),native32_owned_idle=True,D0_only_migration=True,D1_K1_preserved=True,CPU=ref(cpu),private204_installed_source_exact=True,only_native_capture_sizes_changed=True,operator_math_changes=0,STORE_replication=False))
