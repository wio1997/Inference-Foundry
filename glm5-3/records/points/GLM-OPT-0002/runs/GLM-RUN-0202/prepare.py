from common import *
prev=json.loads((p/"runs/GLM-RUN-0201/state.json").read_text());assert prev["status"]=="completed"and not same_process(prev["owner"])
audit=p/"jobs/AUDIT-RUN201-20261005TPOSTMIXED/reduction.json";assert ref(audit)["sha256"]=="68679ad901878ebd8ce0080e48d8721e8e8c6a6ecf0ad33e04269c598a31395f"
cpu=p/"jobs/PP2-SHORT-NATIVE-CPU-REDUCE2-20261005T0543Z/reduction.json";assert ref(cpu)["sha256"]=="061cf413cff2ae72ad1f3c22e5220fbd12ea988aab185e4727e12d28cf64a25e"
v=json.loads(cpu.read_text());assert v["CPU_config_VALID"]and v["SDKinit_finalize0"]and v["actualCPU1_config_reused"]and not v["replayed_nativeCPU_config_calls"]
assert ref(r/"candidate_plan.json")["sha256"]==v["candidate_plan"]["sha256"]and ref(r/"installed_sources.json")["sha256"]==v["private_sources"]["sha256"]
for x in json.loads((r/"installed_sources.json").read_text()):assert ref(x["path"])==x
roots=json.loads((r/"standalone_root_identities.json").read_text());members=json.loads((r/"standalone_native_members.json").read_text())
for k,o in roots.items():fresh(o,members[k],"prepare_"+k);native_idle(o["host"],9081 if o["host"]=="166"else 9900,"before")
proof=json.loads((r/"public_service_proof.json").read_text());assert same_process(proof["host"])and ref(proof["container"]["config"]["path"])==proof["container"]["config"]
assert [v.decode()for v in Path("/proc/"+str(proof["host"]["pid"])+"/cmdline").read_bytes().split(bytes([0]))if v]==proof["argv"]
atomic_json(r/"old_public_service_proof.json",proof)
snap=request("/control/replicas");assert all(not x["group_faulted"]and not x["active_requests"]and not x["draining"]for x in snap["replicas"])
for node in ["166","167"]:
 expected=dict(schema_version=1,cohort_id="GLM-COHORT-0200",budget_tokens=8192,prefill_threshold_tokens=1024,prefill_cadence=1,serial=1)
 assert json.loads(run(node,["cat",str(site/"plugins/local_pp200/issue_budget_policy.json")],"prepare_policy_"+node))==expected
client("before")
atomic_json(r/"prepared_state.json",dict(at=utc(),native32_owned_idle=True,D0_only_migration=True,D1_K1_preserved=True,CPU=ref(cpu),CPU_replayed=0,private202_installed_source_exact=True,operator_math_changes=0,STORE_replication=False))
