from common import *
prev=json.loads((p/"runs/GLM-RUN-0216/state.json").read_text());assert prev["status"]=="failed"and not same_process(prev["owner"])and prev["completed_stages"]==["prepare","fault"]
audit=p/"jobs/AUDIT-RUN216-PARTIAL-20261005T1010Z-v2/reduction.json"
assert ref(audit)["sha256"]=="58d081e74f1952fbe424e89ed9eeb9fd35f8fc692e8d75ed9c09ab6d21de95b2"
v=json.loads(audit.read_text());assert v["partial_native_fit_VALID"]and v["new_native_output_tokens"]==0and v["retained_D1_all_native_counters"]
roots=json.loads((r/"standalone_root_identities.json").read_text());members=json.loads((r/"standalone_native_members.json").read_text())
for k,o in roots.items():
 row=fresh(o,members[k],"prepare_"+k);native_idle(o["host"],9081 if o["host"]=="166"else 9900,"before")
 if k=="node0":assert row["env"]["VLLM_USE_V2_MODEL_RUNNER"]=="1"
proof=json.loads((r/"old_public_service_proof.json").read_text())
assert same_process(proof["host"])and ref(proof["container"]["config"]["path"])==proof["container"]["config"]
assert [x.decode()for x in Path("/proc/"+str(proof["host"]["pid"])+"/cmdline").read_bytes().split(bytes([0]))if x]==proof["argv"]
snap=request("/control/replicas");rows={x["id"]:x for x in snap["replicas"]}
assert rows["D0"]["group_faulted"]and not rows["D1"]["group_faulted"]and all(not x["active_requests"]and not x["draining"]for x in snap["replicas"])
for node,cohort in[("166","0216"),("167","0211")]:
 expected=dict(schema_version=1,cohort_id="GLM-COHORT-"+cohort,budget_tokens=8192,prefill_threshold_tokens=1024,prefill_cadence=1,serial=1)
 assert json.loads(run(node,["cat",str(site/("plugins/local_pp"+cohort.lstrip("0")+"/issue_budget_policy.json"))],"prepare_policy_"+node))==expected
atomic_json(r/"prepared_state.json",dict(at=utc(),native32_owned_idle=True,adoption_only=True,models_started=0,models_signalled=0,prior_evidence_reused=ref(audit),public214_faulted_D0=True))
