from common import *
prev=json.loads((p/"runs/GLM-RUN-0225/state.json").read_text());assert prev["status"]=="completed"and not same_process(prev["owner"])
audit=p/"jobs/AUDIT-RUN225-DRAFT-DENSE-20261005T1340Z/reduction.json"
assert ref(audit)["sha256"]=="9728e24f86c557180c7702a452adcca8143c272e291552b5248c324693e393d8"
v=json.loads(audit.read_text());assert v["functional_acceptance"]and v["HTTP_status"]==200and v["effective_output_tokens"]==2and v["D0_NPU"]==16and v["draft_decode_dense"]and v["draft_replay_rows"]>0
roots=json.loads((r/"standalone_root_identities.json").read_text());members=json.loads((r/"standalone_native_members.json").read_text())
for k,o in roots.items():
 row=fresh(o,members[k],"prepare_"+k);native_idle(o["host"],9081 if o["host"]=="166"else 9900,"before")
 if k=="node0":assert row["env"]["VLLM_USE_V2_MODEL_RUNNER"]=="1"
proof=json.loads((r/"old_public_service_proof.json").read_text())
assert same_process(proof["host"])and ref(proof["container"]["config"]["path"])==proof["container"]["config"]
assert [x.decode()for x in Path("/proc/"+str(proof["host"]["pid"])+"/cmdline").read_bytes().split(bytes([0]))if x]==proof["argv"]
snap=request("/control/replicas");rows={x["id"]:x for x in snap["replicas"]}
assert not rows["D0"]["group_faulted"]and not rows["D1"]["group_faulted"]and all(not x["active_requests"]and not x["draining"]for x in snap["replicas"])
for node,cohort in[("166","0225"),("167","0211")]:
 expected=dict(schema_version=1,cohort_id="GLM-COHORT-"+cohort,budget_tokens=8192,prefill_threshold_tokens=1024,prefill_cadence=1,serial=1)
 assert json.loads(run(node,["cat",str(site/("plugins/local_pp"+cohort.lstrip("0")+"/issue_budget_policy.json"))],"prepare_policy_"+node))==expected
atomic_json(r/"prepared_state.json",dict(at=utc(),native32_owned_idle=True,adoption_only=True,models_started=0,models_signalled=0,prior_evidence_reused=ref(audit),same_public225_retained=True))
# Runtime diagnostic gate only; no frozen source or raw native log mutation.
flag=p/"runs/GLM-RUN-0225/diagnostic_enable.json";assert flag.exists();oldflag=ref(flag);flag.unlink()
atomic_json(r/"metadata_observation_disabled.json",dict(at=utc(),prior_gate=oldflag,gate_removed=True,native_math_changes=0,models_started=0,models_signalled=0,reason="functional E2E without GPU-to-CPU diagnostic sync"))
