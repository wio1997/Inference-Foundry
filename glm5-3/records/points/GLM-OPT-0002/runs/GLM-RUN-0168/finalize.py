from common import *
folder=r/"restored";roots=json.loads((folder/"standalone_root_identities.json").read_text());members=json.loads((folder/"standalone_native_members.json").read_text())
for key,o in roots.items():fresh(o,members[key],"terminal_"+key);native_idle(o["host"],9081 if o["host"]=="166"else 9900,"terminal")
for node,path,expected in[
 ("166",str(plug/"issue_budget_policy.json"),dict(schema_version=1,cohort_id="GLM-COHORT-0137",budget_tokens=4096,prefill_threshold_tokens=1024,prefill_cadence=1,serial=11)),
 ("167",str(site/"plugins/local_pp168/issue_budget_policy.json"),dict(schema_version=1,cohort_id="GLM-COHORT-0137",budget_tokens=8192,prefill_threshold_tokens=4096,prefill_cadence=1,serial=1))]:
 raw=run(node,["cat",path],"terminal_policy_"+node);assert json.loads(raw)==expected
plans=json.loads((folder/"standalone_launch.json").read_text());raw=run("167",["cat",plans["node1"]["log"]],"terminal_D1_native")
events=[json.loads(l.split("GLM_ISSUE_BUDGET_SELECTED ",1)[1])for l in raw.decode(errors="replace").splitlines()if"GLM_ISSUE_BUDGET_SELECTED "in l]
assert events and all(x["cohort"]=="GLM-COHORT-0137"and x["budget_tokens"]==8192and x["prefill_threshold_tokens"]==4096and x["prefill_cadence"]==1and x["serial"]==1and not x["fallback"]and x["native_max"]==8192for x in events),events
atomic_json(r/"policy_consumption.json",dict(at=utc(),native_D1_selected=events,policy_files_verified=True,D0_unchanged=True,operator_math_changes=0))
pilot=json.loads((r/"pilot_SDK_summary.json").read_text());assert pilot["measurement_valid"]and pilot["output_tokens"]==192
clients={mode:json.loads((r/("client_"+mode+"_summary.json")).read_text())for mode in["survivor","final"]}
assert all(x["valid"]for x in clients.values())and clients["survivor"]["effective_output_tokens"]==0
final=clients["final"];assert final["native_delta"]["D0"]["generation_tokens_total"]==0and final["native_delta"]["D1"]["generation_tokens_total"]==final["effective_output_tokens"]
public=json.loads((folder/"public_service_proof.json").read_text());assert same_process(public["host"])and public["SDK_init0"]and request("/healthcheck")["request_num"]==0
snap=request("/control/replicas");assert all(not x["active_requests"]and not x["group_faulted"]for x in snap["replicas"])
latest=json.loads((folder/"identity_observer/latest.json").read_text());assert all(x["status"]=="healthy"for x in latest["groups"])
d=dict(at=utc(),valid=True,functional_acceptance=True,verdict="INCONCLUSIVE",new_native_completed=9,effective_output_tokens=192+final["effective_output_tokens"],D0_new_output_tokens=0,D1_new_output_tokens=192+final["effective_output_tokens"],pilot=ref(r/"pilot_SDK_summary.json"),clients={mode:ref(r/("client_"+mode+"_summary.json"))for mode in clients},policy=ref(r/"policy_consumption.json"),public=ref(folder/"public_service_proof.json"),native32_healthy_idle=True,D0_epoch_unchanged=True,D1_new_epoch=True,SDK_clients_init_finalize0=True,public_SDK_init0_active=True,PDhelpers=0,STORE_replication=False,request_replay=False,limits=["Candidate TP4PP4DCP4/emptytokenmetadata guard/allocated8192 fit and finite cold pair/nativeJSON-SSE-STORE semantics only; no formal61440, stable/global capacity, repeated gain or KEEP","Native retirement exactowned process termination; SDK finalize ACK only claimed for public/clients where observed","Actual perstep batch8192 not asserted by HTTP concurrency"])
atomic_json(r/"functional_summary.json",d)
m=json.loads((r/"manifest.json").read_text());m.update(status="completed",valid=True,verdict="INCONCLUSIVE",results=d);atomic_json(r/"manifest.json",m)
print(json.dumps(d),flush=True)
