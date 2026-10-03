from common import *
folder=r/"restored";roots=json.loads((folder/"standalone_root_identities.json").read_text());members=json.loads((folder/"standalone_native_members.json").read_text())
for key,o in roots.items():fresh(o,members[key],"terminal_"+key);native_idle(o["host"],9081 if o["host"]=="166"else 9900,"terminal")
for node,plugin,serial,threshold in [("166","local_pp175",3,1024),("167","local_pp182",1,1024)]:
 assert json.loads(run(node,["cat",str(site/"plugins"/plugin/"issue_budget_policy.json")],"terminal_policy_"+node))==dict(schema_version=1,cohort_id="GLM-COHORT-0137",budget_tokens=8192,prefill_threshold_tokens=threshold,prefill_cadence=1,serial=serial)
plans=json.loads((folder/"standalone_launch.json").read_text());raw=run("167",["cat",plans["node1"]["log"]],"terminal_D1_native")
events=[json.loads(l.split("GLM_ISSUE_BUDGET_SELECTED ",1)[1])for l in raw.decode(errors="replace").splitlines()if "GLM_ISSUE_BUDGET_SELECTED "in l]
assert events and all(x["budget_tokens"]==8192and x["prefill_threshold_tokens"]==1024and x["serial"]==1and not x["fallback"]for x in events)
assert json.loads(plans["node1"]["argv"][plans["node1"]["argv"].index("--speculative-config")+1])["num_speculative_tokens"]==1
atomic_json(r/"policy_consumption.json",dict(at=utc(),native_D1_selected=events,D0_unchanged=True,static_K1=True,operator_math_changes=0,dynamic_DCP4_guard_preserved=True))
pilot=json.loads((r/"pilot_SDK_summary.json").read_text());assert pilot["measurement_valid"]and pilot["output_tokens"]==192
clients={mode:json.loads((r/("client_"+mode+"_summary.json")).read_text())for mode in ["before","survivor","final"]}
assert all(x["valid"]for x in clients.values())and clients["before"]["effective_output_tokens"]==clients["survivor"]["effective_output_tokens"]==0
final=clients["final"];assert final["native_delta"]["D0"]["generation_tokens_total"]==0and final["native_delta"]["D1"]["generation_tokens_total"]==final["effective_output_tokens"]
public=json.loads((folder/"public_service_proof.json").read_text());assert same_process(public["host"])and public["SDK_init0"]and request("/healthcheck")["request_num"]==0
assert all(not x["active_requests"]and not x["group_faulted"]for x in request("/control/replicas")["replicas"])
assert all(x["status"]=="healthy"for x in json.loads((folder/"identity_observer/latest.json").read_text())["groups"])
d=dict(at=utc(),valid=True,functional_acceptance=True,verdict="INCONCLUSIVE",effective_output_tokens=192+final["effective_output_tokens"],new_native_completed=9,pilot=ref(r/"pilot_SDK_summary.json"),clients={mode:ref(r/("client_"+mode+"_summary.json"))for mode in clients},policy=ref(r/"policy_consumption.json"),public=ref(folder/"public_service_proof.json"),native32_healthy_idle=True,D0_epoch_unchanged=True,D1_new_epoch=True,static_D1_K1=True,D0_K3=True,SDK_clients_init_finalize0=True,public_SDK_init0_active=True,PDhelpers=0,STORE_replication=False,request_replay=False,limits=["StaticK1 fit and finite cold81932to64/short/nativeJSON-SSE-STORE semantics; no long61440/stablecapacity/globalbound/KEEP","Changed D1 native epoch retires prior168 STORE GET/previous beforeleaseRPC; D0 STORE preserved; no replication","DynamicK+DCP4 native guard unchanged; no operator/math implementation edits"])
atomic_json(r/"functional_summary.json",d);m=json.loads((r/"manifest.json").read_text());m.update(status="completed",valid=True,verdict="INCONCLUSIVE",results=d);atomic_json(r/"manifest.json",m);print(json.dumps(d))
