from common import *
import controls
controls.verify("final_policies")
roots=json.loads((r/"standalone_root_identities.json").read_text());members=json.loads((r/"standalone_native_members.json").read_text())
for key,o in roots.items():fresh(o,members[key],"terminal_"+key);native_idle(o["host"],9081 if o["host"]=="166"else 9900,"terminal")
f=json.loads((r/"functional_public_summary.json").read_text());w=json.loads((r/"dynamic_wrapper_summary.json").read_text());d=json.loads((r/"dynamic_summary.json").read_text())
assert f["valid"]and w["functional_acceptance"]and d["functional_acceptance"]and d["actual_new_requests"]==16and d["effective_public_output_tokens"]==14208and d["short_spill_requests"]
assert same_process(json.loads((r/"restored/public_service_proof.json").read_text())["host"])
assert all(x["status"]=="healthy"for x in json.loads((r/"restored/identity_observer/latest.json").read_text())["groups"])
assert all(not x["active_requests"]and not x["group_faulted"]and not x["draining"]for x in request("/control/replicas")["replicas"])
atomic_json(r/"public_spill_final.json",dict(at=utc(),functional_acceptance=True,actual_native32_same_epochs=True,models_operations=0,private_policy_writes=0,new_native_requests=21,effective_output_tokens=14267,function_outputs=59,dynamic_outputs=14208,actual_spill_requests=d["short_spill_requests"],same209workload=True,public_SDKinit0_active=True,client_SDKinit_finalize0=True,STOREboth_exact=True,Current=None,capacity_verdict="pending rawaudit"))
