from common import *
prior=p/"runs/GLM-RUN-0163";s=json.loads((prior/"state.json").read_text());assert s["status"]=="failed"and not same_process(s["owner"])and s["failure_phase"]=="pilot"
audit=p/"jobs/AUDIT-FAIL163V2-20261003T0924Z/reduction.json";a=json.loads(audit.read_text());assert a["readonly_terminal_acceptance"]and a["new_native_inference_outputs"]==64and a["PP38_40_native_ready"]
provenance=json.loads((r/"input_provenance.json").read_text());assert ref(r/"canonical_81932.body.json")==dict(path=str(r/"canonical_81932.body.json"),bytes=provenance["bytes"],sha256=provenance["sha256"]);assert ref(provenance["source"])["sha256"]==provenance["sha256"]
for i in[0,1]:assert (r/("pilot_"+str(i))/"pilot.py").is_file()
folder=r/"restored";roots=json.loads((folder/"standalone_root_identities.json").read_text());members=json.loads((folder/"standalone_native_members.json").read_text())
for key,o in roots.items():
 v=fresh(o,members[key],"prepare_"+key);assert v["env"]["VLLM_PP_LAYER_PARTITION"]==("42,36"if o["host"]=="166"else"38,40");native_idle(o["host"],9081 if o["host"]=="166"else 9900,"prepare")
old=json.loads((r/"old_public_service_proof.json").read_text());assert same_process(old["host"])
client("survivor")
snap=request("/control/replicas");assert all(not x["active_requests"]for x in snap["replicas"])and next(x for x in snap["replicas"]if x["id"]=="D1")["group_faulted"]
config,_=checked_config(folder/"service_config.json");assert {x["epoch"]for x in config["native_domains"]}=={x["epoch"]for x in json.loads((prior/"restored/service_config.json").read_text())["native_domains"]}
atomic_json(r/"adoption_summary.json",dict(at=utc(),valid=True,native32_owned_idle=True,native_process_restart0=True,PP38_40_epoch_adopted=True,old_public_D0_survivor=True,prior_failure_audit=ref(audit),canonical_input=ref(r/"canonical_81932.body.json"),old_D0_retained=True))
