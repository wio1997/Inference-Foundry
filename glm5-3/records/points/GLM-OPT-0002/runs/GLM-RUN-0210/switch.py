from common import *
import controls
old=json.loads((r/"old_public_service_proof.json").read_text())
controls.verify("before_public_switch")
roots=json.loads((r/"standalone_root_identities.json").read_text());members=json.loads((r/"standalone_native_members.json").read_text())
for key,o in roots.items():fresh(o,members[key],"switch_pre_"+key);native_idle(o["host"],9081 if o["host"]=="166"else 9900,"switch_pre")
from retained_store import retained_responses
for native_id,source in retained_responses():
 with http.open("http://127.0.0.1:8000/v1/responses/"+native_id,timeout=30)as response:b=response.read();assert response.status==200
 (r/(native_id+"_switch_before.wire")).write_bytes(b);assert b==Path(source).read_bytes()
retire(old,p/"runs/GLM-RUN-0204/restored/public.gateway.log","retire204_public")
obsdir=p/"runs/GLM-RUN-0204/restored/identity_observer"
for _ in range(30):
 guard()
 if(obsdir/"terminal.json").exists():break
 time.sleep(1)
else:raise RuntimeError("old HOSTobserver didnot end with exactpublicowner")
folder=r/"restored";proof=start_public(folder,"public210")
snapshot=request("/control/replicas");assert all(x["placement_policy"]=="shape_split_idle_spill"and x["shape_split_idle_spill_hint"]for x in snapshot["replicas"])
client("final")
for native_id,source in retained_responses():
 with http.open("http://127.0.0.1:8000/v1/responses/"+native_id,timeout=30)as response:b=response.read();assert response.status==200
 (r/(native_id+"_switch_after.wire")).write_bytes(b);assert b==Path(source).read_bytes()
for key,o in roots.items():fresh(o,members[key],"switch_post_"+key);native_idle(o["host"],9081 if o["host"]=="166"else 9900,"switch_post")
controls.verify("after_public_function")
out=json.loads((r/"client_final_summary.json").read_text());assert out["valid"]and out["effective_output_tokens"]==59and out["native_delta"]["D1"]["generation_tokens_total"]==0
atomic_json(r/"functional_public_summary.json",dict(at=utc(),valid=True,actual_native32_same_epochs=True,models_operations=0,private_policy_writes=0,old_public_SDK_finalize0=True,new_public_SDK_init0_active=True,SDK_clients0=True,effective_output_tokens=59,new_native_requests=5,STOREboth_existing_exact=True,typednewD0_STORE_previous_retrieve=True,semantics4_42_literal=True,new_shape_idle_spill_runtime=True,capacity_claim=False,public=ref(folder/"public_service_proof.json"),CPU_scope=ref(r/"candidate_bundle.json")))
