from common import *
folder=r/"restored";start_public(folder,"public193");client("final")
roots=json.loads((folder/"standalone_root_identities.json").read_text());members=json.loads((folder/"standalone_native_members.json").read_text())
for k,o in roots.items():fresh(o,members[k],"final_"+k)
native_idle("166",9081,"final")
pilot=json.loads((r/"pilot_SDK_summary.json").read_text());final=json.loads((r/"client_final_summary.json").read_text());assert pilot["measurement_valid"]and final["valid"]
atomic_json(r/"functional_summary.json",dict(at=utc(),valid=True,functional_acceptance=True,verdict="INCONCLUSIVE",effective_output_tokens=192+final["effective_output_tokens"],new_native_completed=9,pilot=ref(r/"pilot_SDK_summary.json"),client=ref(r/"client_final_summary.json"),SDK_clients_init_finalize0=True,public_SDKinit0_active=True,native32_owned_idle=True,frontend_count=1,headless_count=1,STORE_replication=False,request_replay=False,operator_math_changes=0,capacity_claim=False))
