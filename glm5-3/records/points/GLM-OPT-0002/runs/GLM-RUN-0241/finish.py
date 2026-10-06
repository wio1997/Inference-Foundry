from common import *
m=json.loads((r/"restored/native_engines_resident.json").read_text())
for e in m["engines"]:
 for key,o in e["roots"].items():fresh(o,e["members"][key],"terminal_"+e["replica_id"]);native_idle(o["host"],9081 if o["host"]=="166"else 9900,"terminal")
a=json.loads((r/"features_summary.json").read_text());b=json.loads((r/"bgfinish_summary.json").read_text())
atomic_json(r/"feature_summary.json",dict(at=utc(),functional_acceptance=True,completed=a["completed"]+b["completed"],effective_output_tokens=a["effective_outputs"]+b["effective_outputs"],features=ref(r/"features_summary.json"),bgstart=ref(r/"bgstart_summary.json"),restart=ref(r/"restart_summary.json"),bgfinish=ref(r/"bgfinish_summary.json"),native32_same=True,native_epochs_unchanged=True,SDK_clients_init_finalize0=True,public241_V14_retained=True,Current=None,limits=["Finite true E2E structured JSON/thinking0-8/null-unlimited/native validation/Responses new+previous schema typedSSE/STORE/background complete-replay/cancel through clean public restart/drain and readd","Cancelled outputs not credited; full work/cost not performance KEEP or global capacity certificate"]))
