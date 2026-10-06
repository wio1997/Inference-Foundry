from common import *
old=p/"runs/GLM-RUN-0240";st=json.loads((old/"state.json").read_text());assert st["status"]=="completed"and not same_process(st["owner"])
proof=json.loads((old/"restored/public_service_proof.json").read_text());assert same_process(proof["host"])and ref(proof["config"]["path"])==proof["config"]
assert [x.decode()for x in Path("/proc/"+str(proof["host"]["pid"])+"/cmdline").read_bytes().split(bytes([0]))if x]==proof["argv"]
conf,_=checked_config(proof["config"]["path"]);assert conf["native_domains"]==checked_config(r/"restored/service_config.json")[0]["native_domains"]
m=json.loads((r/"restored/native_engines_resident.json").read_text())
for e in m["engines"]:
 for key,o in e["roots"].items():fresh(o,e["members"][key],"prepare_"+e["replica_id"]);native_idle(o["host"],9081 if o["host"]=="166"else 9900,"before")
snap=request("/control/replicas");assert len(snap["replicas"])==2and all(not x["active_requests"]and not x["draining"]and not x["group_faulted"]for x in snap["replicas"])
atomic_json(r/"prepared_state.json",dict(at=utc(),native32_same_idle=True,public240_exact_V14=True,models_signalled=0))
