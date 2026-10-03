from common import *
old=p/"runs/GLM-RUN-0166";s=json.loads((old/"state.json").read_text());assert s["status"]=="failed"and not same_process(s["owner"])
f=p/"jobs/AUDIT-FAIL166-20261003T1008Z/reduction.json";assert ref(f)["sha256"]=="da85a78bd666ab75776bf49f7920d6e2de0aaefb425dff5bfe3ff8af98de2e7c"
f=p/"jobs/PP4-FULLCONFIG-CPU-20261003T0941Z/reduction.json";assert ref(f)["sha256"]=="52f7279cb4ed5d46ee3d79eaf461f2cba166c7c72a49157ec520091a49d52fb5"
roots=json.loads((r/"standalone_root_identities.json").read_text());members=json.loads((r/"standalone_native_members.json").read_text())
for key,o in roots.items():fresh(o,members[key],"before_"+key)
native_idle("166",9081,"before")
assert next(x for x in request("/control/replicas")["replicas"] if x["id"]=="D1")["group_faulted"]
proof=json.loads((p/"jobs/AUDIT-RUN165-20261003T0925Z/public_service_proof.json").read_text());assert same_process(proof["host"]);assert ref(proof["container"]["config"]["path"])==proof["container"]["config"];atomic_json(r/"old_public_service_proof.json",proof);assert request("/healthcheck")["request_num"]==0
files={f.name:f.read_text()for f in(r/"plugin_src").iterdir()};path="/data/tiankuan/wio/glm52-pd/deploy/plugins/local_pp167"
code="""from pathlib import Path
import sys,json,hashlib,os
a=json.load(sys.stdin);p=Path(a['path']);assert not p.exists();p.mkdir()
rows=[]
for name,s in a['files'].items():
 assert Path(name).name==name
 f=p/name;f.write_text(s);os.chmod(f,0o600 if name.endswith('.json')else 0o644);b=f.read_bytes();rows.append(dict(path=str(f),bytes=len(b),sha256=hashlib.sha256(b).hexdigest()))
print(json.dumps(rows))
"""
for node in["166","167"]:atomic_json(r/("plugin_install_"+node+".json"),json.loads(run(node,["python3","-c",code],"plugin_install_"+node,input=json.dumps(dict(path=path,files=files)).encode())))
atomic_json(r/"CPU_geometry_reused.json",dict(at=utc(),unchanged_native_geometry=True,evidence=ref(p/"jobs/PP4-FULLCONFIG-CPU-20261003T0941Z/reduction.json"),empty_guard_cpu=ref(p/"jobs/PP-EMPTY-TOKEN-CPU-20261003T1013Z/reduction.json"),new_guard_checked_in_native_workers_after_restore=True))
for node,path,expected in [("166",str(site/"plugins/local_engines137/issue_budget_policy.json"),dict(schema_version=1,cohort_id="GLM-COHORT-0137",budget_tokens=4096,prefill_threshold_tokens=1024,prefill_cadence=1,serial=11)),("167",str(site/"plugins/local_pp166/issue_budget_policy.json"),dict(schema_version=1,cohort_id="GLM-COHORT-0137",budget_tokens=8192,prefill_threshold_tokens=4096,prefill_cadence=1,serial=1))]:
 v=json.loads(run(node,["cat",path],"old_policy_"+node));assert v==expected
provenance=json.loads((r/"input_provenance.json").read_text());assert ref(r/"canonical_81932.body.json")["sha256"]==provenance["sha256"]
client("survivor");atomic_json(r/"prepared_state.json",dict(at=utc(),current_native32_identity_same=True,D1_failed_not_healthy=True,D0_no_changes=True,CPU_PP4_22_20_20_16_budget8192=True,models_operations=0,private_new_plugin="/data/tiankuan/wio/glm52-pd/deploy/plugins/local_pp167"))
