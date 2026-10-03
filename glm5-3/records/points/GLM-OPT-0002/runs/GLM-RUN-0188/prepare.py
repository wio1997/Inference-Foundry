from common import *
old=p/"runs/GLM-RUN-0187";v=json.loads((old/"state.json").read_text());assert v["status"]=="completed"and not same_process(v["owner"])
audit=p/"jobs/AUDIT-RUN187-20261003T1556Z/reduction.json";assert ref(audit)["sha256"]=="7bd3e404437eddc77df9b1f195adbabf93a6f4a3137d6476f83b9748b6ec6ece"
cpu=p/"jobs/LOCAL-CADENCE-CPU-20261003T1604Z/reduction.json";assert ref(cpu)["sha256"]=="e1fee5da7fd5369ba0a874f8e54fdcadc05bdd829440a34cd76e1a46ec3a2e9d"
assert json.loads(cpu.read_text())["actual"]["source_v4_sha256"]=="fc973c8c90f1bedff23e75308b769d39115d0ec1252b84a1088f6933070dc471"
roots=json.loads((r/"standalone_root_identities.json").read_text());members=json.loads((r/"standalone_native_members.json").read_text())
for key,o in roots.items():fresh(o,members[key],"prepare_"+key);native_idle(o["host"],9081 if o["host"]=="166"else 9900,"before")
proof=json.loads((r/"public_service_proof.json").read_text());assert same_process(proof["host"])and ref(proof["container"]["config"]["path"])==proof["container"]["config"]
assert [v.decode()for v in Path("/proc/"+str(proof["host"]["pid"])+"/cmdline").read_bytes().split(bytes([0]))if v]==proof["argv"];atomic_json(r/"old_public_service_proof.json",proof)
assert all(not v["group_faulted"]and not v["active_requests"]and not v["draining"]for v in request("/control/replicas")["replicas"])
for node,plugin,serial in[("166","local_pp175",5),("167","local_pp184",3)]:
 assert json.loads(run(node,["cat",str(site/"plugins"/plugin/"issue_budget_policy.json")],"prepare_policy_"+node))==dict(schema_version=1,cohort_id="GLM-COHORT-0137",budget_tokens=8192,prefill_threshold_tokens=1024,prefill_cadence=1,serial=serial)
files={f.name:f.read_text()for f in(r/"plugin_src").iterdir()}
install="""from pathlib import Path
import sys,json,hashlib,os
a=json.load(sys.stdin);p=Path(a['path']);assert not p.exists();p.mkdir();rows=[]
for name,s in a['files'].items():
 assert Path(name).name==name
 f=p/name;f.write_text(s);os.chmod(f,0o600 if name.endswith('.json')else 0o644);b=f.read_bytes();rows.append(dict(path=str(f),bytes=len(b),sha256=hashlib.sha256(b).hexdigest()))
print(json.dumps(rows))
"""
atomic_json(r/"plugin_install_166.json",json.loads(run("166",["python3","-c",install],"plugin_install_166",input=json.dumps(dict(path=str(site/"plugins/local_pp188"),files=files)).encode())))
client("before")
atomic_json(r/"prepared_state.json",dict(at=utc(),native32_owned_idle=True,D0_only_migration=True,D1_K1_preserved=True,CPU=ref(cpu),localcadence_NPU_APPLIED_pending=True,operator_math_changes=0,STORE_replication=False))
