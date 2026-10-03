from common import *
old=p/"runs/GLM-RUN-0174";s=json.loads((old/"state.json").read_text());assert s["status"]=="completed"and not same_process(s["owner"])
assert ref(p/"jobs/AUDIT-RUN174-20261003T1124Z/reduction.json")["sha256"]=="50b8e4ed1cff75904926e3541ec16c6ec7ce75f375b828be217fdfa2f9b04654"
roots=json.loads((r/"standalone_root_identities.json").read_text());members=json.loads((r/"standalone_native_members.json").read_text())
for key,o in roots.items():fresh(o,members[key],"prepare_"+key);native_idle(o["host"],9081 if o["host"]=="166"else 9900,"before")
proof=json.loads((p/"jobs/AUDIT-RUN174-20261003T1124Z/public_service_proof.json").read_text());assert same_process(proof["host"])and ref(proof["container"]["config"]["path"])==proof["container"]["config"]
assert [v.decode()for v in Path("/proc/"+str(proof["host"]["pid"])+"/cmdline").read_bytes().split(bytes([0]))if v]==proof["argv"]
assert all(not v["group_faulted"]and not v["active_requests"]and not v["draining"]for v in request("/control/replicas")["replicas"]);atomic_json(r/"old_public_service_proof.json",proof)
for node,path,expected in[("166",str(plug/"issue_budget_policy.json"),dict(schema_version=1,cohort_id="GLM-COHORT-0137",budget_tokens=4096,prefill_threshold_tokens=1024,prefill_cadence=1,serial=11)),("167",str(site/"plugins/local_pp168/issue_budget_policy.json"),dict(schema_version=1,cohort_id="GLM-COHORT-0137",budget_tokens=8192,prefill_threshold_tokens=4096,prefill_cadence=1,serial=7))]:
 assert json.loads(run(node,["cat",path],"prepare_policy_"+node))==expected
files={f.name:f.read_text()for f in(r/"plugin_src").iterdir()}
install="""from pathlib import Path
import sys,json,hashlib,os
a=json.load(sys.stdin);p=Path(a['path']);assert not p.exists();p.mkdir();rows=[]
for name,s in a['files'].items():
 assert Path(name).name==name
 f=p/name;f.write_text(s);os.chmod(f,0o600 if name.endswith('.json')else 0o644);b=f.read_bytes();rows.append(dict(path=str(f),bytes=len(b),sha256=hashlib.sha256(b).hexdigest()))
print(json.dumps(rows))
"""
atomic_json(r/"plugin_install_166.json",json.loads(run("166",["python3","-c",install],"plugin_install_166",input=json.dumps(dict(path=str(site/"plugins/local_pp175"),files=files)).encode())))
shell="ulimit -c 0; export PD_LOCAL_IP=172.16.10.166 PD_NIC=business; source "+str(site/"scripts/pd_common_env.sh")+"; export PYTHONPATH="+str(r)+":"+str(g/"runtime")+":"+str(plug)+":$PYTHONPATH; python3 "+str(plug/"native_acl_lifecycle.py")+" script "+str(r/"cpu_config.py")+" D0_PP4"
raw=run("166",["docker","exec","glm52-single","bash","-c",shell],"CPU_geometry",timeout=300)
events=[json.loads(l)for l in raw.decode().splitlines()if l.startswith('{"event":')];assert events[0]["event"]=="task_acl_init"and events[-1]["event"]=="task_acl_finalize"and events[0]["returncode"]==events[-1]["returncode"]==0
v=next(x for x in events if x["event"]=="CPU_geometry_config");assert v["config_accepted"]and(v["TP"],v["PP"],v["DCP"],v["world"])==(4,4,4,16)and v["workers"]==v["inference"]==v["NPU_tensor_allocations"]==0
atomic_json(r/"CPU_geometry.json",dict(at=utc(),actual=v,SDKinit_finalize0=True,source=ref(r/"cpu_config.py"),geometry_reused=ref(p/"jobs/PP4-FULLCONFIG-CPU-20261003T0941Z/reduction.json"),candidate_guard=ref(p/"runs/GLM-RUN-0168/restored/PP_empty_guard_workers.json")))
assert ref(r/"canonical_81932.body.json")["sha256"]==json.loads((r/"input_provenance.json").read_text())["sha256"]
client("before")
atomic_json(r/"prepared_state.json",dict(at=utc(),native32_same=True,D1_retained=True,new_target="166 D0 PP4",model_operations=0,old_D0_policy_unchanged=True,D1_policy_unchanged=True))
