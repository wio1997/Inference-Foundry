from common import *
old=p/"runs/GLM-RUN-0162";s=json.loads((old/"state.json").read_text());assert s["status"]=="completed"and not same_process(s["owner"])
f=p/"jobs/AUDIT-RUN162-20261003T0844Z/reduction.json";assert ref(f)["sha256"]=="0bdb834b768d5763a9b7e5d5ef61a0baaa5a43c2afd0f78dd8908ec9146502e5"
f=p/"jobs/PCP-AND-PP-FULLCONFIG-20261003T0633Z/reduction.json";assert ref(f)["sha256"]=="976c341bee5f9eb874220409b6ef31e3101e1470ba2d841c78197e97bd35147b"
roots=json.loads((r/"standalone_root_identities.json").read_text());members=json.loads((r/"standalone_native_members.json").read_text())
for key,o in roots.items():fresh(o,members[key],"before_"+key);native_idle(o["host"],9081 if o["host"]=="166"else 9900,"before")
proof=json.loads((p/"jobs/AUDIT-RUN162-20261003T0844Z/public_service_proof.json").read_text());assert same_process(proof["host"]);assert ref(proof["container"]["config"]["path"])==proof["container"]["config"];atomic_json(r/"old_public_service_proof.json",proof);assert request("/healthcheck")["request_num"]==0
files={f.name:f.read_text()for f in(r/"plugin_src").iterdir()};path="/data/tiankuan/wio/glm52-pd/deploy/plugins/local_pp163"
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
shell="ulimit -c 0; export PD_LOCAL_IP=172.16.10.166 PD_NIC=business; source "+str(site/"scripts/pd_common_env.sh")+"; export PYTHONPATH="+path+":"+str(g/"runtime")+":$PYTHONPATH; python3 "+path+"/native_acl_lifecycle.py script "+str(r/"cpu_config.py")+" PP38_40"
raw=run("166",["docker","exec","glm52-single","bash","-c",shell],"cpuPP38_40",timeout=180);events=[json.loads(l)for l in raw.decode().splitlines()if l.startswith('{"event":')];assert events[0]["event"]=="task_acl_init"and events[-1]["event"]=="task_acl_finalize"and events[0]["returncode"]==events[-1]["returncode"]==0
cfg=next(v for v in events if v["event"]=="CPU_geometry_config");assert cfg["config_accepted"]and cfg["PP_partition"]=="38,40"and cfg["scheduler"]["allocated_max_num_batched_tokens"]==8192and cfg["hf_boundary_type"]=="full";atomic_json(r/"cpuPP38_40_summary.json",cfg)
for node,path,expected in [("166",str(site/"plugins/local_engines137/issue_budget_policy.json"),dict(schema_version=1,cohort_id="GLM-COHORT-0137",budget_tokens=4096,prefill_threshold_tokens=1024,prefill_cadence=1,serial=11)),("167",str(site/"plugins/local_budget156/issue_budget_policy.json"),dict(schema_version=1,cohort_id="GLM-COHORT-0137",budget_tokens=8192,prefill_threshold_tokens=4096,prefill_cadence=1,serial=3))]:
 v=json.loads(run(node,["cat",path],"old_policy_"+node));assert v==expected
client("before");atomic_json(r/"prepared_state.json",dict(at=utc(),current_native32_same=True,D0_no_changes=True,CPU_PP38_40_budget8192=True,models_operations=0,private_new_plugin="/data/tiankuan/wio/glm52-pd/deploy/plugins/local_pp163"))
