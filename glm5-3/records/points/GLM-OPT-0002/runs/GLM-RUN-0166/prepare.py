from common import *
old=p/"runs/GLM-RUN-0165";s=json.loads((old/"state.json").read_text());assert s["status"]=="completed"and not same_process(s["owner"])
f=p/"jobs/AUDIT-RUN165-20261003T0925Z/reduction.json";assert ref(f)["sha256"]=="df3bbbbfd0994c1a7bcdd7d35a32bdb4d42ce06d17759a2613721b409f26c416"
f=p/"jobs/PP4-FULLCONFIG-CPU-20261003T0941Z/reduction.json";assert ref(f)["sha256"]=="52f7279cb4ed5d46ee3d79eaf461f2cba166c7c72a49157ec520091a49d52fb5"
roots=json.loads((r/"standalone_root_identities.json").read_text());members=json.loads((r/"standalone_native_members.json").read_text())
for key,o in roots.items():fresh(o,members[key],"before_"+key);native_idle(o["host"],9081 if o["host"]=="166"else 9900,"before")
proof=json.loads((p/"jobs/AUDIT-RUN165-20261003T0925Z/public_service_proof.json").read_text());assert same_process(proof["host"]);assert ref(proof["container"]["config"]["path"])==proof["container"]["config"];atomic_json(r/"old_public_service_proof.json",proof);assert request("/healthcheck")["request_num"]==0
files={f.name:f.read_text()for f in(r/"plugin_src").iterdir()};path="/data/tiankuan/wio/glm52-pd/deploy/plugins/local_pp166"
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
shell="ulimit -c 0; export PD_LOCAL_IP=172.16.10.166 PD_NIC=business; source "+str(site/"scripts/pd_common_env.sh")+"; export PYTHONPATH="+path+":"+str(g/"runtime")+":$PYTHONPATH; python3 "+path+"/native_acl_lifecycle.py script "+str(r/"cpu_config.py")+" PP4_22_20_20_16"
raw=run("166",["docker","exec","glm52-single","bash","-c",shell],"cpuPP4_22_20_20_16",timeout=180);events=[json.loads(l)for l in raw.decode().splitlines()if l.startswith('{"event":')];assert events[0]["event"]=="task_acl_init"and events[-1]["event"]=="task_acl_finalize"and events[0]["returncode"]==events[-1]["returncode"]==0
cfg=next(v for v in events if v["event"]=="CPU_geometry_config");assert cfg["config_accepted"]and cfg["PP_partition"]=="22,20,20,16"and cfg["scheduler"]["allocated_max_num_batched_tokens"]==8192and cfg["TP"]==4and cfg["PP"]==4and cfg["DCP"]==4and all(x=="full"for x in cfg["hf_boundary_types"].values());atomic_json(r/"cpuPP4_22_20_20_16_summary.json",cfg)
for node,path,expected in [("166",str(site/"plugins/local_engines137/issue_budget_policy.json"),dict(schema_version=1,cohort_id="GLM-COHORT-0137",budget_tokens=4096,prefill_threshold_tokens=1024,prefill_cadence=1,serial=11)),("167",str(site/"plugins/local_pp163/issue_budget_policy.json"),dict(schema_version=1,cohort_id="GLM-COHORT-0137",budget_tokens=8192,prefill_threshold_tokens=4096,prefill_cadence=1,serial=1))]:
 v=json.loads(run(node,["cat",path],"old_policy_"+node));assert v==expected
provenance=json.loads((r/"input_provenance.json").read_text());assert ref(r/"canonical_81932.body.json")["sha256"]==provenance["sha256"]
client("before");atomic_json(r/"prepared_state.json",dict(at=utc(),current_native32_same=True,D0_no_changes=True,CPU_PP4_22_20_20_16_budget8192=True,models_operations=0,private_new_plugin="/data/tiankuan/wio/glm52-pd/deploy/plugins/local_pp166"))
