from common import *
old=p/"runs/GLM-RUN-0198";previous=json.loads((old/"state.json").read_text());assert previous["status"]=="completed"and not same_process(previous["owner"])
audit=json.loads((p/"jobs/AUDIT-RUN198-20261005TRESUME/reduction.json").read_text());assert audit["measurement_valid"]and audit["mechanism"]["verdict"]=="EXERCISED"and audit["verdict"]=="REJECT"
roots=json.loads((r/"standalone_root_identities.json").read_text());members=json.loads((r/"standalone_native_members.json").read_text())
for k,o in roots.items():fresh(o,members[k],"prepare_"+k)
native_idle("166",9081,"before")
proof=json.loads((r/"public_service_proof.json").read_text());assert same_process(proof["host"])and ref(proof["container"]["config"]["path"])==proof["container"]["config"]
assert [v.decode()for v in Path("/proc/"+str(proof["host"]["pid"])+"/cmdline").read_bytes().split(bytes([0]))if v]==proof["argv"]
assert all(not x["group_faulted"]and not x["active_requests"]and not x["draining"]for x in request("/control/replicas")["replicas"])
atomic_json(r/"old_public_service_proof.json",proof)
files={f.name:f.read_text()for f in(r/"plugin_src").iterdir()}
install='''from pathlib import Path
import sys,json,hashlib,os
a=json.load(sys.stdin);p=Path(a["path"]);assert not p.exists();p.mkdir();rows=[]
for name,s in a["files"].items():
 assert Path(name).name==name
 f=p/name;f.write_text(s);os.chmod(f,0o600 if name.endswith(".json")else 0o644);b=f.read_bytes();rows.append(dict(path=str(f),bytes=len(b),sha256=hashlib.sha256(b).hexdigest()))
print(json.dumps(rows))
'''
plans=json.loads((r/"candidate_plans.json").read_text())
for k,x in plans.items():
 node=x["host"];path=site/"plugins/local_pp200"
 atomic_json(r/("plugin_install_"+node+".json"),json.loads(run(node,["python3","-c",install],"plugin_install_"+node,input=json.dumps(dict(path=str(path),files=files)).encode())))
 shell="ulimit -c 0; export PD_LOCAL_IP=172.16.10."+node+" PD_NIC=business; source "+str(site/"scripts/pd_common_env.sh")+"; export PYTHONPATH="+str(path)+":$PYTHONPATH VLLM_HOST_IP=172.16.10."+node+" "+" ".join(a+"="+shlex.quote(b)for a,b in x["environment"].items())+"; python3 "+str(path/"native_acl_lifecycle.py")+" script "+str(path/"api_config_probe.py")+" "+shlex.quote(json.dumps(x["argv"]))
 raw=run(node,["docker","exec","glm52-single","bash","-c",shell],"CPU_exactplan_"+node,timeout=240);acks=[json.loads(l)for l in raw.decode().splitlines()if l.startswith('{"event":')]
 assert acks[0]["event"]=="task_acl_init"and acks[-1]["event"]=="task_acl_finalize"and acks[0]["returncode"]==acks[-1]["returncode"]==0
 v=next(x for x in acks if x["event"]=="fullCLI_PP32_exactplan");assert v["node_rank"]==x["node_rank"]and v["workers"]==v["models"]==v["inference"]==0
for k,o in roots.items():fresh(o,members[k],"afterCPU_"+k)
native_idle("166",9081,"afterCPU")
client("before")
atomic_json(r/"prepared_state.json",dict(at=utc(),native32_owned_idle=True,exactCLI_bothhosts_CPU=True,SDK0=True,new_inference=0,model_operations=0))
