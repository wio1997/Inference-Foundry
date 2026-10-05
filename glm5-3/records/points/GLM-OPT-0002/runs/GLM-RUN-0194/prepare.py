from common import *
old=p/"runs/GLM-RUN-0193";s=json.loads((old/"state.json").read_text());assert s["status"]=="failed"and not same_process(s["owner"])and s["failure_phase"]=="fault"
assert json.loads((old/"prepare.phase.json").read_text())["status"]=="succeeded"
ack=json.loads((old/"physical_fault_ack.json").read_text());assert ack["both_exact_old_native_domains_retired"]and all(x["status"]=="fault"for x in ack["observer"]["groups"])
oldclient=json.loads((old/"client_retired_summary.json").read_text());assert oldclient["valid"]and oldclient["effective_output_tokens"]==0and all(x["status"]==503and x["rejected_before_lease_and_RPC"]for x in oldclient["requests"])
roots=json.loads((r/"standalone_root_identities.json").read_text())
for k,o in roots.items():run(o["host"],["python3","-c",(r/"vacant_probe.py").read_text()],"vacant_before_"+k,input=json.dumps(dict(old=o)).encode())
proof=json.loads((r/"public_service_proof.json").read_text());assert same_process(proof["host"])and ref(proof["container"]["config"]["path"])==proof["container"]["config"]
assert [v.decode()for v in Path("/proc/"+str(proof["host"]["pid"])+"/cmdline").read_bytes().split(bytes([0]))if v]==proof["argv"]
snap=request("/control/replicas");assert all(x["group_faulted"]and not x["active_requests"]for x in snap["replicas"])
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
 node=x["host"];path=site/"plugins/coupled_pp194"
 atomic_json(r/("plugin_install_"+node+".json"),json.loads(run(node,["python3","-c",install],"plugin_install_"+node,input=json.dumps(dict(path=str(path),files=files)).encode())))
 shell="ulimit -c 0; export PD_LOCAL_IP=172.16.10."+node+" PD_NIC=business; source "+str(site/"scripts/pd_common_env.sh")+"; export PYTHONPATH="+str(path)+":$PYTHONPATH VLLM_HOST_IP=172.16.10."+node+" "+" ".join(a+"="+shlex.quote(b)for a,b in x["environment"].items())+"; python3 "+str(path/"native_acl_lifecycle.py")+" script "+str(path/"api_config_probe.py")+" "+shlex.quote(json.dumps(x["argv"]))
 raw=run(node,["docker","exec","glm52-single","bash","-c",shell],"CPU_exactplan_"+node,timeout=240);acks=[json.loads(l)for l in raw.decode().splitlines()if l.startswith('{"event":')]
 assert acks[0]["event"]=="task_acl_init"and acks[-1]["event"]=="task_acl_finalize"and acks[0]["returncode"]==acks[-1]["returncode"]==0
 v=next(x for x in acks if x["event"]=="fullCLI_PP32_exactplan");assert v["node_rank"]==x["node_rank"]and v["workers"]==v["models"]==v["inference"]==0
for k,o in roots.items():run(o["host"],["python3","-c",(r/"vacant_probe.py").read_text()],"vacant_afterCPU_"+k,input=json.dumps(dict(old=o)).encode())
client("retired")
atomic_json(r/"prepared_state.json",dict(at=utc(),native0_vacant=True,old193_retirement_reused=True,exactCLI_bothhosts_CPU=True,SDK0=True,new_inference=0,model_operations=0))
