from common import *
old=p/"runs/GLM-RUN-0167";s=json.loads((old/"state.json").read_text());assert s["status"]=="failed"and not same_process(s["owner"])
f=p/"jobs/AUDIT-FAIL167-20261003T1018Z/reduction.json";assert ref(f)["sha256"]=="3ff24349627f4eeb165b51a6877b235cdaa0896aee24a3cb667211e3db05a9cd"
f=p/"jobs/PP4-FULLCONFIG-CPU-20261003T0941Z/reduction.json";assert ref(f)["sha256"]=="52f7279cb4ed5d46ee3d79eaf461f2cba166c7c72a49157ec520091a49d52fb5"
roots=json.loads((r/"standalone_root_identities.json").read_text());members=json.loads((r/"standalone_native_members.json").read_text())
fresh(roots["node0"],members["node0"],"before_node0")
native_idle("166",9081,"before")
assert next(x for x in request("/control/replicas")["replicas"] if x["id"]=="D1")["group_faulted"]
proof=json.loads((p/"jobs/AUDIT-RUN165-20261003T0925Z/public_service_proof.json").read_text());assert same_process(proof["host"]);assert ref(proof["container"]["config"]["path"])==proof["container"]["config"];atomic_json(r/"old_public_service_proof.json",proof);assert request("/healthcheck")["request_num"]==0
files={f.name:f.read_text()for f in(r/"plugin_src").iterdir()};path="/data/tiankuan/wio/glm52-pd/deploy/plugins/local_pp168"
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
probe_code='import pathlib,subprocess,re,json\nraw=subprocess.check_output(["npu-smi","info"],text=True,timeout=60)\nassert raw.count("No running processes found")==8 and "VLLMWorker"not in raw\ntop=subprocess.check_output(["docker","top","glm52-single","-eo","pid,ppid,stat,comm,args"],text=True)\nfor line in top.splitlines()[1:]:\n row=line.split(None,4)\n if len(row)==5 and row[2][0]not in["Z","X"]:\n  assert not(row[3].startswith("VLLM") or "/bin/vllm serve "in row[4] or "native_acl_lifecycle.py cli serve "in row[4]),"Unexpected active native process"\nsockets=subprocess.check_output(["ss","-ltnp"],text=True)\nassert not re.search(r":(9900|29958)\\s",sockets)\nprint(json.dumps(dict(native_NPU_processes=0,native_sockets_absent=True,no_unknown_native=True,npu_smi=raw)))\n'
atomic_json(r/"prior_D1_absent.json",json.loads(run("167",["python3","-c",probe_code],"prior_D1_absent")))
provenance=json.loads((r/"input_provenance.json").read_text());assert ref(r/"canonical_81932.body.json")["sha256"]==provenance["sha256"]
client("survivor");atomic_json(r/"prepared_state.json",dict(at=utc(),D0_native16_same=True,D1_empty_after167_verified=True,D0_no_changes=True,CPU_PP4_22_20_20_16_budget8192=True,models_operations=0,private_new_plugin="/data/tiankuan/wio/glm52-pd/deploy/plugins/local_pp168"))
