from common import *
prev=json.loads((p/"runs/GLM-RUN-0210/state.json").read_text());assert prev["status"]=="completed"and not same_process(prev["owner"])
cpu=p/"jobs/D1-K1-GRAPH2-FULLCLI-CPU5-20261005T0831Z/reduction.json";assert ref(cpu)["sha256"]=="7505f14d8714fb9c333b84d0b67af8d3158a38ada7fd53986812dff887d5338e"
v=json.loads(cpu.read_text());assert v["CPU_config_VALID"]and v["SDKinit_finalize0"]and v["current_native32_same_idle_counters"]
assert all(v[k]==0for k in ["new_workers","new_models","new_inference","new_NPU_tensors"])
roots=json.loads((r/"standalone_root_identities.json").read_text());members=json.loads((r/"standalone_native_members.json").read_text())
for k,o in roots.items():fresh(o,members[k],"prepare_"+k);native_idle(o["host"],9081 if o["host"]=="166"else 9900,"before")
proof=json.loads((r/"public_service_proof.json").read_text());assert same_process(proof["host"])and ref(proof["container"]["config"]["path"])==proof["container"]["config"]
assert [v.decode()for v in Path("/proc/"+str(proof["host"]["pid"])+"/cmdline").read_bytes().split(bytes([0]))if v]==proof["argv"]
atomic_json(r/"old_public_service_proof.json",proof)
snap=request("/control/replicas");assert all(not x["group_faulted"]and not x["active_requests"]and not x["draining"]for x in snap["replicas"])
for node,cohort,serial in[("166","0204",1),("167","0200",5)]:
 expected=dict(schema_version=1,cohort_id="GLM-COHORT-"+cohort,budget_tokens=8192,prefill_threshold_tokens=1024,prefill_cadence=1,serial=serial)
 assert json.loads(run(node,["cat",str(site/("plugins/local_pp"+cohort.lstrip("0")+"/issue_budget_policy.json"))],"prepare_policy_"+node))==expected
x=json.loads((r/"candidate_plan.json").read_text());baseline=json.loads((r/"standalone_launch.json").read_text())["node1"]
oldpath=baseline["argv"][1].rsplit("/",1)[0];normalized=json.loads(json.dumps(baseline).replace(oldpath,x["argv"][1].rsplit("/",1)[0]).replace("GLM-COHORT-0200","GLM-COHORT-0211").replace("29968","29974").replace("logs/local_pp200_167.log","logs/local_pp211_167.log"))
comp=json.loads(x["argv"][x["argv"].index("--compilation-config")+1]);assert comp==dict(cudagraph_mode="FULL_DECODE_ONLY",cudagraph_capture_sizes=[2,4,8,16,32],max_cudagraph_capture_size=32)
normalized["argv"][normalized["argv"].index("--compilation-config")+1]=x["argv"][x["argv"].index("--compilation-config")+1];assert normalized==x
files={f.name:f.read_text()for f in(r/"plugin_src").iterdir()if f.is_file()}
install="from pathlib import Path;import sys,json,hashlib,os;a=json.load(sys.stdin);p=Path(a['path']);assert not p.exists();p.mkdir();rows=[]\nfor n,s in a['files'].items():\n f=p/n;f.write_text(s);os.chmod(f,0o600 if n.endswith('.json')else 0o644);b=f.read_bytes();rows.append(dict(path=str(f),bytes=len(b),sha256=hashlib.sha256(b).hexdigest()))\nprint(json.dumps(rows))"
raw=run("167",["python3","-c",install],"install_native211",input=json.dumps(dict(path=x["argv"][1].rsplit("/",1)[0],files=files)).encode());atomic_json(r/"installed_sources.json",json.loads(raw))
client("before")
atomic_json(r/"prepared_state.json",dict(at=utc(),native32_owned_idle=True,D1_only_migration=True,D0_preserved=True,CPU=ref(cpu),only_native_extra_capture2=True,operator_math_changes=0,STORE_replication=False))
