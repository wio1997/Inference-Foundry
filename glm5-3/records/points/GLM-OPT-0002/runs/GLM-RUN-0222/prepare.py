from common import *
prev=json.loads((p/"runs/GLM-RUN-0221/state.json").read_text());assert prev["status"]=="completed" and not same_process(prev["owner"])
audit=p/"jobs/AUDIT-RUN221-METADATA-20261005T1150Z-v2/reduction.json"
assert ref(audit)["sha256"]=="0d416a76b6e20618e4ae3e0b19097e356a9802399202e2eb2288d92a34470573"
v=json.loads(audit.read_text());assert v["HTTP_status"]==500 and v["D0_native_NPUworkers_inactive"] and v["D0_NPU"]==0 and v["extent_mismatches"]==[]
roots=json.loads((r/"standalone_root_identities.json").read_text());members=json.loads((r/"standalone_native_members.json").read_text())
fresh(roots["node1"],members["node1"],"prepare_node1");native_idle("167",9900,"before")
assert all(not same_process(dict(pid=x["pid"],**x["identity"])) for x in members["node0"]["owned_targets"] if x["pid"] in members["node0"]["npu_worker_pids"])
proof=json.loads((r/"old_public_service_proof.json").read_text());assert same_process(proof["host"])and ref(proof["container"]["config"]["path"])==proof["container"]["config"]
assert [x.decode()for x in Path("/proc/"+str(proof["host"]["pid"])+"/cmdline").read_bytes().split(bytes([0]))if x]==proof["argv"]
snap=request("/control/replicas");assert all(not x["active_requests"]and not x["draining"]for x in snap["replicas"]);by={x["id"]:x for x in snap["replicas"]};assert by["D0"]["group_faulted"]and not by["D1"]["group_faulted"]
for node,cohort in[("167","0211")]:
 expected=dict(schema_version=1,cohort_id="GLM-COHORT-"+cohort,budget_tokens=8192,prefill_threshold_tokens=1024,prefill_cadence=1,serial=1)
 assert json.loads(run(node,["cat",str(site/("plugins/local_pp"+cohort.lstrip("0")+"/issue_budget_policy.json"))],"prepare_policy_"+node))==expected
x=json.loads((r/"candidate_plan.json").read_text());assert x["environment"]["ASCEND_LAUNCH_BLOCKING"]=="0"and x["environment"]["VLLM_USE_V2_MODEL_RUNNER"]=="1"and "--enforce-eager"not in x["argv"]
files={f.name:f.read_text()for f in(r/"plugin_src").iterdir()if f.is_file()}
install="from pathlib import Path;import sys,json,os;a=json.load(sys.stdin);p=Path(a['path']);assert not p.exists();p.mkdir();rows=[((p/n).write_text(s),os.chmod(p/n,0o600 if n.endswith('.json')else 0o644))for n,s in a['files'].items()];print(json.dumps(dict(installed=True)))"
run("166",["python3","-c",install],"install_native222",input=json.dumps(dict(path=x["argv"][1].rsplit("/",1)[0],files=files)).encode())
atomic_json(r/"prepared_state.json",dict(at=utc(),retained_D1_idle=True,D1_retained=True,targetGraph_same_draftNONE_diagnostic=True,native_math_edits=0))
