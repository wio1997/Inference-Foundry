from common import *
prev=json.loads((p/"runs/GLM-RUN-0217/state.json").read_text());assert prev["status"]=="failed"and not same_process(prev["owner"])and prev["completed_stages"]==["prepare","adopt"]
audit=p/"jobs/AUDIT-RUN217-FAILURE-20261005T1030Z-v2/reduction.json"
assert ref(audit)["sha256"]=="34173c5915fc32c307a8f6553120f6c71cf2d9d3cfb71ec676b1139b5b260fa7"
v=json.loads(audit.read_text());assert v["native_all_original_D0_targets_inactive"]and v["D0_NPU_processes"]==0and v["D1_all_totals_unchanged"]
roots=json.loads((r/"standalone_root_identities.json").read_text());members=json.loads((r/"standalone_native_members.json").read_text())
for x in members["node0"]["owned_targets"]:assert not same_process(dict(pid=x["pid"],**x["identity"]))
fresh(roots["node1"],members["node1"],"prepare_node1");native_idle("167",9900,"before")
proof=json.loads((r/"old_public_service_proof.json").read_text())
assert same_process(proof["host"])and ref(proof["container"]["config"]["path"])==proof["container"]["config"]
assert [x.decode()for x in Path("/proc/"+str(proof["host"]["pid"])+"/cmdline").read_bytes().split(bytes([0]))if x]==proof["argv"]
snap=request("/control/replicas");rows={x["id"]:x for x in snap["replicas"]}
assert rows["D0"]["group_faulted"]and not rows["D1"]["group_faulted"]and all(not x["active_requests"]and not x["draining"]for x in snap["replicas"])
for node,cohort in[("167","0211")]:
 expected=dict(schema_version=1,cohort_id="GLM-COHORT-"+cohort,budget_tokens=8192,prefill_threshold_tokens=1024,prefill_cadence=1,serial=1)
 assert json.loads(run(node,["cat",str(site/("plugins/local_pp"+cohort.lstrip("0")+"/issue_budget_policy.json"))],"prepare_policy_"+node))==expected

x=json.loads((r/"candidate_plan.json").read_text());assert x["environment"]["ASCEND_LAUNCH_BLOCKING"]=="1"and x["environment"]["VLLM_USE_V2_MODEL_RUNNER"]=="1"
files={f.name:f.read_text()for f in(r/"plugin_src").iterdir()if f.is_file()}
install="from pathlib import Path;import sys,json,os;a=json.load(sys.stdin);p=Path(a['path']);assert not p.exists();p.mkdir()\nfor n,s in a['files'].items():\n f=p/n;f.write_text(s);os.chmod(f,0o600 if n.endswith('.json')else 0o644)\nprint(json.dumps(dict(installed=True)))"
run("166",["python3","-c",install],"install_native218",input=json.dumps(dict(path=x["argv"][1].rsplit("/",1)[0],files=files)).encode())
atomic_json(r/"prepared_state.json",dict(at=utc(),prior_D0_all_inactive=True,D1_preserved=True,prior_failure=ref(audit),diagnostic_blocking=True,math_edits=0))
