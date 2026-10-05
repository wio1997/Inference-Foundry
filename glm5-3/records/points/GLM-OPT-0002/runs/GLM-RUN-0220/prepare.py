from common import *
prev=json.loads((p/"runs/GLM-RUN-0219/state.json").read_text());assert prev["status"]=="completed"and not same_process(prev["owner"])
audit=p/"jobs/AUDIT-RUN219-DIAGNOSTIC-20261005T1120Z/reduction.json"
assert ref(audit)["sha256"]=="a6359df30c1ea1bb9e99d75aa1ddf130823e6733930eda1b8a730ed1a4b18d34"
v=json.loads(audit.read_text());assert v["measurement_valid"]and v["diagnostic_semantic_pass"]and v["query_extent_all_matched"]
roots=json.loads((r/"standalone_root_identities.json").read_text());members=json.loads((r/"standalone_native_members.json").read_text())
for k,o in roots.items():fresh(o,members[k],"prepare_"+k);native_idle(o["host"],9081 if o["host"]=="166"else 9900,"before")
proof=json.loads((r/"old_public_service_proof.json").read_text());assert same_process(proof["host"])and ref(proof["container"]["config"]["path"])==proof["container"]["config"]
assert [x.decode()for x in Path("/proc/"+str(proof["host"]["pid"])+"/cmdline").read_bytes().split(bytes([0]))if x]==proof["argv"]
snap=request("/control/replicas");assert all(not x["group_faulted"]and not x["active_requests"]and not x["draining"]for x in snap["replicas"])
for node,cohort in[("166","0219"),("167","0211")]:
 expected=dict(schema_version=1,cohort_id="GLM-COHORT-"+cohort,budget_tokens=8192,prefill_threshold_tokens=1024,prefill_cadence=1,serial=1)
 assert json.loads(run(node,["cat",str(site/("plugins/local_pp"+cohort.lstrip("0")+"/issue_budget_policy.json"))],"prepare_policy_"+node))==expected
x=json.loads((r/"candidate_plan.json").read_text());assert x["environment"]["ASCEND_LAUNCH_BLOCKING"]=="0"and x["environment"]["VLLM_USE_V2_MODEL_RUNNER"]=="1"and "--enforce-eager"not in x["argv"]
files={f.name:f.read_text()for f in(r/"plugin_src").iterdir()if f.is_file()}
install="from pathlib import Path;import sys,json,os;a=json.load(sys.stdin);p=Path(a['path']);assert not p.exists();p.mkdir()\\nfor n,s in a['files'].items():\\n f=p/n;f.write_text(s);os.chmod(f,0o600 if n.endswith('.json')else 0o644)\\nprint(json.dumps(dict(installed=True)))"
run("166",["python3","-c",install],"install_native220",input=json.dumps(dict(path=x["argv"][1].rsplit("/",1)[0],files=files)).encode())
atomic_json(r/"prepared_state.json",dict(at=utc(),native32_owned_idle=True,D1_retained=True,sameGraph_diagnostic=True,native_math_edits=0))
