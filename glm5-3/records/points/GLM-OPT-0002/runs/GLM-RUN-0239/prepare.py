from common import *
prev=json.loads((p/"runs/GLM-RUN-0238/state.json").read_text());assert prev["status"]=="completed"and not same_process(prev["owner"])
audit=p/"jobs/AUDIT-RUN238-V1-COMPAT-20261005T1750Z-v2/reduction.json";assert ref(audit)["sha256"]=="676626042eee54385a33819266b029ce458c3bf2dc40635a877d4c1f0837610e"
assert json.loads(audit.read_text())["functional_acceptance"]
cpu=p/"jobs/NATIVE-V1-COMPATIBILITY-CPU-20261005T1820Z/reduction.json";assert ref(cpu)["sha256"]=="0a3f732f8b2404f83c6c67370393c522cf38e16ba923bf9c2a79d5621bfbd9e0"
assert json.loads(cpu.read_text())["CPU_contract_VALID"]
roots=json.loads((r/"standalone_root_identities.json").read_text());members=json.loads((r/"standalone_native_members.json").read_text())
for k,o in roots.items():fresh(o,members[k],"prepare_"+k);native_idle(o["host"],9081 if o["host"]=="166"else 9900,"before")
proof=json.loads((r/"old_public_service_proof.json").read_text());assert same_process(proof["host"])and ref(proof["container"]["config"]["path"])==proof["container"]["config"]
assert [x.decode()for x in Path("/proc/"+str(proof["host"]["pid"])+"/cmdline").read_bytes().split(bytes([0]))if x]==proof["argv"]
snap=request("/control/replicas");rows={x["id"]:x for x in snap["replicas"]};assert rows["D0"]["group_faulted"]and not rows["D1"]["group_faulted"]and all(not x["active_requests"]and not x["draining"]for x in rows.values())
for node,cohort in[("166","0228"),("167","0211")]:
 expected=dict(schema_version=1,cohort_id="GLM-COHORT-"+cohort,budget_tokens=8192,prefill_threshold_tokens=1024,prefill_cadence=1,serial=1)
 assert json.loads(run(node,["cat",str(site/("plugins/local_pp"+cohort.lstrip("0")+"/issue_budget_policy.json"))],"prepare_policy_"+node))==expected
x=json.loads((r/"candidate_plan.json").read_text());assert x["environment"]["ASCEND_LAUNCH_BLOCKING"]=="0"and x["environment"]["VLLM_USE_V2_MODEL_RUNNER"]=="1"and "--enforce-eager"not in x["argv"]
files={f.name:f.read_text()for f in(r/"plugin_src").iterdir()if f.is_file()}
install="from pathlib import Path;import sys,json,os;a=json.load(sys.stdin);p=Path(a['path']);assert not p.exists();p.mkdir();rows=[((p/n).write_text(s),os.chmod(p/n,0o600 if n.endswith('.json')else 0o644))for n,s in a['files'].items()];print(json.dumps(dict(installed=True)))"
run("166",["python3","-c",install],"install_native239",input=json.dumps(dict(path=x["argv"][1].rsplit("/",1)[0],files=files)).encode())
for node,id,source_run,name in[("D0","resp_glm_run230_D0_new","GLM-RUN-0230","newD0_create"),("D1","resp_glm_run211_D1_new","GLM-RUN-0211","newD1_create")]:
 base=json.loads((p/"runs"/source_run/"client_final_summary.json").read_text());row=next(x for x in base["requests"]if x["name"]==name);url="http://172.16.10."+("166:9081"if node=="D0"else"167:9900")+"/v1/responses/"+id
 with http.open(url,timeout=15)as response:raw=response.read();assert response.status==200and json.loads(raw)==json.loads(Path(row["wire"]["path"]).read_text())
 (r/("before_"+node+"_native_STORE.wire")).write_bytes(raw)
fault=json.loads((state/"response_owners.json.fault").read_text());assert fault["groups"]["local-228-166"]["faulted"]and not fault["groups"]["local-211-167"]["faulted"]
atomic_json(r/"prepared_state.json",dict(at=utc(),native32_owned_idle=True,D1_retained=True,D0_public_already_faulted=True,native_STORE230D0and211D1_retained=True,native_math_edits=0))
