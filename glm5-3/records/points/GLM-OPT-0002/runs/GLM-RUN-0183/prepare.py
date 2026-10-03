from common import *
old=p/"runs/GLM-RUN-0182";s=json.loads((old/"state.json").read_text());assert s["status"]=="failed" and not same_process(s["owner"])
audit=p/"jobs/AUDIT-FAIL180-RECOVERY181V2-20261003T1437Z/reduction.json";assert ref(audit)["sha256"]=="7458a483048acef621d25acb9d46973b08063687afbf30dcfb2fd11191a6c233"
cpu=p/"jobs/PP4-K1-CPU-20261003T1406Z/reduction.json";assert ref(cpu)["sha256"]=="acbe80296be16a54e905e6f326e92bf5c41b98a6d265f5a8ab5b39144b9b0ca4"
roots=json.loads((r/"standalone_root_identities.json").read_text());members=json.loads((r/"standalone_native_members.json").read_text())
for key,o in roots.items():fresh(o,members[key],"prepare_"+key);native_idle(o["host"],9081 if o["host"]=="166"else 9900,"before")
sdk=[json.loads(l)for l in(old/"client_before.stdout").read_text().splitlines()if l.startswith('{"event":')]
assert [x["event"]for x in sdk]==["task_acl_init","task_acl_finalize"]and all(x["returncode"]==0for x in sdk)
attempts=json.loads((old/"client_before_summary.json").read_text());assert attempts["valid"]is False and attempts["effective_output_tokens"]==0
assert len(attempts["requests"])==3and all(x["method"]=="GET"or x["status"]==503 for x in attempts["requests"])
def sums(f):
 d={}
 for l in Path(f).read_text().splitlines():
  if l and not l.startswith("#"):
   k=l.split("{")[0].split()[0];d[k]=d.get(k,0)+float(l.rsplit(" ",1)[1])
 return d
for node in ["166","167"]:
 before=sums(old/("before_"+node+".metrics"));after=sums(r/("before_"+node+".metrics"))
 assert all(before[k]==after[k]for k in before if k.endswith("_total"))
atomic_json(r/"prior182_prelaunch_failure.json",dict(at=utc(),verdict="INVALID",cause="client top-level base HTTP URL overwritten by stored response row",SDKinit_finalize0=sdk,native32_same=True,native_all_counters_delta0=True,model_operations=0,new_inference_requests=0,plugin182_inert_preserved=True,raw_failure=ref(old/"client_before.stderr")))

proof=json.loads((r/"public_service_proof.json").read_text());assert same_process(proof["host"]) and ref(proof["container"]["config"]["path"])==proof["container"]["config"];atomic_json(r/"old_public_service_proof.json",proof)
assert [v.decode()for v in Path("/proc/"+str(proof["host"]["pid"])+"/cmdline").read_bytes().split(bytes([0]))if v]==proof["argv"]
assert all(not v["group_faulted"]and not v["active_requests"]and not v["draining"]for v in request("/control/replicas")["replicas"])
for node,plugin,serial,threshold in [("166","local_pp175",3,1024),("167","local_pp168",17,4096)]:
 assert json.loads(run(node,["cat",str(site/"plugins"/plugin/"issue_budget_policy.json")],"prepare_policy_"+node))==dict(schema_version=1,cohort_id="GLM-COHORT-0137",budget_tokens=8192,prefill_threshold_tokens=threshold,prefill_cadence=1,serial=serial)
files={f.name:f.read_text()for f in(r/"plugin_src").iterdir()}
install="from pathlib import Path\nimport sys,json,hashlib,os\na=json.load(sys.stdin);p=Path(a['path']);assert not p.exists();p.mkdir();rows=[]\nfor name,s in a['files'].items():\n assert Path(name).name==name\n f=p/name;f.write_text(s);os.chmod(f,0o600 if name.endswith('.json')else 0o644);b=f.read_bytes();rows.append(dict(path=str(f),bytes=len(b),sha256=hashlib.sha256(b).hexdigest()))\nprint(json.dumps(rows))\n"
for node in ["166","167"]:atomic_json(r/("plugin_install_"+node+".json"),json.loads(run(node,["python3","-c",install],"plugin_install_"+node,input=json.dumps(dict(path=str(site/"plugins/local_pp183"),files=files)).encode())))
atomic_json(r/"CPU_geometry_reused.json",dict(at=utc(),evidence=ref(cpu),static_K1_nativeaccepted=True,dynamic_DCP4_guard_preserved=True,not_NPU_fit_or_performance=True))
client("before")
atomic_json(r/"prepared_state.json",dict(at=utc(),native32_same=True,D0_retained=True,new_target="167 D1 staticK1 PP4",model_operations=0))
