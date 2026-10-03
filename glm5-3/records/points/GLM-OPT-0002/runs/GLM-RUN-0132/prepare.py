from pathlib import Path
import json,sys,subprocess,hashlib,re
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc,same_process
from standalone_service_config import render
from owner_guard import start_watchdog,guard
start_watchdog();r=Path(__file__).parent;old=r.parent/"GLM-RUN-0131"
state=json.loads((old/"state.json").read_text());assert state["status"]=="failed"and not same_process(state["owner"])and state["failure_phase"]=="pilot"
pins=json.loads((old/"controller_spec.json").read_text())["stages"][0]["sources"]
for x in pins:
 assert Path(x["path"]).read_bytes()==Path(x["snapshot"]).read_bytes()and hashlib.sha256(Path(x["path"]).read_bytes()).hexdigest()==x["sha256"]
z=subprocess.run([sys.executable,str(r/"epoch_check.py"),"adopt_initial"],capture_output=True,timeout=240);(r/"adopt_initial.stdout").write_bytes(z.stdout);(r/"adopt_initial.stderr").write_bytes(z.stderr);z.check_returncode()
assert not re.search(r":8000\s",subprocess.check_output(["ss","-ltnp"],text=True))
owners=json.loads((r/"adopted_model_identities.json").read_text());client={}
for k,o in owners.items():
 assert o["host"]=="166"
 status=Path("/proc/"+str(o["pid"])+"/status").read_text()
 ids=[int(v)for v in next(l for l in status.splitlines()if l.startswith("NSpid:")).split()[1:]]
 assert len(ids)==2 and ids[0]==o["pid"]
 client[k]=dict(o,pid=ids[1]);atomic_json(r/"api_namespace_mapping.json",dict(host=o,host_pid=ids[0],container_pid=ids[1],source="host/proc/status/NSpid",container="glm52-single"))
 code="from pathlib import Path;import json;p=Path('/proc/"+str(ids[1])+"');s=(p/'stat').read_text();print(json.dumps(dict(boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),start_ticks=s[s.rfind(')')+2:].split()[19],argv=[v.decode()for v in(p/'cmdline').read_bytes().split(bytes([0]))if v])))"
 z=subprocess.run(["docker","exec","glm52-single","python3","-c",code],capture_output=True,timeout=20);z.check_returncode();v=json.loads(z.stdout);assert {k:v[k]for k in ["boot_id","start_ticks"]}==o["identity"]and v["argv"]==o["argv"]
atomic_json(r/"adopted_client_namespace_identities.json",client)
semantic=json.loads((old/"semantic_summary.json").read_text());rows=semantic["requests"];assert len(rows)==3and [x["content"]for x in rows]==["4","42","GLM_OK_731"]and all(x["semantic_pass"]and x["status"]=="completed"for x in rows)
for row in rows:
 for k in ["body","wire"]:
  item=row[k];raw=Path(item["path"]).read_bytes();assert len(raw)==item["bytes"]and hashlib.sha256(raw).hexdigest()==item["sha256"]
acks=[json.loads(l)for l in(old/"native_semantic.stdout").read_text().splitlines()if l.startswith('{"event":')]
assert [x["event"]for x in acks]==["task_acl_init","task_acl_finalize"]and all(x["returncode"]==0for x in acks)
error=(old/"semantic_final_node0.stderr").read_text();assert "FileNotFoundError" in error and "No such file or directory: 'docker'" in error
assert not(old/"pilot_summary.json").exists()and not(old/"public_service_proof.json").exists()
atomic_json(r/"reused131_audit.json",dict(at=utc(),run_id=old.name,overall_verdict="INVALID",cause="host epoch_probe invoked inside client container; docker absent",native_engine_fit=True,semantic_partial_valid=True,semantic_reused=rows,effective_output_tokens=11,SDK0=True,source84_unchanged=True,fresh_host_epoch=json.loads((r/"adopt_initial.json").read_text()),new_inference=0,no_model_operations=True,limits=["Run131 overall planned contract incomplete; partial semantics reused without reexecution","No native quality proof or capacity acceptance"]))
config=render(r,r.parent/"GLM-RUN-0125");atomic_json(r/"service_config.json",config)
atomic_json(r/"prepared_state.json",dict(at=utc(),native_cohort="GLM-COHORT-0131",adopted=True,API1=True,NPU32=True,public8000_absent=True,old125_journal_preserved=True,host_container_namespace_separated=True))
print("adopted coupled131 native resident engine; host roots32 checked and client API namespace mapped")
