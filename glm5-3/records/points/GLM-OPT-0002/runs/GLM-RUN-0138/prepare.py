from pathlib import Path
import sys,json,subprocess,hashlib,re,urllib.request
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc,same_process
from native_engines_service_config import render
from owner_guard import start_watchdog,guard
start_watchdog();r=Path(__file__).parent;old=r.parent/"GLM-RUN-0137"
s=json.loads((old/"state.json").read_text());assert s["status"]=="failed"and s["failure_phase"]=="pilot"and not same_process(s["owner"])
for pin in json.loads((old/"controller_spec.json").read_text())["stages"][0]["sources"]:
 raw=Path(pin["path"]).read_bytes();assert raw==Path(pin["snapshot"]).read_bytes()and hashlib.sha256(raw).hexdigest()==pin["sha256"]
z=subprocess.run([sys.executable,str(r/"epoch_check.py"),"adopt_initial"],capture_output=True,timeout=240);(r/"adopt_initial.stdout").write_bytes(z.stdout);(r/"adopt_initial.stderr").write_bytes(z.stderr);z.check_returncode()
assert not re.search(r":8000\s",subprocess.check_output(["ss","-ltnp"],text=True))
sem=json.loads((old/"semantic_summary.json").read_text());assert sem["semantic_acceptance"]and sem["semantic_pass_cases"]==sem["semantic_cases"]==6and sem["effective_public_output_tokens"]==22
for row in sem["requests"]:
 assert row["semantic_pass"]and row["status"]=="completed"
 for key in["body","wire"]:
  item=row[key];raw=Path(item["path"]).read_bytes();assert len(raw)==item["bytes"]and hashlib.sha256(raw).hexdigest()==item["sha256"]
 value=json.loads(Path(row["wire"]["path"]).read_text());choice=value["choices"][0];assert choice["message"]["content"].strip()==row["expected"]and choice["finish_reason"]=="stop"and len(choice["token_ids"])==value["usage"]["completion_tokens"]
for name in["native_semantic","native_pilot"]:
 ack=[json.loads(l)for l in(old/(name+".stdout")).read_text().splitlines()if l.startswith('{"event":')];assert[ x["event"]for x in ack]==["task_acl_init","task_acl_finalize"]and all(x["returncode"]==0for x in ack)
pilot=json.loads((old/"pilot_summary.json").read_text());assert not pilot["requests"]and pilot["effective_public_output_tokens"]==0
error=(old/"native_pilot.stderr").read_text();assert"FileNotFoundError"in error and"No such file or directory: 'ssh'"in error
assert not(old/"public_service_proof.json").exists()
atomic_json(r/"reused137_audit.json",dict(at=utc(),run_id=old.name,overall_verdict="INVALID",cause="Client container hasnoSSH; host epochcheck mistakenlyinvokedinsideclient beforepilotRPC",native_both_fit=True,semantic_partial_valid=True,semantic=sem,effective_output_tokens_reused_not_recredited=22,SDK0=True,oldsource82_unchanged=True,fresh_host_epoch=json.loads((r/"adopt_initial.json").read_text()),originalpilot_requests=0,new_inference=0,no_model_operations=True))
config=render(r,r.parent/"GLM-RUN-0125");atomic_json(r/"service_config.json",config)
atomic_json(r/"prepared_state.json",dict(at=utc(),adopted=True,API2=True,NPU32=True,public8000_absent=True,old125_journal_preserved=True,host_container_namespace_separated=True,new_model_operations=0,semantic22_reused=True,limits=["137INVALIDoriginalsourcepreserved; no semanticrerun/nativefitreload; clientnoSSH guardchecksall32onHOSTouterdriver"]))
print("adopted both137native independent engines; sixsemantic reused, clientsshfixture corrected withoutmodelops")
