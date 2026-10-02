from pathlib import Path
import json,hashlib,sys,subprocess,shlex
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,same_process,utc
j=Path(__file__).parent;p=j.parents[1];r=p/"runs/GLM-RUN-0053";s=json.loads((r/"state.json").read_text());assert s["status"]=="failed"and not same_process(s["owner"])
def ref(f):
 b=f.read_bytes();return dict(path=str(f),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
spec=json.loads((r/"controller_spec.json").read_text());pins=spec["stages"][0]["sources"];assert len(pins)==20
for z in pins:assert Path(z["path"]).read_bytes()==Path(z["snapshot"]).read_bytes()and hashlib.sha256(Path(z["path"]).read_bytes()).hexdigest()==z["sha256"]
ev=json.loads((r/"deployment_events.json").read_text());assert not any(z["event"]in["native_role_started","exact_joint52_cohort_removed"]for z in ev)
cmds=[z for z in ev if z["event"]=="command"];cleanup=[z for z in cmds if"GLM_EXPECTED_COHORT"in" ".join(z["argv"])]
assert cleanup and all("GLM_CLEANUP_PREFLIGHT_ONLY=1"in z["argv"]for z in cleanup)
assert any("/data/tiankuan/wio/glm53-pd/deploy/plugins/pd_dual_run53/glm_tool_contract.py"in z.get("stdout_tail","")for z in cmds)
assert not any((r/n).exists()for n in["startup_model_identities.json","native_attempts.jsonl"])
old=json.loads((r/"prior_model_owners.json").read_text());actual={}
for node in["166","167"]:
 code="from pathlib import Path;import json;owners="+repr([z for z in old.values()if z["host"]==node])+";out=[]\nfor z in owners:\n p=Path('/proc')/str(z['pid']);live=p.exists();a=None\n if live:\n  st=(p/'stat').read_text();a=dict(boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),start_ticks=st[st.rfind(')')+2:].split()[19],argv=[v.decode()for v in(p/'cmdline').read_bytes().split(bytes([0]))if v]);assert a['start_ticks']==z['identity']['start_ticks']and a['boot_id']==z['identity']['boot_id']and a['argv']==z['argv']\n out.append(dict(role=z['role'],rank=z['rank'],pid=z['pid'],same_original_owner=live,actual=a))\nprint(json.dumps(out))"
 argv=["python3","-c",code]
 if node=="167":argv=["ssh","-o","BatchMode=yes","root@172.16.10.167",shlex.join(argv)]
 actual[node]=json.loads(subprocess.check_output(argv))
out=dict(at=utc(),run_id=r.name,classification="GPT planned argv accidentally replaced fixed glm52-pd site path",verdict="INVALID",state=s,pins=20,inference_attempts=0,new_models=0,new_native_roles=0,task_signals=0,old52_actual_owners=actual,source_spec=ref(r/"controller_spec.json"),deployment_events=ref(r/"deployment_events.json"),failure_command=cmds[-1],limits=["Source/CPU unusedSFAworkspace proof unaffected; no actualNPUworkspace/Graph/PD experiment occurred","Only166 readonly priorowner preflight ran; old52cleanup neverreached","RealZcode launchVALID means controllerlaunched only; nativeCLI preflight exposed badpath"])
atomic_json(j/"reduction.json",out);atomic_json(r/"reduction_brief.json",dict(out,measurement_valid=True));m=json.loads((r/"manifest.json").read_text());m.update(status="failed",valid=False,verdict="INVALID",config_failure_detail=ref(j/"reduction.json"));atomic_json(r/"manifest.json",m)
(r/"summary.md").write_text("# GLM-RUN-0053\n\nINVALID GPT plannedargv fixedsitepath glm52-pd accidentallyrenamedglm53-pd, missingtoolplugin/CLIAPIreject BEFORE old52cleanup/newmodels/requests. 20sourcepins valid,0task signals/0newroles/0inference. Source/CPUunusedSFAworkspace proof unaffected; no GPU/Graph/workspaceverdict. New54 fieldwise engineIDs/pluginversion retains fixedsitepath, same actual52 priorowners. Raw/spec/code frozen. CurrentNone.\n")
atomic_json(j/"result.json",dict(schema_version=1,job_id=j.name,status="completed",summary="Run53 INVALID plannedargv wrongfixedsitepath; nativeCLI rejected beforeprior52cleanup/anynewmodel/requests/signals;20pins/sourceproof unaffected",execution=dict(inner_exit_code=0,acceptance="passed",processes=[]),findings=[],evidence=[dict(id="reduction",locator="20frozenpins/commandfatal/beforecleanup/actualold52owners",**ref(j/"reduction.json"))],unknowns=out["limits"],decision_request=None,next_check_at=None));print(json.dumps(dict(verdict="INVALID",reduction=ref(j/"reduction.json"))))

