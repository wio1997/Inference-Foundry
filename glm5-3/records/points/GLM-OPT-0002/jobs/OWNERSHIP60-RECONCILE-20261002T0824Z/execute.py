from pathlib import Path
import json,hashlib,subprocess,shlex,sys
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc,same_process
j=Path(__file__).parent;p=j.parents[1];r=p/"runs/GLM-RUN-0060"
def ref(f):
 b=f.read_bytes();return dict(path=str(f),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
state=json.loads((r/"state.json").read_text());assert state["status"]=="failed"and state["failure_phase"]=="deploy"and not same_process(state["owner"])
pins=json.loads((r/"controller_spec.json").read_text())["stages"][0]["sources"];assert len(pins)==34
for x in pins:assert ref(Path(x["path"]))["sha256"]==x["sha256"]and Path(x["path"]).read_bytes()==Path(x["snapshot"]).read_bytes()
events=json.loads((r/"deployment_events.json").read_text());last=events[-1];assert last["returncode"]==1 and "unattributed nativeworker"in last["stderr_tail"]
assert not any(x["event"]in["native_role_started","exact_P58_stopped_D56_retained"]for x in events)
assert not list(r.glob("CPU_fullCLI*.stdout"))and not(r/"startup_model_identities.json").exists()
prior=json.loads((r/"prior_model_owners.json").read_text());rows={}
source=r/"stop_prior_cohort.py"
for node in ["166","167"]:
 plug=Path("/data/tiankuan/wio/glm52-pd/deploy/plugins/pd_dual_run60")
 if node=="167":
  subprocess.run(["ssh","-o","BatchMode=yes","root@172.16.10.167","mkdir -p "+str(plug)],check=True)
  subprocess.run(["scp","-q",str(source),"root@172.16.10.167:"+str(plug/"stop_prior_cohort.py")],check=True)
  snapshot=r.parent/"GLM-RUN-0057"/("prior_"+node+".owner_preflight.json")
  subprocess.run(["scp","-q",str(snapshot),"root@172.16.10.167:"+str(plug/"prior_members.json")],check=True)
 def cmd(args):
  if node=="167":args=["ssh","-o","BatchMode=yes","root@172.16.10.167",shlex.join(args)]
  return subprocess.run(args,capture_output=True,timeout=100)
 z=cmd(["sha256sum",str(plug/"stop_prior_cohort.py")]);z.check_returncode();assert z.stdout.decode().split()[0]==ref(source)["sha256"]
 args=["env","GLM_EXPECTED_COHORT="+json.dumps([o for o in prior.values()if o["host"]==node]),"GLM_PRIOR_SNAPSHOT="+str(plug/"prior_members.json"),"GLM_CLEANUP_PREFLIGHT_ONLY=1","python3",str(plug/"stop_prior_cohort.py")]
 z=cmd(args);(j/(node+".preflight.stdout")).write_bytes(z.stdout);(j/(node+".preflight.stderr")).write_bytes(z.stderr);z.check_returncode();x=json.loads(z.stdout);rows[node]=dict(raw=ref(j/(node+".preflight.stdout")),P_members=len(x["owned_P_targets"]),D_members=len(x["retained_D_targets"]),NPU_P=x["npu_P"],NPU_D=x["npu_D"],historical_P_zombies=x["historical_P_zombies"])
limits=["Originalguard failure only records 'unattributed nativeworker', no PID/identity snapshot, exacttransientactor ownership/reaper unresolved","Currentreadonly source-bound preflight succeeds, originalP58 APIepochs andD56 retained; not retrospectively proof originalguard false or actorwasP","No nativeCLIfullconfig/newmodels/requests/signals in60, notcandidateGPUfit failure","Foreignworker gate remains; add diagnostic identity before rejection innew61, never blindlysignal orphan/compiler/container/D","SameP416/chunk256/1GiB/GMU.46 candidate actualGPU/new11pilot pending"]
out=dict(at=utc(),run_id=r.name,verdict="INCONCLUSIVE",classification="Pre-cleanup ownership guard rejected unattributed nativeworker beforeCPU/newmodels; later readonly exactprior preflight succeeds, no originalidentity receipt",state=state,pins=len(pins),failing_event=dict(stderr=last["stderr_tail"],stdout=last["stdout_tail"]),current_reconciliation=rows,source=ref(source),signals=0,new_models=0,inference_attempts=0,effective_outputs=0,limits=limits,next_candidate=dict(run_id="GLM-RUN-0061",P_HCCL_MB=416,D_HCCL_MB=512,P_batch=256,D_batch=1024,actual_prior_cohort="P58/D56",diagnostic="print exactunattributed states/identities before assertion; keep strict scope"))
atomic_json(j/"reduction.json",out)
atomic_json(r/"reduction_brief.json",dict(run_id=r.name,at=utc(),verdict="INCONCLUSIVE",measurement_valid=False,functional_acceptance=False,classification=out["classification"],full_reduction=ref(j/"reduction.json"),limits=limits,inference_attempts=0,completed_native_requests=0,effective_public_output_tokens=0))
m=json.loads((r/"manifest.json").read_text());m.update(status="failed",valid=False,verdict="INCONCLUSIVE",ownership_preflight_detail=ref(j/"reduction.json"));atomic_json(r/"manifest.json",m)
(r/"summary.md").write_text("# GLM-RUN-0060\n\nINCONCLUSIVE pre-cleanup ownership: unknownnativeworker triggersstrictguard; originalPID/identityreceiptmissing. 34pins/frozen sourcechecked; nofullCLI/newmodel/request/signals. Laterreadonlysamechecker currentpriorP58/D56 exactpreflight succeeds; currentNPU P"+str(sum(x["NPU_P"]for x in rows.values()))+"/D"+str(sum(x["NPU_D"]for x in rows.values()))+". Not retrospectively falseguard/nativeGPUcandidatefailure. Next61 addsdiagnosticidentities/strictscope, sameP416/chunk256 headroomcandidate withD56retained, nooldqueue.\n")
atomic_json(j/"result.json",dict(schema_version=1,job_id=j.name,status="completed",summary="Run60 INCONCLUSIVE unmappednativeworker pre-cleanup;34pins zero actions, currentexactreadonly priorpreflight succeeds; next61 strictscope diagnostic",execution=dict(inner_exit_code=0,acceptance="passed",processes=[]),findings=[],evidence=[dict(id="reduction",locator="frozenfailure/source34/currentexactAPI-D/NPU membership readonlypreflight",**ref(j/"reduction.json"))],unknowns=limits,decision_request=None,next_check_at=None));print(json.dumps(ref(j/"reduction.json")))

