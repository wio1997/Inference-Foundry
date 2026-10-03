from pathlib import Path
import json,sys,hashlib,subprocess,os,copy
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from native_identity_observer import observe,host_probe,apply_observation
from phase_runner import atomic_json,utc,process_identity
j=Path(__file__).parent;p=j.parents[1]
def ref(f):
 b=f.read_bytes();return dict(path=str(f),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
pins=json.loads((j/"source_pins.json").read_text())
for pin in pins:assert Path(pin["path"]).read_bytes()==Path(pin["snapshot"]).read_bytes()and ref(Path(pin["path"]))["sha256"]==pin["sha256"]
proof=p/"jobs/NATIVE-FAULT-CPU3-20261003T0443Z/reduction.json";assert ref(proof)["sha256"]=="178ddad3d34d22b275df09b136dcb1a00617215ac5f91396af5d5d7b7b3bbb70"
config=p/"runs/GLM-RUN-0147/service_config.json";before=observe(config);assert len(before["groups"])==2and all(x["status"]=="healthy"for x in before["groups"])
assert sum(h["expected_native_workers"]for x in before["groups"]for h in x["hosts"])==32
atomic_json(j/"native_before.json",before);contracts=["actual_host_both_native_roots_32_worker_identities_and_ancestry_alive"]
owner=json.loads((p/"runs/GLM-RUN-0147/public_service_host.json").read_text());posts=[]
def post(url,body):
 posts.append(dict(url=url,body=body));return dict(status=200,body=dict(faulted=True))
assert apply_observation(before,owner,"http://CPU-mock",post)==[]and not posts
contracts.append("healthy_observation_no_quarantine_posts")
unknown=observe(config,probe=lambda host,root,workers:dict(status="unknown",error="CPU injected SSH uncertainty"))
assert all(x["status"]=="unknown"for x in unknown["groups"])and apply_observation(unknown,owner,"http://CPU-mock",post)==[]and not posts
contracts.append("SSH_uncertainty_never_becomes_confirmed_identity_fault")
def fault_probe(host,root,workers):
 return dict(status="fault"if host=="167"else"healthy",fixture=True,reason="CPU injected identity loss")
fault=observe(config,probe=fault_probe);actions=apply_observation(fault,owner,"http://CPU-mock",post);assert len(actions)==len(posts)==1and posts[0]["url"].endswith("/local-167/fault")
assert posts[0]["body"]["epoch"]==next(x["epoch"]for x in before["groups"]if x["id"]=="local-167")
contracts.append("one_confirmed_fault_targets_only_recorded_epoch_no_nativeHTTP")
lost=copy.deepcopy(owner);lost["identity"]["start_ticks"]="1";n=len(posts)
try:apply_observation(fault,lost,"http://CPU-mock",post)
except RuntimeError:pass
else:raise AssertionError("deadpublicowner appliedfault")
assert len(posts)==n;contracts.append("ended_public_owner_prevents_mutation")
# Actual task-created CPU child exercises host /proc detection; never signal native processes.
ident=process_identity(os.getpid());root=dict(pid=os.getpid(),identity={k:ident[k]for k in["boot_id","start_ticks"]},argv=[v.decode()for v in Path("/proc/self/cmdline").read_bytes().split(bytes([0]))if v])
child=subprocess.Popen([sys.executable,"-c","import time;time.sleep(60)"],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
try:
 ci=process_identity(child.pid);worker=dict(pid=child.pid,identity={k:ci[k]for k in["boot_id","start_ticks"]})
 a=host_probe("166",root,[worker]);assert a["status"]=="healthy"and len(a["rows"])==2;contracts.append("actual_CPU_child_identity_and_ancestry_detected")
 bad=copy.deepcopy(root);bad["identity"]["start_ticks"]="1";assert host_probe("166",bad,[worker])["status"]=="fault";contracts.append("actual_host_start_identity_mismatch_fault")
 bad=copy.deepcopy(root);bad["argv"]=["CPU mismatch"];assert host_probe("166",bad,[worker])["status"]=="fault";contracts.append("actual_host_root_argv_mismatch_fault")
 child.terminate();child.wait(timeout=5)
 a=host_probe("166",root,[worker]);assert a["status"]=="fault"and any(x.get("cause")=="pid_missing"for x in a["rows"]);contracts.append("actual_owned_CPU_child_exit_detected_no_native_signals")
finally:
 if child.poll()is None:child.terminate();child.wait(timeout=5)
after=observe(config);assert all(x["status"]=="healthy"for x in after["groups"])and [{k:x[k]for k in["id","epoch","members"]}for x in before["groups"]]==[{k:x[k]for k in["id","epoch","members"]}for x in after["groups"]]
contracts.append("actual_native32_identity_domains_unchanged_after_CPU_fixture")
atomic_json(j/"native_after.json",after)
out=dict(at=utc(),valid=True,contracts=contracts,source_pins=pins,actual_native_before=ref(j/"native_before.json"),actual_native_after=ref(j/"native_after.json"),CPU_mock_fault_posts=posts,native_quarantine_posts=0,native_requests=0,native_signals=0,NPU_workers_started=0,CPU_owned_fixture_child_terminated=True,limits=["Native HOST roots/recorded32workers/ancestry read-only, no NPUoccupancy/health/inference/physicalnativefault proof","Faultselection andPOST mocked; oneactualCPUchild exit only, real nativefault/E2E stillpending","No SDK operations required for HOST identity-only observer; model/frontends/nativeepochs unchanged"])
atomic_json(j/"reduction.json",out);b=(j/"reduction.json").read_bytes()
atomic_json(j/"result.json",dict(schema_version=1,job_id=j.name,status="completed",summary="HOST observer actualrecordednative32 identityreadonly +10CPUcontracts/SSHunknown/no-nativefaultPOSTs/ownedCPUchildexit detection; physicalnativefault pending",execution=dict(inner_exit_code=0,acceptance="passed",processes=[]),findings=[],evidence=[dict(id="reduction",**ref(j/"reduction.json"),locator="actualHOST nativeidentityread-only/CPUchild/mockquarantine selection")],unknowns=out["limits"],decision_request=None,next_check_at=None))
print(json.dumps(dict(contracts=len(contracts),native32_identity=True,native_requests=0,native_signals=0)))
