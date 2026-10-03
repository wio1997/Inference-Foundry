from pathlib import Path
import json,sys,subprocess,hashlib,re,shlex
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,same_process,utc
j=Path(__file__).parent;p=j.parents[1];r=p/"runs/GLM-RUN-0124";g=p.parents[2]
s=json.loads((r/"state.json").read_text());assert s["status"]=="failed"and not same_process(s["owner"])
assert json.loads((r/"prepare.phase.json").read_text())["status"]=="succeeded"
assert json.loads((r/"fullapi.phase.json").read_text())["exit_code"]==1
for v in json.loads((r/"controller_spec.json").read_text())["stages"][0]["sources"]:
 assert Path(v["path"]).read_bytes()==Path(v["snapshot"]).read_bytes() and hashlib.sha256(Path(v["path"]).read_bytes()).hexdigest()==v["sha256"]
assert "ModuleNotFoundError: No module named 'acl'"in(r/"first.gateway.log").read_text()
assert not(r/"attempts.json").exists()and not(r/"public_service_owner.json").exists()
ports=subprocess.check_output(["ss","-ltnp"],text=True);assert not re.search(r":(?:8000|8002)\s",ports)
z=subprocess.run([sys.executable,str(j/"epoch_check.py"),"failure124"],capture_output=True,timeout=240);(j/"epochs.stdout").write_bytes(z.stdout);(j/"epochs.stderr").write_bytes(z.stderr);z.check_returncode()
def metric(f):
 o={}
 for line in f.read_text().splitlines():
  if line.startswith("vllm:")and any(k in line for k in["generation_tokens_total","prompt_tokens_total","request_success_total"]):
   name=line.split("{")[0].split()[0];o[name]=o.get(name,0)+float(line.rsplit(" ",1)[1])
 return o
deltas={}
for key in["D0","D1"]:
 a=metric(r/("initial_0_"+key+".metrics"));b=metric(j/("failure124_"+key+".metrics"));deltas[key]={k:b[k]-v for k,v in a.items()};assert all(v==0 for v in deltas[key].values())
shell="export PD_LOCAL_IP=172.16.10.166 PD_NIC=business; source /data/tiankuan/wio/glm52-pd/deploy/scripts/pd_common_env.sh; export PYTHONPATH="+str(g/"runtime")+":$PYTHONPATH; python3 "+shlex.quote(str(j/"child_probe.py"))+" "+shlex.quote(str(j))
argv=["docker","exec","glm52-single","bash","-c",shell];atomic_json(j/"command.json",dict(argv=argv,models=0,inference=0))
z=subprocess.run(argv,capture_output=True,timeout=120);(j/"CPU.stdout").write_bytes(z.stdout);(j/"CPU.stderr").write_bytes(z.stderr);z.check_returncode()
proof=json.loads((j/"child_proof.json").read_text());assert proof["SDK_init"]==proof["SDK_finalize"]==0and proof["child_exit"]==0
out=dict(at=utc(),run124_verdict="INVALID",cause="launch fixture overwrote CANN PYTHONPATH; acl import failed before public listener/inference",native_counter_deltas=deltas,API2NPU32_same=True,port8000_free=True,native_requests=0,model_operations=0,corrected_launch_CPU=proof,limits=["CPU actual child ACL lifecycle plus serviceentry CLIhelp, not publiclistener/fullnativeAPI proof"])
atomic_json(j/"reduction.json",out);atomic_json(r/"reduction_brief.json",out)
m=json.loads((r/"manifest.json").read_text());m.update(status="failed",valid=False,verdict="INVALID",results=out);atomic_json(r/"manifest.json",m)
(r/"summary.md").write_text("# GLM-RUN-0124\n\nINVALID: launch fixture overwrote CANN PYTHONPATH; child acl import failed before listener/native requests. Prepare passed; models unchanged and idle. Preserve original source/logs; fixed child launch SDK lifecycle+CLIhelp verified in "+j.name+". No capacity credit.\n")
b=(j/"reduction.json").read_bytes();atomic_json(j/"result.json",dict(schema_version=1,job_id=j.name,status="completed",summary="Run124 fixtureINVALID zeroinference/API2NPU32same; corrected actualchild CPU SDKinit/final0 +entryCLIhelp validated",execution=dict(inner_exit_code=0,acceptance="passed",processes=[]),findings=[],evidence=[dict(id="reduction",path=str(j/"reduction.json"),bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),locator="originalfailedacl/zero native counters/currentepochs/actualchildSDK0")],unknowns=out["limits"],decision_request=None,next_check_at=None))
print(json.dumps(out))
