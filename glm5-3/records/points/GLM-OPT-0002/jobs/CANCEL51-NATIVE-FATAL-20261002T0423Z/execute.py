from pathlib import Path
import sys,json,hashlib,os,signal,time,subprocess,shlex
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import same_process,atomic_json,utc
j=Path(__file__).parent;p=j.parents[1];r=p/"runs/GLM-RUN-0051";s=json.loads((r/"state.json").read_text());assert s["status"]=="running"and s["active_stage"]=="deploy"and s["spec_sha256"]=="c5bbb67f9db75bd37ff3ba8a3bab00ab3420ce6f8b9aa0658bb0c65244d48cfa"and same_process(s["owner"])
owner=json.loads(Path("/data/tiankuan/wio/glm52-pd/controller-owner.json").read_text());assert owner["owner"]==s["owner"]and owner["run_id"]==r.name
pid=s["owner"]["pid"];argv=[x.decode()for x in Path("/proc/"+str(pid)+"/cmdline").read_bytes().split(bytes([0]))if x];assert argv==["/usr/bin/python3","/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime/controller.py","work",str(r/"controller_spec.json")]
logs={}
for node,rank in[("166",0),("167",1)]:
 args=["cat","/data/tiankuan/wio/glm52-pd/deploy/logs/D_run51_"+str(rank)+".log"]
 if node=="167":args=["ssh","-o","BatchMode=yes","root@172.16.10.167",shlex.join(args)]
 z=subprocess.run(args,capture_output=True,timeout=60);z.check_returncode();f=j/("D"+str(rank)+".fatal_before_cancel.log");f.write_bytes(z.stdout);txt=z.stdout.decode(errors="replace")
 assert "No valid cudagraph sizes after rounding to multiple of 6"in txt and "max_cudagraph_capture_size (4)"in txt and "cudagraph_capture_sizes ([1, 2, 4])"in txt
 assert not(r/"attempts.json").exists()
 logs[node]=dict(path=str(f),bytes=len(z.stdout),sha256=hashlib.sha256(z.stdout).hexdigest(),first_errors=[l for l in txt.splitlines()if"ValueError: No valid cudagraph sizes"in l][:2])
assert same_process(s["owner"])
atomic_json(j/"pre_signal.json",dict(at=utc(),state=s,controller_owner=owner,argv=argv,fatal=logs,reason="Frozen51readiness fatal list lacks ValueError; actual bothDnative geometry fatal preKVtensor. Cancel uniquecontroller via its native handler; no API/NPU signals here"))
os.kill(pid,signal.SIGTERM)
for _ in range(120):
 t=json.loads((r/"state.json").read_text())
 if t["status"]=="cancelled"and not same_process(s["owner"]):break
 time.sleep(.5)
else:raise RuntimeError("controller cancellation not confirmed; no further signals")
atomic_json(j/"reduction.json",dict(at=utc(),before=s,after=t,signal="SIGTERM unique controller only",signals=1,models_started=0,inference=0,native_failure="K5 FULL Graph defaultcapture [1,2,4]/cap4 -> no multiple6",failure_order="initialize_kv_cache/initialize_attn_backend before initialize_kv_cache_tensors; no physicalKVfit conclusion",logs=logs,limits=["Cancelled controller stage exact child only; native API roots remain for nextsinglecontroller guardedcleanup","No oldqueue/no source or raw mutation"]))
b=(j/"reduction.json").read_bytes()
atomic_json(j/"result.json",dict(schema_version=1,job_id=j.name,status="completed",summary="Run51 nativeD K5Graph geometry fatal captured bothhosts; uniquecontroller gracefulcancel confirmed,1taskSIGTERM/0requests/0modelstarts",execution=dict(inner_exit_code=0,acceptance="passed",processes=[]),findings=[],evidence=[dict(id="reduction",path=str(j/"reduction.json"),bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),locator="exactcontroller/actualbothDnativefatal/preandpostcancel")],unknowns=["Actual KVtensor allocation/dualfit remainsunproved; API/NPU owners retainedforguardednextcontrollercleanup"],decision_request=None,next_check_at=None))
print(json.dumps(t))

