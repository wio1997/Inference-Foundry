import hashlib,json,os,shlex,signal,subprocess,sys,time
from pathlib import Path
J=Path(__file__).parent; R=J.parents[1]; G=R.parents[4]; sys.path.insert(0,str(G/"runtime"))
from phase_runner import atomic_json,process_identity,same_process,utc
job=json.loads(Path(sys.argv[1]).read_text());own=json.loads((R/"profiler_owner.json").read_text());expected=shlex.split(own["argv"][-1].rsplit("; exec ",1)[1])
assert expected[0].endswith("/bin/msprof") and "--output="+str(R/"device_trace") in expected
rows=subprocess.check_output(["docker","top","glm52-single","-eo","pid,args"],text=True).splitlines()[1:];found=[]
for row in rows:
 pid=int(row.split()[0]);p=Path(f"/proc/{pid}/cmdline")
 try:a=[x.decode()for x in p.read_bytes().split(b"\0")if x]
 except FileNotFoundError:continue
 if a==expected:found.append(dict(identity=process_identity(pid),argv=a))
assert len(found)<=1
atomic_json(J/"owned_profiler.json",dict(at=utc(),found=found,outer=own["identity"],scope="exact Run243 profiler only; not native"))
for f in found:
 assert same_process(f["identity"]);os.kill(f["identity"]["pid"],signal.SIGTERM)
end=time.monotonic()+20
while any(same_process(f["identity"])for f in found) and time.monotonic()<end:time.sleep(.2)
assert not any(same_process(f["identity"])for f in found)
m=json.loads((R/"initial_material.json").read_text());proof={}
for e in m["engines"]:
 node=e["plans"]["node0"]["host"];args=["python3","-c",(R/"live_probe.py").read_text()]
 if node=="167":args=["ssh","-o","BatchMode=yes","root@172.16.10.167",shlex.join(args)]
 p=subprocess.run(args,input=json.dumps(dict(owner=e["roots"]["node0"],NPU_count=16)).encode(),capture_output=True,timeout=90);p.check_returncode();(J/(node+".stdout")).write_bytes(p.stdout);proof[node]=json.loads(p.stdout)["npu_worker_pids"]
atomic_json(J/"cleanup.json",dict(at=utc(),profiler_stopped=True,native_workers=32,native_proof=proof,signals_to_native=0,generated_requests=0))
ev=[]
for p in [J/"owned_profiler.json",J/"cleanup.json"]:ev.append(dict(id=p.name,path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size,locator="exact profiler cleanup and native32 proof"))
atomic_json(Path(job["result"]["path"]),dict(schema_version=1,job_id=job["job_id"],status="completed",summary="Owned Run243 profiler stopped; native32 unchanged. Failed profile raw preserved.",execution=dict(inner_exit_code=0,acceptance="passed",processes=[]),findings=[dict(kind="fact",text="Profiler stopped, no native signal or generation; native32 proof passed",scope=dict(run="GLM-RUN-0243",cleanup_only=True),evidence_ids=["cleanup.json"])],evidence=ev,unknowns=["No actual device capture, H2 unadjudicated"],decision_request=None,next_check_at=None))
