import hashlib,json,os,shlex,signal,subprocess,sys,time
from pathlib import Path
J=Path(__file__).parent; R=Path("/data/tiankuan/wio/Inference-Foundry/glm5-3/records/points/GLM-OPT-0002/runs/GLM-RUN-0244"); G=R.parents[4]; sys.path.insert(0,str(G/"runtime"))
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
atomic_json(J/"owned_profiler.json",dict(at=utc(),found=found,outer=own["identity"],scope="exact Run244 profiler only; not native"))
assert not found, "Profiler still active; no signals in read-only audit"
end=time.monotonic()+20
while any(same_process(f["identity"])for f in found) and time.monotonic()<end:time.sleep(.2)
assert not any(same_process(f["identity"])for f in found)
m=json.loads((R/"initial_material.json").read_text());proof={}
for e in m["engines"]:
 node=e["plans"]["node0"]["host"];args=["python3","-c",(R/"live_probe.py").read_text()]
 if node=="167":args=["ssh","-o","BatchMode=yes","root@172.16.10.167",shlex.join(args)]
 p=subprocess.run(args,input=json.dumps(dict(owner=e["roots"]["node0"],NPU_count=16)).encode(),capture_output=True,timeout=90);p.check_returncode();(J/(node+".stdout")).write_bytes(p.stdout);proof[node]=json.loads(p.stdout)["npu_worker_pids"]
atomic_json(J/"cleanup.json",dict(at=utc(),profiler_stopped=True,native_workers=32,native_proof=proof,signals_to_native=0,generated_requests=0))
import urllib.request,re
http=urllib.request.build_opener(urllib.request.ProxyHandler({}))
health={}
for host,port in [("166",9081),("167",9900),("166",8000)]:
 with http.open(f"http://172.16.10.{host}:{port}/"+("healthcheck" if port==8000 else "health"),timeout=10) as response:health[host+":"+str(port)]=response.status
 if port!=8000:
  with http.open(f"http://172.16.10.{host}:{port}/metrics",timeout=10) as response:raw=response.read().decode()
  vals=re.findall(r"^vllm:num_requests_(?:running|waiting)\{[^\n]*\}\s+([0-9.eE+-]+)",raw,re.M)
  assert vals and all(float(x)==0 for x in vals)
public=json.loads((R/"initial_public_owner.json").read_text());assert same_process(public["identity"])
store_equal={}
for rid in ["resp_glm_run241_base","resp_glm_run241_bg32"]:
 with http.open("http://172.16.10.167:9900/v1/responses/"+rid,timeout=10) as response:raw=response.read()
 (J/(rid+".wire")).write_bytes(raw);store_equal[rid]=json.loads(raw)==json.loads((R/("store_before_"+rid+".wire")).read_text());assert store_equal[rid]
atomic_json(J/"final_site.json",dict(at=utc(),health=health,idle=True,native_workers=32,native_proof=proof,public_owner=public,D1_STORE_equal=store_equal,signals_to_native=0,generated_requests=0,Current=None))
controllers={}
for rn in ["GLM-RUN-0242","GLM-RUN-0243","GLM-RUN-0244"]:
 p=R.parent/rn/"state.json";d=json.loads(p.read_text());assert not same_process(d["owner"])
 controllers[rn]=dict(status=d["status"],process_active=False,spec_sha256=d["spec_sha256"])
atomic_json(J/"controllers.json",dict(at=utc(),controllers=controllers,queued_execution_replayed=False))
ev=[]
for p in [J/"owned_profiler.json",J/"cleanup.json",J/"final_site.json",J/"controllers.json"]:ev.append(dict(id=p.name,path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size,locator="exact profiler cleanup and native32 proof"))
atomic_json(Path(job["result"]["path"]),dict(schema_version=1,job_id=job["job_id"],status="completed",summary="Run244 profiler absent; native32 unchanged. Fresh native32/health/idle/public/D1 STORE valid;242/243/244 controllers inactive.",execution=dict(inner_exit_code=0,acceptance="passed",processes=[]),findings=[dict(kind="fact",text="Profiler absent, no signals or generation; native32 proof passed",scope=dict(run="GLM-RUN-0244",cleanup_only=True),evidence_ids=["cleanup.json"])],evidence=ev,unknowns=["No matched product E2E baseline/gain; H2 remains INCONCLUSIVE/PARKED"],decision_request=None,next_check_at=None))
