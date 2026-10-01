import pathlib,json,hashlib,sys,subprocess
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc
r=pathlib.Path(__file__).parent;d=r/"prefix_study"
for n in ["prepare","identity","prefix"]:assert json.loads((r/(n+".phase.json")).read_text())["status"]=="succeeded"
data=json.loads((d/"summary.json").read_text());assert data["valid"] and data["new_requests"]==6 and data["effective_output_tokens"]==8194
for rank,node in [(0,"166"),(1,"167")]:
 name="DP_run25_"+str(rank)+".log";argv=["cat","/data/tiankuan/wio/glm52-pd/deploy/logs/"+name]
 if node=="167":argv=["ssh","-o","BatchMode=yes","root@172.16.10.167","cat /data/tiankuan/wio/glm52-pd/deploy/logs/"+name]
 p=subprocess.run(argv,capture_output=True,timeout=30);p.check_returncode();(r/name).write_bytes(p.stdout)
art=[]
for p in sorted(r.rglob("*")):
 if p.is_file() and p.name not in ["manifest.json","summary.md","state.json","artifact_index.json","controller.log","finalize.log","finalize.phase.json"]:
  raw=p.read_bytes();art.append({"path":str(p),"bytes":len(raw),"sha256":hashlib.sha256(raw).hexdigest()})
atomic_json(r/"artifact_index.json",art)
m=json.loads((r/"manifest.json").read_text());m.update(status="completed",valid=True,verdict="INCONCLUSIVE",completed_at=utc(),new_completed_inference_requests=6,new_client_attempts=6,new_effective_output_tokens=8194,artifact_index={"path":str(r/"artifact_index.json"),"sha256":hashlib.sha256((r/"artifact_index.json").read_bytes()).hexdigest(),"entries":len(art)})
atomic_json(r/"manifest.json",m)
(r/"summary.md").write_text("# "+r.name+"\n\nRetained exact Run25 coupled native DP2/TP16/DCP16/EP32 both4096/K5/Graph. Reused exact Run23 canonical prefix/tail bodies, six new requests/8194 valid outputs; native counters/strict contracts/idle retained. Same bodies enable conditional prefill/cache analysis, not same-output or isolated code gain. No reload/tools/cold-control replay; this novel prefix has actual observed cache counters. No full61440/KEEP/stable capacity.\n")
print(json.dumps({"requests":6,"outputs":8194,"rows":data["requests"]}))
