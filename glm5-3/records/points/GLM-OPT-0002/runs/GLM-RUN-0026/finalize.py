import pathlib,json,hashlib,sys,subprocess
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc
r=pathlib.Path(__file__).parent
for n in ["prepare","identity","capability","pilot"]:assert json.loads((r/(n+".phase.json")).read_text())["status"]=="succeeded"
cap=json.loads((r/"capability/capability_summary.json").read_text());dyn=json.loads((r/"pilot/dynamic/load_result.json").read_text())
assert cap["valid"] and dyn["valid"]
counts=sum(v["successful_inference_requests"] for v in cap["load_results"].values())+dyn["successful_inference_requests"]
outputs=sum(v["effective_output_tokens"] for v in cap["load_results"].values())+dyn["effective_output_tokens"]
assert counts==15 and outputs==10640
for rank,node in [(0,"166"),(1,"167")]:
 name="DP_run25_"+str(rank)+".log";argv=["cat","/data/tiankuan/wio/glm52-pd/deploy/logs/"+name]
 if node=="167":argv=["ssh","-o","BatchMode=yes","root@172.16.10.167","cat /data/tiankuan/wio/glm52-pd/deploy/logs/"+name]
 p=subprocess.run(argv,capture_output=True,timeout=30);p.check_returncode();(r/name).write_bytes(p.stdout)
art=[]
for p in sorted(r.rglob("*")):
 if p.is_file() and p.name not in ["manifest.json","summary.md","state.json","artifact_index.json","controller.log","finalize.log","finalize.phase.json"]:
  raw=p.read_bytes();art.append({"path":str(p),"bytes":len(raw),"sha256":hashlib.sha256(raw).hexdigest()})
atomic_json(r/"artifact_index.json",art)
m=json.loads((r/"manifest.json").read_text());m.update(status="completed",valid=True,verdict="INCONCLUSIVE",completed_at=utc(),new_completed_inference_requests=counts,new_effective_output_tokens=outputs,new_client_attempts=17,expected_rejections=1,cancelled_inference_attempts=1,cancelled_outputs_credit=0,artifact_index={"path":str(r/"artifact_index.json"),"sha256":hashlib.sha256((r/"artifact_index.json").read_bytes()).hexdigest(),"entries":len(art)})
atomic_json(r/"manifest.json",m)
(r/"summary.md").write_text("# "+r.name+"\n\nRetained exact healthy native DP2/TP16/DCP16/EP32 Run25 joint cohort without reload/cold/tools replay. New capability and six dynamic requests valid:15 completed requests/10640 outputs, one native400 and one cancelled attempt excluded from credit. Logical HTTP drain/readd retains both native physical DP ranks. Native implementations unchanged; expert communication, cache topology and scheduler deployment distinguish prior PD windows. No full61440/KEEP/stable capacity.\n")
print(json.dumps({"valid":True,"requests":counts,"outputs":outputs,"dynamic_tps":dyn["effective_tps"]}))
