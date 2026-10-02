
import json,pathlib,sys,hashlib,subprocess
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc
r=pathlib.Path(__file__).parent
paths={"cold":r/"cold_summary.json","tools":r/"tools_TP32/summary.json","capability":r/"capability/capability_summary.json","dynamic":r/"pilot/dynamic/load_result.json","prefix":r/"prefix_study/summary.json"}
results={k:json.loads(p.read_text())for k,p in paths.items()};assert all(v["valid"]for v in results.values())
for name in ["deploy","identity","cold","tools","capability","prefix","pilot"]:
 p=json.loads((r/(name+".phase.json")).read_text());assert p["status"]=="succeeded"and p["exit_code"]==0
cap=results["capability"]["load_results"]
counts=1+len(results["tools"]["requests"])+sum(v["successful_inference_requests"]for v in cap.values())+results["dynamic"]["successful_inference_requests"]+results["prefix"]["new_requests"]
outputs=2048+results["tools"]["effective_output_tokens"]+sum(v["effective_output_tokens"]for v in cap.values())+9088+4097
assert counts==20 and sum(v["effective_output_tokens"]for v in cap.values())==1264
for rank,node in [(0,"166"),(1,"167")]:
 p=["cat","/data/tiankuan/wio/glm52-pd/deploy/logs/TP_run44_"+str(rank)+".log"]
 if node=="167":p=["ssh","-o","BatchMode=yes","root@172.16.10.167",__import__("shlex").join(p)]
 x=subprocess.run(p,capture_output=True);x.check_returncode();(r/("TP_run44_"+str(rank)+".log")).write_bytes(x.stdout)
artifacts=[]
for p in r.rglob("*"):
 if p.is_file()and p.name not in ["manifest.json","summary.md","state.json","controller.log","finalize.log","finalize.phase.json","artifact_index.json"]:
  raw=p.read_bytes();artifacts.append({"path":str(p),"bytes":len(raw),"sha256":hashlib.sha256(raw).hexdigest()})
atomic_json(r/"artifact_index.json",artifacts)
m=json.loads((r/"manifest.json").read_text());m.update(status="completed",valid=True,verdict="INCONCLUSIVE",completed_at=utc(),new_client_attempts=22,new_completed_inference_requests=counts,new_effective_output_tokens=outputs,expected_rejections=1,cancelled_inference_attempts=1,cancelled_output_credit=0,artifact_index={"path":str(r/"artifact_index.json"),"sha256":hashlib.sha256((r/"artifact_index.json").read_bytes()).hexdigest()});atomic_json(r/"manifest.json",m)
(r/"summary.md").write_text("# "+r.name+"\n\nNative single DP1/TP32/DCP1/EP32 scheduler across both physical hosts, node1 headless.20completed/"+str(outputs)+"outputs of22 attempts;400/cancel zero credit. One cold control,4tool contracts,6gateway capability positives,3canonical matched prefix/tails,6dynamic arrivals; actual native commit/wire/cache/lease/idle logs retained. Same model/operator/K5/Graph/budget16384/.80, physical/scheduler topology changed. No isolated performance gain/KEEP/stablecapacity before independent raw audit.\n")
print(json.dumps({"completed":counts,"outputs":outputs,"dynamic_tps":results["dynamic"]["effective_tps"]}))
