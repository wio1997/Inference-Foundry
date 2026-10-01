import pathlib,json,hashlib,subprocess,sys
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc
r=pathlib.Path(__file__).parent
names={"prefix":r/"prefix_study/summary.json","PD":r/"pd_compat/load_result.json","dynamic":r/"pilot/dynamic/load_result.json"}
results={k:json.loads(p.read_text()) for k,p in names.items()}
assert all(x["valid"] for x in results.values())
for n in ["prefix","pd","pilot"]:
 p=json.loads((r/(n+".phase.json")).read_text());assert p["status"]=="succeeded" and p["exit_code"]==0
for node,name in [("166","P_166.log"),("167","D_167.log")]:
 argv=["cat","/data/tiankuan/wio/glm52-pd/deploy/logs/"+name]
 if node=="167":argv=["ssh","-o","BatchMode=yes","root@172.16.10.167","cat /data/tiankuan/wio/glm52-pd/deploy/logs/"+name]
 p=subprocess.run(argv,capture_output=True);assert p.returncode==0;(r/name).write_bytes(p.stdout)
artifacts=[]
for p in sorted(r.rglob("*")):
 if p.is_file() and p.name not in ["manifest.json","state.json","controller.log","finalize.log","finalize.phase.json","artifact_index.json"]:
  artifacts.append({"path":str(p),"bytes":p.stat().st_size,"sha256":hashlib.sha256(p.read_bytes()).hexdigest()})
idx=r/"artifact_index.json";atomic_json(idx,artifacts)
m=json.loads((r/"manifest.json").read_text());m.update(status="completed",verdict="INCONCLUSIVE",valid=True,completed_at=utc(),new_inference_requests=13,new_effective_output_tokens=17538,results={k:{key:value for key,value in x.items() if key!="requests"} for k,x in results.items()},artifact_index={"path":str(idx),"bytes":idx.stat().st_size,"sha256":hashlib.sha256(idx.read_bytes()).hexdigest(),"entries":len(artifacts)})
atomic_json(r/"manifest.json",m)
(r/"summary.md").write_text("# "+r.name+"\n\nP21 K5/batch4096 and D22 K5/batch8192 retained without restart. Six paired prefix/tail requests, PD81932→256 and six dynamic requests valid; 13 new client inferences/17538 effective outputs. Run22 cold81932→2048 reused by reference, not counted anew. Raw native usage/length/DONE/idle and source/config identities retained. Native operator implementation unchanged; scheduler budget/KV/deployment costs differ. Prefix pairs use new datasetseed20261002 with identical canonical bodies per pair, requestseed omitted/native1024; cache observed without reset. Diagnostic output2048 is not full61440 formal/SLA or stable capacity. API/cancel/n2 evidence reused from identical native parser/runtime in Run20/21; not rerun or counted anew. INCONCLUSIVE; native memory and conditional prefix cost await reduction.\n")
print(json.dumps({"run":r.name,"valid":True,"outputs":17538,"dynamic_tps":results["dynamic"]["effective_tps"]}))
