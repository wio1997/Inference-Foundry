import json,pathlib,sys,hashlib,subprocess
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc
r=pathlib.Path(__file__).parent
paths={"cold":r/"cold_summary.json","tools_DP0":r/"tools_DP0/summary.json","tools_DP1":r/"tools_DP1/summary.json","capability":r/"capability/capability_summary.json","dynamic":r/"pilot/dynamic/load_result.json","prefix":r/"prefix_study/summary.json"}
results={k:json.loads(p.read_text()) for k,p in paths.items()};assert all(v["valid"] for v in results.values())
for name in ["deploy","identity","cold","tools0","tools1","capability","pilot","prefix"]:
 stage=json.loads((r/(name+".phase.json")).read_text());assert stage["status"]=="succeeded" and stage["exit_code"]==0
outputs=results["cold"]["effective_output_tokens"]+results["tools_DP0"]["effective_output_tokens"]+results["tools_DP1"]["effective_output_tokens"]+sum(v["effective_output_tokens"] for v in results["capability"]["load_results"].values())+results["dynamic"]["effective_output_tokens"]+results["prefix"]["effective_output_tokens"]
counts=2+len(results["tools_DP0"]["requests"])+len(results["tools_DP1"]["requests"])+sum(v["successful_inference_requests"] for v in results["capability"]["load_results"].values())+results["dynamic"]["successful_inference_requests"]+results["prefix"]["new_requests"];assert counts==31
for rank,node in [(0,"166"),(1,"167")]:
 name="DP_run28_"+str(rank)+".log";argv=["cat","/data/tiankuan/wio/glm52-pd/deploy/logs/"+name]
 if node=="167":argv=["ssh","-o","BatchMode=yes","root@172.16.10.167","cat /data/tiankuan/wio/glm52-pd/deploy/logs/"+name]
 p=subprocess.run(argv,capture_output=True);p.check_returncode();(r/name).write_bytes(p.stdout)
artifacts=[]
for p in sorted(r.rglob("*")):
 if p.is_file() and p.name not in ["manifest.json","summary.md","state.json","controller.log","finalize.log","finalize.phase.json","artifact_index.json"]:
  raw=p.read_bytes();artifacts.append({"path":str(p),"bytes":len(raw),"sha256":hashlib.sha256(raw).hexdigest()})
idx=r/"artifact_index.json";atomic_json(idx,artifacts)
m=json.loads((r/"manifest.json").read_text());m.update(status="completed",valid=True,verdict="INCONCLUSIVE",completed_at=utc(),new_completed_inference_requests=counts,new_effective_output_tokens=outputs,new_client_attempts=33,expected_rejections=1,cancelled_inference_attempts=1,cancelled_outputs_credit=0,results={k:{"path":str(paths[k]),"sha256":hashlib.sha256(paths[k].read_bytes()).hexdigest(),"valid":v["valid"]} for k,v in results.items()},artifact_index={"path":str(idx),"sha256":hashlib.sha256(idx.read_bytes()).hexdigest(),"entries":len(artifacts)})
atomic_json(r/"manifest.json",m)
(r/"summary.md").write_text("# "+r.name+"\n\nNative two-host coupled DP2/TP16/DCP16/EP32, legal local scheduler default (PD-only recompute flag omitted), hierarchy MC2, fused0/DSA-CPfalse, K5/Graph/budget4096/.87. Exact prior Run25 coupled DP0/DP1 stopped together by identity after both health/idle/source guards; new Worker metadata entrypoint rebuilt only task cohort; PDproxy remains stopped sinceRun24; proxy8000 intentionally unavailable in coupled local mode. Opt-in source-guarded synchronized Graph padding metadata prototype installed on32 workers; two same-body cold81932->2048 controls, native tools on each rank, chat/completion/stream/nonstream/n2/expected400/drain/readd/cancel+idle and six dynamic plus six matchedprefix/tail requests pass. "+str(counts)+" completed new native requests/"+str(outputs)+" valid committed outputs, one400 and one cancelled attempt retained with zero cancellation credit. Native operator implementation unchanged; same EP32/TP16/DCP16 and communication/config asbaseline25-27; only Worker metadata control changes, fresh cache observed; native token trajectories not compared for accuracy. No formal61440/repeated KEEP/stable capacity. Raw source/process/config/response/native logs retained; joint DP cohort requires both ranks alive even if one has no real request.\n")
print(json.dumps({"valid":True,"completed_requests":counts,"outputs":outputs,"dynamic_tps":results["dynamic"]["effective_tps"]}))
