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
 name="DP_run33_"+str(rank)+".log";argv=["cat","/data/tiankuan/wio/glm52-pd/deploy/logs/"+name]
 if node=="167":argv=["ssh","-o","BatchMode=yes","root@172.16.10.167","cat /data/tiankuan/wio/glm52-pd/deploy/logs/"+name]
 p=subprocess.run(argv,capture_output=True);p.check_returncode();(r/name).write_bytes(p.stdout)
artifacts=[]
for p in sorted(r.rglob("*")):
 if p.is_file() and p.name not in ["manifest.json","summary.md","state.json","controller.log","finalize.log","finalize.phase.json","artifact_index.json"]:
  raw=p.read_bytes();artifacts.append({"path":str(p),"bytes":len(raw),"sha256":hashlib.sha256(raw).hexdigest()})
idx=r/"artifact_index.json";atomic_json(idx,artifacts)
m=json.loads((r/"manifest.json").read_text());m.update(status="completed",valid=True,verdict="INCONCLUSIVE",completed_at=utc(),new_completed_inference_requests=counts,new_effective_output_tokens=outputs,new_client_attempts=33,expected_rejections=1,cancelled_inference_attempts=1,cancelled_outputs_credit=0,results={k:{"path":str(paths[k]),"sha256":hashlib.sha256(paths[k].read_bytes()).hexdigest(),"valid":v["valid"]} for k,v in results.items()},artifact_index={"path":str(idx),"sha256":hashlib.sha256(idx.read_bytes()).hexdigest(),"entries":len(artifacts)})
atomic_json(r/"manifest.json",m)
(r/"summary.md").write_text("# "+r.name+"\n\nNative DP2/TP16/DCP16/EP32 K5 FULL_DECODE_ONLY budget16384 gmu.80. Exact jointly cleaned failed Run32 cohort by saved PID boot start argv and task plugin environment, after new full-config CPU guard. Source-guarded metadata predicates unchanged from V1; V2 only enables native logger namespace. No vendor/operator computation edits. Two cold controls, eight tools, nine capability, six dynamic and six matchedprefix/tail requests; all native commit contracts checked, rejected and cancelled attempts zero credit. "+str(counts)+" new completed requests/"+str(outputs)+" committed outputs. Memory, live metadata receipts and latency require raw reduction. No KEEP/stablecapacity from one bounded window.\n")

print(json.dumps({"valid":True,"completed_requests":counts,"outputs":outputs,"dynamic_tps":results["dynamic"]["effective_tps"]}))
