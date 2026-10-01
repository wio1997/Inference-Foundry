import pathlib,json,hashlib,sys
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc
r=pathlib.Path(__file__).parent
data=json.loads((r/"capability/capability_summary.json").read_text());assert data["valid"]
counts=sum(v["successful_inference_requests"]for v in data["load_results"].values());outputs=sum(v["effective_output_tokens"]for v in data["load_results"].values());assert counts==9 and outputs==1552
for name in ["prepare","capability"]:
 p=json.loads((r/(name+".phase.json")).read_text());assert p["status"]=="succeeded" and p["exit_code"]==0
events=[json.loads(x)for x in (r/"capability/router_trace.jsonl").read_text().splitlines()];assert len([x for x in events if x["event"]=="lease_acquired"])==9
contract_events=[x for x in events if x["event"]=="upstream_stream_contract"];assert len(contract_events)==9 and all(not x["audit_error"]for x in contract_events)
rows=[]
for p in r.rglob("*"):
 if p.is_file()and p.name not in ["manifest.json","summary.md","state.json","controller.log","finalize.log","finalize.phase.json","artifact_index.json"]:
  raw=p.read_bytes();rows.append({"path":str(p),"bytes":len(raw),"sha256":hashlib.sha256(raw).hexdigest()})
atomic_json(r/"artifact_index.json",rows)
m=json.loads((r/"manifest.json").read_text());m.update(status="completed",valid=True,verdict="INCONCLUSIVE",completed_at=utc(),new_client_attempts=11,new_completed_inference_requests=9,new_effective_output_tokens=1552,expected_rejections=1,cancelled_inference_attempts=1,cancelled_output_credit=0,artifact_index={"path":str(r/"artifact_index.json"),"sha256":hashlib.sha256((r/"artifact_index.json").read_bytes()).hexdigest()},native_group_reused="exact Run34 owners, no reload")
atomic_json(r/"manifest.json",m)
(r/"summary.md").write_text("# "+r.name+"\n\nOpt-in coupled physicalgroup fault-awaregateway on exact nativeRun34cohort, no modelreload.9 completed native chat/completion/stream/nonstream/n2/drain/readd/arrival/cancel contracts1552outputs; native400 andcancelattempt zero credit. Epoch/source/request/lease/auditwire/nativeidle evidence retained forofflineZcodeaudit. Group native400 stayshealthy, logical drain retainsphysicalranks. Actualnative500failure fromRun32 reusedCPUonly; no newlivefault/recovery/persistent crashproof. No performance or KEEP claim.\n")
print(json.dumps({"valid":True,"completed":counts,"outputs":outputs}))
