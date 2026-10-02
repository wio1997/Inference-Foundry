from pathlib import Path
import json,hashlib,sys
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc
r=Path(__file__).parent;s=json.loads((r/"responses_summary.json").read_text());assert s["valid"]and s["new_inference_attempts"]==7 and s["new_completed_inference_requests"]==6 and s["new_effective_output_tokens"]==192 and s["cancelled_output_credit"]==0
tools=json.loads((r/"tools_TP32/summary.json").read_text());assert tools["valid"]and len(tools["requests"])==4
s["tools_native_output_tokens"]=tools["effective_output_tokens"];s["all_new_inference_attempts"]=11;s["all_new_completed_inference_requests"]=10;s["all_new_effective_output_tokens"]=192+tools["effective_output_tokens"]
for stage in ["deploy","prepare","tools","responses"]:
 phase=json.loads((r/(stage+".phase.json")).read_text());assert phase["status"]=="succeeded"and phase["exit_code"]==0
journal=json.loads((r/"response_owners.json.fault").read_text());assert not journal["open"]and all(not x["faulted"]for x in journal["groups"].values())
paths=[]
for f in sorted(r.rglob("*")):
 if f.is_file()and f.name not in ["manifest.json","state.json","summary.md","controller.log","finalize.log","finalize.phase.json","artifact_index.json"]:
  raw=f.read_bytes();paths.append({"path":str(f),"bytes":len(raw),"sha256":hashlib.sha256(raw).hexdigest()})
atomic_json(r/"artifact_index.json",paths)
m=json.loads((r/"manifest.json").read_text());m.update(status="completed",valid=True,verdict="INCONCLUSIVE",completed_at=utc(),results=s,artifact_index={"path":str(r/"artifact_index.json"),"sha256":hashlib.sha256((r/"artifact_index.json").read_bytes()).hexdigest()});atomic_json(r/"manifest.json",m)
(r/"summary.md").write_text("# "+r.name+"\n\nNative STORE1 Responses owner-affinity finite E2E on new Run46 physical TP32DCP1EP32 native reduce-samplingFalse/STORE1 rollback cohort. Seven newinference attempts/six commits192outputs, one actual native background cancelledzero credit. NativeJSON/customID/previous/SSE/background/retrieve/cancel/starting_after/rawbytepreservation; cleanproxyrestart whileactual nativebackgroundlive0HTTPleases; sameepochlogicaldrain/readd; native random topk/topp/chatlogprobs. HTTPGET polls notextra inference, repeatedretrieve notextraoutputcredit. NativeAPIstate notreplicated, physicalAPIepochrestart recoveryunproved; no performance/KEEP/stablecapacity. Independent fullrawauditpending.\n")
print(json.dumps({"valid":True,"outputs":192,"commit":6,"cancelcredit":0}))
