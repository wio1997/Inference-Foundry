import json,hashlib,sys
from pathlib import Path
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc
r=Path(__file__).parent;cap=json.loads((r/"capability/capability_summary.json").read_text());restart=json.loads((r/"restart/summary.json").read_text())
assert cap["valid"] and restart["valid"]
for phase in ["prepare","capability","restart"]:assert json.loads((r/(phase+".phase.json")).read_text())["status"]=="succeeded"
counts=sum(x["successful_inference_requests"]for x in cap["load_results"].values())+restart["new_completed_inference_requests"];assert counts==11
tokens=sum(x["effective_output_tokens"]for x in cap["load_results"].values())+restart["new_effective_output_tokens"]
journal=json.loads((r/"fault_state.json").read_text());assert not journal["open"] and all(not x["faulted"]for x in journal["groups"].values())
exec(compile((r/"adopt_models.py").read_text(),str(r/"adopt_models.py"),"exec"))
old=r.parent/"GLM-RUN-0038";assert json.loads((r/"adopted_model_identities.json").read_text())==json.loads((old/"adopted_model_identities.json").read_text())
artifacts=[]
for p in sorted(r.rglob("*")):
 if p.is_file()and p.name not in ["manifest.json","summary.md","state.json","controller.log","finalize.log","finalize.phase.json","artifact_index.json"]:
  raw=p.read_bytes();artifacts.append({"path":str(p),"bytes":len(raw),"sha256":hashlib.sha256(raw).hexdigest()})
atomic_json(r/"artifact_index.json",artifacts)
m=json.loads((r/"manifest.json").read_text());m.update(status="completed",valid=True,verdict="INCONCLUSIVE",completed_at=utc(),new_client_attempts=13,new_completed_inference_requests=counts,new_effective_output_tokens=tokens,expected_native400=1,cancelled_inference_attempts=1,cancelled_output_credit=0,native_reloads=0,native_epoch_unchanged=True)
atomic_json(r/"manifest.json",m)
(r/"summary.md").write_text("# "+r.name+"\n\nOpt-in persistentcoupled gateway on exact retainedhealthy Run38 group. Sevennew nativecapability positives+400/cancelzero, fournew postcleanrestart requests (Chat/Completion/ResponsesJSON/SSE) on samephysicalepoch. Native source/config/APIowners retained. Elevencomplete/13attempts/"+str(tokens)+" committedoutputs. Byte-exact nativeupstream/client restart wire, strictUsage/finish/DONE or typedResponsecompleted, lease/nativeidle and cleanjournal verified. CPU42failure/unclean latches have separate scope; no newnativefault/recovery/OSpowerloss/capacity/KEEP proof.\n")
print(json.dumps({"valid":True,"attempts":13,"completed":counts,"outputs":tokens}))
