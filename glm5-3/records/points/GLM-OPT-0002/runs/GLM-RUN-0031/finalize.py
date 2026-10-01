import pathlib,json,sys,hashlib
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc
r=pathlib.Path(__file__).parent;s=json.loads((r/"responses_summary.json").read_text());assert s["valid"]
for stage in ["prepare","responses"]:assert json.loads((r/(stage+".phase.json")).read_text())["status"]=="succeeded"
rows=[json.loads(l)for l in (r/"router_trace.jsonl").read_text().splitlines()];leases=[x for x in rows if x["event"]=="lease_acquired"];assert len(leases)==5
for lease in leases:
 seq=[x for x in rows if x.get("lease_id")==lease["lease_id"]];released=[x for x in seq if x["event"]=="lease_released"];assert len(released)==1 and released[0]["released"]
assert json.loads((r/"gateway_terminal.json").read_text())["exit_code"]==-15
m=json.loads((r/"manifest.json").read_text());m.update(status="completed",valid=True,verdict="INCONCLUSIVE",completed_at=utc(),**{k:v for k,v in s.items()if k.startswith("new_")},summary_ref={"path":str(r/"responses_summary.json"),"sha256":hashlib.sha256((r/"responses_summary.json").read_bytes()).hexdigest()});atomic_json(r/"manifest.json",m)
(r/"summary.md").write_text("# "+r.name+"\n\nFinite Responses API compatibility on exact retained Run28 cohort.3 newcompleted JSON/SSE requests/"+str(s["new_effective_output_tokens"])+" native committed output tokens;5 expected native400/404 preserve configured store-disabled semantics, raw bodies/wires/headers/events retained. Native named SSE response.completed is terminal, no Chat DONE sentinel emitted. Bothnative idle and5gateway leases released, gateway exited. Run30 firstnativeJSON/2outputs referencedonly, no newcredit/replay. No stateful store/background support enabled, no throughput/KEEP/capacity claim.\n")
print(json.dumps(s))
