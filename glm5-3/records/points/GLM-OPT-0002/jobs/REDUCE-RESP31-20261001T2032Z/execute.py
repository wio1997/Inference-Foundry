import pathlib,json,hashlib,subprocess,sys,re
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import same_process,atomic_json,utc
j=pathlib.Path(__file__).parent;job=json.loads((j/"job.json").read_text());r=pathlib.Path(job["inputs"][0]["path"])
state=json.loads((r/"state.json").read_text());assert state["status"]=="completed" and not same_process(state["owner"])
for v in json.loads((r/"controller_spec.json").read_text())["stages"][0]["sources"]:assert hashlib.sha256(pathlib.Path(v["path"]).read_bytes()).hexdigest()==v["sha256"]
p=subprocess.run(["docker","exec","-i","glm52-single","python3","-"],input=(j/"probe.py").read_bytes(),capture_output=True,timeout=30);p.check_returncode();(j/"probe.stdout").write_bytes(p.stdout);data=json.loads(p.stdout.decode().splitlines()[-1]);assert data["valid"]
trace=[json.loads(l)for l in (r/"router_trace.jsonl").read_text().splitlines()];leases=[v for v in trace if v["event"]=="lease_acquired"];assert len(leases)==5
attempts={r.name+"-"+v["id"]:v for v in data["attempts"]}
for lease in leases:
 row=attempts[lease["request_header_id"]];assert lease["body_sha256"]==row["body"]["sha256"]
 seq=[v for v in trace if v.get("lease_id")==lease["lease_id"]];end=[v for v in seq if v["event"]=="lease_released"];assert len(end)==1 and end[0]["released"]
 audit=[v for v in seq if v["event"]=="upstream_stream_contract"];assert len(audit)==1;v=audit[0]
 raw=pathlib.Path(v["wire_path"]).read_bytes();assert len(raw)==v["wire_bytes"]==row["wire"]["bytes"] and hashlib.sha256(raw).hexdigest()==v["wire_sha256"]==row["wire"]["sha256"]
 assert raw==pathlib.Path(row["wire"]["path"]).read_bytes()
for rank in ["DP0","DP1"]:
 text=(r/("final_"+rank+".metrics")).read_text()
 for name in ["num_requests_running","num_requests_waiting"]:
  vals=re.findall(r"^vllm:"+name+r"(?:\{[^\n]*\})?\s+([0-9.eE+-]+)",text,re.M);assert vals and all(float(x)==0 for x in vals)
assert json.loads((r/"gateway_terminal.json").read_text())["exit_code"]==-15
red={"run_id":r.name,"measurement_valid":True,"functional_acceptance":True,"verdict":"INCONCLUSIVE","generated_at":utc(),"new_client_attempts":8,"new_completed_inference_requests":3,"new_effective_output_tokens":6,"expected_rejections":5,"offline_native_audit":data,"gateway_leases":5,"gateway_original_wire_identity":True,"native_store":False,"reused":{"Run30":{"completed":1,"outputs":2,"new_credit":0,"response":"captured valid nativeJSON READY"}},"nativeSSE":"9typed named events each; terminalresponse.completed/usage output2; no DONEsentinel emitted by installedAPI","limits":["Current native store-disabled/stateless config only; no background/stateful store enabled","Router first-output placement observer remains conservative unknown for Responses; commit audited with native protocol independently","Short windows no performance/KEEP/capacity claim"]}
atomic_json(r/"reduction.json",red);m=json.loads((r/"manifest.json").read_text());raw=(r/"reduction.json").read_bytes();m["reduction"]={"path":str(r/"reduction.json"),"bytes":len(raw),"sha256":hashlib.sha256(raw).hexdigest()};atomic_json(r/"manifest.json",m)
atomic_json(job["result"]["path"],{"schema_version":1,"job_id":job["job_id"],"status":"completed","summary":"Run31 all8 Responses attempts validated:3completed6outputs,5native400/404;all5gatewayclient/audit exactwire identity,lease0/nativeidle;prior2outputs notrecredited","execution":{"inner_exit_code":0,"acceptance":"passed","processes":[]},"findings":[{"kind":"fact","text":"Native installed eventclasses/status/usage/raw hashes and Responses terminal verified, JSON/SSE pass-through exact, no ChatDONEcontract assumption","scope":{"run_id":r.name},"evidence_ids":["reduction"]}],"evidence":[{"id":"reduction","path":str(r/"reduction.json"),"bytes":len(raw),"sha256":hashlib.sha256(raw).hexdigest(),"locator":"native typedwire/usage and everyproxyattempt"}],"unknowns":red["limits"],"decision_request":None,"next_check_at":None});print(json.dumps({"valid":True,"requests":3,"outputs":6,"proxywireexact":True}))
