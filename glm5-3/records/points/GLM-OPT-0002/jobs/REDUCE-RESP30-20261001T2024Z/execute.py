import pathlib,json,hashlib,subprocess,sys
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import same_process,atomic_json,utc
j=pathlib.Path(__file__).parent;job=json.loads((j/"job.json").read_text());r=pathlib.Path(job["inputs"][0]["path"]);state=json.loads((r/"state.json").read_text())
assert state["status"]=="failed" and state["failure_phase"]=="responses" and not same_process(state["owner"])
a=json.loads((r/"attempts.json").read_text());assert len(a)==1 and a[0]["http_status"]==200 and a[0]["error_type"]=="AttributeError"
for key in ["wire","body"]:
 row=a[0][key];raw=pathlib.Path(row["path"]).read_bytes();assert len(raw)==row["bytes"] and hashlib.sha256(raw).hexdigest()==row["sha256"]
probe="""import pathlib,json
from vllm.entrypoints.openai.responses.protocol import ResponsesResponse
r=pathlib.Path('/data/tiankuan/wio/Inference-Foundry/glm5-3/records/points/GLM-OPT-0002/runs/GLM-RUN-0030')
value=json.loads((r/'direct0_json.wire').read_bytes());x=ResponsesResponse.model_validate(value);assert x.status=='completed' and not value.get('error') and x.max_output_tokens==64
u=x.usage.model_dump();assert u['input_tokens']==13 and u['output_tokens']==2 and u['total_tokens']==15
print(json.dumps({'valid':True,'usage':u,'status':x.status,'native_schema_has_error_field':'error' in ResponsesResponse.model_fields,'output':value['output'],'inference_calls':0}))
"""
p=subprocess.run(["docker","exec","-i","glm52-single","python3","-"],input=probe.encode(),capture_output=True,timeout=30);p.check_returncode();(j/"probe.stdout").write_bytes(p.stdout);data=json.loads(p.stdout.decode().splitlines()[-1]);assert data["valid"] and not data["native_schema_has_error_field"]
red={"run_id":r.name,"valid_partial_evidence":True,"verdict":"INVALID","reason":"Driver assumed installed ResponsesResponse.error attribute; native raw response valid, no API/native model failure","new_client_attempts":1,"new_completed_inference_requests":1,"new_effective_output_tokens":2,"revalidated_native_response":data,"raw_request":a[0],"controller_terminal":state,"gateway_terminal":json.loads((r/"gateway_terminal.json").read_text()),"remaining_attempts":8,"limits":["One nativeJSONResponse, stateless/storedisabled, not fullfeature/stream/performance proof","Captured first response reused only, never new credit or GPU replay"]}
atomic_json(r/"reduction.json",red);m=json.loads((r/"manifest.json").read_text());m.update(status="failed",valid=False,verdict="INVALID",partial_evidence_valid=True,new_client_attempts=1,new_completed_inference_requests=1,new_effective_output_tokens=2,reason=red["reason"]);atomic_json(r/"manifest.json",m)
(r/"summary.md").write_text("# "+r.name+"\n\nINVALID driver: installednative ResponsesResponse omits error attribute. FirstdirectJSONnative response is valid/completed:input13/output2/total15,READY; wire1107B SHA6e85ff1592d1aebb68ded9d57d77ce4651a16a8a8fb81dbb355e35affef7d24c. Source/spec/raw preserved. One actual completed attempt/2 committed outputs partial evidence, gatewayexited-15/controllerdead. Remaining8 attempts belongnewRun, no firstrequest replay. No NativeResponses/model rejection or capacityclaim.\n")
raw=(r/"reduction.json").read_bytes();atomic_json(job["result"]["path"],{"schema_version":1,"job_id":job["job_id"],"status":"completed","summary":"Run30 driver INVALID; captured nativeJSONResponse valid2 outputs, no inference replay,8 remaining attempts","execution":{"inner_exit_code":0,"acceptance":"passed","processes":[]},"findings":[{"kind":"fact","text":red["reason"],"scope":{"run_id":r.name},"evidence_ids":["partial"]}],"evidence":[{"id":"partial","path":str(r/"reduction.json"),"bytes":len(raw),"sha256":hashlib.sha256(raw).hexdigest(),"locator":"native schema/raw hash/usage/status and faileddriver"}],"unknowns":red["limits"],"decision_request":None,"next_check_at":None});print(json.dumps({"outputs":2,"native_valid":True,"GPU_calls":0}))
