import pathlib
import json,hashlib,re,sys
from pathlib import Path
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc
j=Path(__file__).parent; job=json.loads((j/"job.json").read_text()); r=Path(job["inputs"][0]["path"])
def ref(p,ident):
 return {"id":ident,"path":str(p),"bytes":p.stat().st_size,"sha256":hashlib.sha256(p.read_bytes()).hexdigest(),"locator":"entire curated artifact"}
def metric(p,key):
 values=[]
 for line in p.read_text().splitlines():
  if re.match(r"^vllm:"+re.escape(key)+r"(?:\{|\s)",line):
   values.append(float(line.rsplit(" ",1)[1]))
 return sum(values) if values else None
from sse_observer import NativeSSEObserver
from phase_runner import same_process
state=json.loads((r/"state.json").read_text());assert state["status"]=="completed" and not same_process(state["owner"])
spec=json.loads((r/"controller_spec.json").read_text())
for pin in spec["stages"][0]["sources"]:assert hashlib.sha256(Path(pin["path"]).read_bytes()).hexdigest()==pin["sha256"]
cap=r/"capability";results=list(sorted(cap.glob("*/load_result.json")))
audits=[];negatives=[]
for rp in results:
 data=json.loads(rp.read_text());assert data["valid"]
 plan=json.loads((rp.parent/"plan.json").read_text());assert len(data["requests"])==len(plan["requests"])
 for row in data["requests"]:
  assert row["valid"] and not row["error"]
  body=Path(row["body_path"]).read_bytes();assert hashlib.sha256(body).hexdigest()==row["body_sha256"]
  raw=rp.parent/(str(row["id"])+".response.jsonl");events=[json.loads(line) for line in raw.read_text().splitlines()]
  if row["outcome"]=="expected_rejection":
   import base64
   assert row["http_status"]==row["expected"]["http_status"]==400 and len(events)==1
   payload=base64.b64decode(events[0]["error_body_base64"]);assert len(payload)==events[0]["error_body_bytes"] and hashlib.sha256(payload).hexdigest()==events[0]["error_body_sha256"]
   negatives.append({"body":ref(Path(row["body_path"]),"negative-body"),"response":ref(raw,"negative-response"),"status":400,"output_credit":0});continue
  assert row["outcome"]=="completed" and row["done"] and row["finish_reasons"]
  digest=hashlib.sha256();bp=json.loads(body)
  if bp.get("stream"):
   observer=NativeSSEObserver(collect_contract=True)
   for event in events:
    val=event["data"];assert isinstance(val,str);observer.feed(("data: "+val+"\n\n").encode())
    if val!="[DONE]":
     for c in json.loads(val).get("choices",[]):
      delta=c.get("delta",{});text=delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning") or c.get("text") or ""
      if text or delta.get("tool_calls") or delta.get("function_call"):digest.update(json.dumps(delta,sort_keys=True,ensure_ascii=False).encode())
   c=observer.contract();assert c["done"] and not c["unknown"] and not c["native_error"] and c["usage"]==row["usage"] and c["finish_reasons"]==row["finish_reasons"]
   assert digest.hexdigest()==row["delta_stream_sha256"]
  else:
   assert len(events)==1;value=events[0]["data"];assert not value.get("error") and value["usage"]==row["usage"]
   assert {str(c.get("index",0)):c["finish_reason"] for c in value["choices"]}==row["finish_reasons"]
  assert len(row["finish_reasons"])==bp.get("n",1) and row["usage"]["completion_tokens"]==row["expected"]["output_tokens"]
  audits.append({"result":str(rp),"request_id":row["id"],"body_sha256":row["body_sha256"],"usage":row["usage"],"done":True,"finish_reasons":row["finish_reasons"],"response_events":len(events),"response":ref(raw,"response-"+str(len(audits)))})
assert len(audits)==9 and len(negatives)==1 and sum(a["usage"]["completion_tokens"] for a in audits)==1552
trace=[json.loads(x) for x in (cap/"router_trace.jsonl").read_text().splitlines()];leases=[x for x in trace if x["event"]=="lease_acquired"];assert len(leases)==9
for lease in leases:
 seq=[x for x in trace if x.get("lease_id")==lease["lease_id"]]
 releases=[x for x in seq if x["event"]=="lease_released"];assert len(releases)==1 and releases[0]["released"]
ev=json.loads((cap/"capability_events.json").read_text())
cancel=[x for x in ev if x["event"]=="client_closed_after_first_output"];assert len(cancel)==1 and json.loads(cancel[0]["first_event"])["choices"]
assert len([x for x in ev if x["event"]=="cancel_lease_released"])==1
assert ev[-1]["event"]=="gateway_stopped" and ev[-1]["exit_code"]==-15
assert all(v==0 for vals in [x for x in ev if x["event"]=="final"][-1]["metrics"].values() for v in vals.values())

groups=json.loads((r/"execution_groups.json").read_text());assert len(groups)==1
group=groups[0];assert group["members"]==["DP0","DP1"]
for entry in ev:
 if "snapshot" in entry:
  for item in entry["snapshot"]["replicas"]:
   assert item["execution_group"]==group["id"] and item["native_owner_epoch"]==group["epoch"] and not item["group_faulted"]
bindings={}
for lease in leases:
 assert lease["body_sha256"] not in bindings
 seq=[x for x in trace if x.get("lease_id")==lease["lease_id"]]
 contracts=[x for x in seq if x["event"]=="upstream_stream_contract"];assert len(contracts)==1
 c=contracts[0];assert c["audit_error"] is None
 p=Path(c["wire_path"]);raw=p.read_bytes();assert len(raw)==c["wire_bytes"] and hashlib.sha256(raw).hexdigest()==c["wire_sha256"]
 releases=[x for x in seq if x["event"]=="lease_released"];assert len(releases)==1 and releases[0]["released"] and not releases[0]["backend_failure"]
 headers=[x for x in seq if x["event"]=="upstream_headers"];assert len(headers)==1
 bindings[lease["body_sha256"]]={"lease":lease,"status":headers[0]["status"],"contract":c,"release":releases[0],"wire":ref(p,lease["lease_id"])}
proxy_completed=0
for rp in results:
 if rp.parent.name.startswith("local_"):continue
 d=json.loads(rp.read_text())
 for row in d["requests"]:
  match=bindings[row["body_sha256"]];assert match["status"]==row["http_status"]
  raw=Path(match["wire"]["path"]).read_bytes();bp=json.loads(Path(row["body_path"]).read_bytes())
  events=[json.loads(x) for x in (rp.parent/(row["id"]+".response.jsonl")).read_text().splitlines()]
  if row["outcome"]=="expected_rejection":
   import base64
   assert raw==base64.b64decode(events[0]["error_body_base64"]);continue
  if bp["stream"]:
   obs=NativeSSEObserver(collect_contract=True);obs.feed(raw);c=obs.contract()
   assert c==match["contract"]["contract"] and c["done"] and not c["unknown"] and not c["native_error"] and c["usage"]==row["usage"] and c["finish_reasons"]==row["finish_reasons"]
   frames=[x[5:].strip() for x in raw.decode().splitlines() if x.startswith("data:")]
   assert frames==[x["data"] for x in events]
  else:
   val=json.loads(raw);assert val==events[0]["data"] and val["usage"]==row["usage"] and not val.get("error")
  proxy_completed+=1
assert proxy_completed==7
cancel_payload=json.loads((cap/"cancel.body.json").read_text())
cancel_sha=hashlib.sha256(json.dumps(cancel_payload,ensure_ascii=False,separators=(",",":")).encode()).hexdigest()
match=bindings[cancel_sha];assert match["status"]==200 and not match["contract"]["contract"]["done"] and match["contract"]["contract"]["usage"] is None
assert cancel[0]["first_event"] in [x[5:].strip() for x in Path(match["wire"]["path"]).read_text().splitlines() if x.startswith("data:")]
assert len(bindings)==9
identity=json.loads((r/"adopted_model_identities.json").read_text())
prior=json.loads((r.parent/"GLM-RUN-0034/adopted_model_identities.json").read_text());assert identity==prior
expected_epoch=hashlib.sha256(json.dumps({k:{f:v[f]for f in ["host","pid","identity","argv"]}for k,v in prior.items()},sort_keys=True).encode()).hexdigest()
assert group["epoch"]==expected_epoch
source=(r/"adopt_models.py").read_text().replace('atomic_json(root/"adopted_model_identities.json",identities)','atomic_json(j/"current_model_identities.json",identities)')
ns={"__file__":str(r/"adopt_models.py"),"j":j};exec(compile(source,"read-only exact current cohort","exec"),ns);assert ns["identities"]==identity
import urllib.request
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}));idle={}
for rank,node,port in [(0,"166",9081),(1,"167",9900)]:
 with opener.open("http://172.16.10."+node+":"+str(port)+"/metrics",timeout=10) as reply:text=reply.read().decode()
 p=j/("final_DP"+str(rank)+".metrics");p.write_text(text)
 idle["DP"+str(rank)]={k:metric(p,k)for k in ["num_requests_running","num_requests_waiting"]};assert all(v==0 for v in idle["DP"+str(rank)].values())
for row in json.loads((r/"artifact_index.json").read_text()):
 p=Path(row["path"]);assert p.stat().st_size==row["bytes"] and hashlib.sha256(p.read_bytes()).hexdigest()==row["sha256"],str(p)
out={"run_id":r.name,"measurement_valid":True,"functional_acceptance":True,"verdict":"INCONCLUSIVE","generated_at":utc(),"new_client_attempts":11,"new_completed_inference_requests":9,"new_effective_output_tokens":1552,"negative_attempts":1,"cancelled_attempts":1,"cancelled_output_credit":0,"native_group":group,"native_identities":identity,"native_idle":idle,"offline_response_receipts":audits,"negative_receipts":negatives,"proxy_wire_receipts":list(bindings.values()),"limits":["Native healthy capability only; failure-latch behavior CPU contract using actual Run32 native error payload, no new live fault/recovery","No persistent gateway recovery proof","Client response jsonl retains event sequence; exact upstream audit wires hashed independently, not byte-exact client wire","Nonstream observer unknown is handled by independently validating native JSON Usage/finish","No performance/KEEP/stable capacity claim"]}
p=r/"reduction.json";atomic_json(p,out)
m=json.loads((r/"manifest.json").read_text());m["reduction"]=ref(p,"reduction");atomic_json(r/"manifest.json",m)
atomic_json(job["result"]["path"],{"schema_version":1,"job_id":job["job_id"],"status":"completed","summary":"All11 native attempts audited:9 complete1552outputs;400/cancel zero credit; optin group epochs/drain/readd/leases/native wires/nativeidle/sourcepins verified","execution":{"inner_exit_code":0,"acceptance":"passed","processes":[]},"findings":[{"kind":"fact","text":"Actual healthy native capability preserved with physicalgroup declared; CPU failure latch separately bounded","scope":{"run_id":r.name,"new_native_fault_injection":False},"evidence_ids":["reduction"]}],"evidence":[ref(p,"reduction")],"unknowns":out["limits"],"decision_request":None,"next_check_at":None})
print(json.dumps(ref(p,"reduction")))
