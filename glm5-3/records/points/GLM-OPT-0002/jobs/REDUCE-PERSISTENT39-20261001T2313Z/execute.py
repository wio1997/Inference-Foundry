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
assert len(audits)==7 and len(negatives)==1 and sum(a["usage"]["completion_tokens"] for a in audits)==1296

identity=json.loads((r/"adopted_model_identities.json").read_text())
startup=json.loads((r/"startup_model_identities.json").read_text())
for k,v in identity.items():assert all(v[f]==startup[k][f]for f in ["host","pid","identity","argv"])
assert set(identity)=={"DP0","DP1","DP2","DP3"}
group=json.loads((r/"execution_groups.json").read_text())[0]
assert group["members"]==["DP0","DP1","DP2","DP3"]
assert group["epoch"]==hashlib.sha256(json.dumps({k:{f:v[f]for f in ["host","pid","identity","argv"]}for k,v in identity.items()},sort_keys=True).encode()).hexdigest()
wire_receipts=[];all_bindings={}
for scope,count in [("capability",9)]:
 trace=[json.loads(x)for x in (r/scope/"router_trace.jsonl").read_text().splitlines()]
 leases=[x for x in trace if x["event"]=="lease_acquired"];assert len(leases)==count
 by_id={};by_body={}
 for lease in leases:
  key=lease["lease_id"];assert key not in all_bindings
  seq=[x for x in trace if x.get("lease_id")==key]
  rel=[x for x in seq if x["event"]=="lease_released"];assert len(rel)==1 and rel[0]["released"] and not rel[0]["backend_failure"]
  hdr=[x for x in seq if x["event"]=="upstream_headers"];assert len(hdr)==1
  contracts=[x for x in seq if x["event"]=="upstream_stream_contract"];assert len(contracts)==1
  contract=contracts[0];assert contract["audit_error"] is None
  raw=Path(contract["wire_path"]).read_bytes();assert len(raw)==contract["wire_bytes"] and hashlib.sha256(raw).hexdigest()==contract["wire_sha256"]
  val={"scope":scope,"lease":lease,"status":hdr[0]["status"],"contract":contract,"release":rel[0]}
  all_bindings[key]=val
  if lease["request_header_id"] is not None:
   assert lease["request_header_id"] not in by_id;by_id[lease["request_header_id"]]=val
  by_body.setdefault(lease["body_sha256"],[]).append(val)
  wire_receipts.append(val)
 result_paths=[x for x in sorted((r/"capability").glob("*/load_result.json")) if not x.parent.name.startswith("local_")] if scope=="capability" else [r/"pilot/dynamic/load_result.json"]
 matches=set()
 for result_path in result_paths:
  data=json.loads(result_path.read_text())
  for row in data["requests"]:
   rid=r.name+"-pilot-"+row["id"]
   match=by_id[rid] if scope=="pilot" else by_body[row["body_sha256"]][0]
   if scope=="capability":assert len(by_body[row["body_sha256"]])==1
   assert match["lease"]["body_sha256"]==row["body_sha256"] and match["status"]==row["http_status"]
   key=match["lease"]["lease_id"];assert key not in matches;matches.add(key)
   raw=Path(match["contract"]["wire_path"]).read_bytes()
   events=[json.loads(x)for x in (result_path.parent/(row["id"]+".response.jsonl")).read_text().splitlines()]
   bp=json.loads(Path(row["body_path"]).read_bytes())
   if row["outcome"]=="expected_rejection":
    import base64
    assert raw==base64.b64decode(events[0]["error_body_base64"]);continue
   if bp["stream"]:
    obs=NativeSSEObserver(collect_contract=True);obs.feed(raw);c=obs.contract()
    assert c==match["contract"]["contract"] and c["done"] and not c["native_error"] and not c["unknown"] and c["usage"]==row["usage"] and c["finish_reasons"]==row["finish_reasons"]
    assert [x[5:].strip()for x in raw.decode().splitlines()if x.startswith("data:")]==[x["data"]for x in events]
   else:assert json.loads(raw)==events[0]["data"]
 if scope=="capability":
  ev=json.loads((r/"capability/capability_events.json").read_text())
  cancel=[x for x in ev if x["event"]=="client_closed_after_first_output"];assert len(cancel)==1
  cp=json.loads((r/"capability/cancel.body.json").read_text());ch=hashlib.sha256(json.dumps(cp,ensure_ascii=False,separators=(",",":")).encode()).hexdigest()
  match=by_body[ch][0];assert not match["contract"]["contract"]["done"] and match["contract"]["contract"]["usage"] is None
  assert match["lease"]["lease_id"] not in matches;matches.add(match["lease"]["lease_id"])
  assert cancel[0]["first_event"]in [x[5:].strip()for x in Path(match["contract"]["wire_path"]).read_text().splitlines()if x.startswith("data:")]
  assert len([x for x in ev if x["event"]=="cancel_lease_released"])==1
  for e in ev:
   if "snapshot"in e:
    for item in e["snapshot"]["replicas"]:assert item["execution_group"]==group["id"]and item["native_owner_epoch"]==group["epoch"]and not item["group_faulted"]
  assert all(v==0 for vals in [x for x in ev if x["event"]=="final"][-1]["metrics"].values()for v in vals.values())
 else:
  ev=json.loads((r/"pilot/pilot_events.json").read_text())
  assert all(item["active_requests"]==0 and not item["group_faulted"]for item in json.loads((r/"pilot/final_placement.json").read_text())["replicas"])
 assert ev[-1]["event"]=="gateway_stopped" and ev[-1]["exit_code"]==-15
 assert len(matches)==len(leases)
source=(r/"adopt_models.py").read_text().replace('atomic_json(root/"adopted_model_identities.json",identities)','atomic_json(j/"current_model_identities.json",identities)').replace('atomic_json(root/"execution_groups.json",','atomic_json(j/"current_execution_groups.json",')
ns={"__file__":str(r/"adopt_models.py"),"j":j};exec(compile(source,"read-only exact current TP32 cohort","exec"),ns);assert ns["identities"]==identity
import urllib.request
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
idle={}
for n,node,port in [("DP0","166",9081),("DP1","166",9082),("DP2","167",9900),("DP3","167",9901)]:
 with opener.open("http://172.16.10."+node+":"+str(port)+"/metrics",timeout=10)as response:text=response.read().decode()
 f=j/("final_"+n+".metrics");f.write_text(text);idle[n]={k:metric(f,k)for k in ["num_requests_running","num_requests_waiting"]};assert all(v==0 for v in idle[n].values())
for entry in json.loads((r/"artifact_index.json").read_text()):
 p=Path(entry["path"]);assert p.stat().st_size==entry["bytes"] and hashlib.sha256(p.read_bytes()).hexdigest()==entry["sha256"],str(p)
import subprocess,os
cp=subprocess.run(["docker","exec","glm52-single","python3",str(j/"native_response_validate.py")],capture_output=True,timeout=120)
(j/"native_response_validation.stdout").write_bytes(cp.stdout);(j/"native_response_validation.stderr").write_bytes(cp.stderr);cp.check_returncode()
native_response=json.loads((j/"native_responses_validation.json").read_text());assert native_response["valid"] and native_response["native_inference_calls"]==0
restart_rows=json.loads((r/"restart/attempts.json").read_text());assert len(restart_rows)==4 and all(v["status"]=="completed" and v["http_status"]==200 for v in restart_rows)
trace=[json.loads(l)for l in(r/"restart/router_trace.jsonl").read_text().splitlines()];leases=[x for x in trace if x["event"]=="lease_acquired"];assert len(leases)==4
restart_audits=[]
for row in restart_rows:
 for k in ["body","wire"]:
  f=Path(row[k]["path"]);assert f.stat().st_size==row[k]["bytes"] and hashlib.sha256(f.read_bytes()).hexdigest()==row[k]["sha256"]
 body=json.loads(Path(row["body"]["path"]).read_bytes());raw=Path(row["wire"]["path"]).read_bytes()
 lease=next(x for x in leases if x["request_header_id"]==r.name+"-restart-"+row["id"]);assert lease["body_sha256"]==row["body"]["sha256"]
 events=[x for x in trace if x.get("lease_id")==lease["lease_id"]];release=[x for x in events if x["event"]=="lease_released"]
 assert len(release)==1 and release[0]["released"] and not release[0]["backend_failure"]
 receipt=next(x for x in events if x["event"]=="upstream_stream_contract");assert receipt["audit_error"] is None
 upstream=Path(receipt["wire_path"]);assert upstream.read_bytes()==raw and hashlib.sha256(raw).hexdigest()==receipt["wire_sha256"] and len(raw)==receipt["wire_bytes"]
 if row["id"].startswith("responses"):
  result=native_response["validated"][row["id"]];assert result["usage"]==row["usage"] and result["usage"]["output_tokens"]==row["output_credit"]
 else:
  observer=NativeSSEObserver(collect_contract=True);observer.feed(raw);contract=observer.contract();assert contract==row["contract"] and contract["done"] and not contract["unknown"] and not contract["native_error"] and contract["finish_reasons"]=={"0":"length"} and contract["usage"]["completion_tokens"]==body["max_tokens"]==row["output_credit"]
 restart_audits.append({"id":row["id"],"body":row["body"],"wire":row["wire"],"lease":lease,"upstream":receipt,"release":release[0],"credit":row["output_credit"]})
journal=json.loads((r/"fault_state.json").read_text());assert not journal["open"] and all(not x["faulted"]for x in journal["groups"].values())
assert journal["groups"][group["id"]]["epoch"]==group["epoch"]
assert os.stat(r/"fault_state.json").st_mode&0o777==0o600
for path in [r/"capability/journal_after_clean_shutdown.json",r/"restart/journal_after_shutdown.json"]:
 assert json.loads(path.read_text())==journal
start=json.loads((r/"restart/before.json").read_text());assert start["same_epoch_restart_admissible"] and start["journal"]["open"] and all(not x["faulted"]for x in start["journal"]["groups"].values())
for x in start["placement"]["replicas"]:assert x["fault_journal_enabled"] and x["fault_journal_error"] is None and not x["group_faulted"] and x["native_owner_epoch"]==group["epoch"]
for event in json.loads((r/"capability/capability_events.json").read_text()):
 if event["event"]=="reused_local_full_request":
  assert event["sha256"]==hashlib.sha256(Path(event["path"]).read_bytes()).hexdigest()
outputs=1296+sum(row["output_credit"]for row in restart_rows)
m=json.loads((r/"manifest.json").read_text());assert m["valid"] and m["new_client_attempts"]==13 and m["new_completed_inference_requests"]==11 and m["new_effective_output_tokens"]==outputs
out={"run_id":r.name,"measurement_valid":True,"functional_acceptance":True,"verdict":"INCONCLUSIVE","generated_at":utc(),"new_client_attempts":13,"new_completed_inference_requests":11,"new_effective_output_tokens":outputs,"native_reloads":0,"native_epoch_unchanged":True,"cancelled_output_credit":0,"expected_native400":1,"native_group":group,"native_identities":identity,"final_native_idle":idle,"offline_response_receipts":audits,"negative_receipts":negatives,"proxy_wire_receipts":wire_receipts,"restart_wire_receipts":restart_audits,"native_response_validation":native_response,"journal_after_clean_shutdown":journal,"limits":["Native positive persistentgateway/400/cancel/drain/generation/byte-exact restart ChatCompletionResponses E2E valid on retained healthyphysicalgroup","Native fault/unclean branches remainCPU42 historicalpayload/process-exit fixtures, no newnativefailure/recovery/OSpowerloss/storage-disaster proof","Reused directlocal Run38 controls no replay/newcredit","No performance/KEEP/stablecapacity claim"]}
p=r/"reduction.json";atomic_json(p,out);m["reduction"]=ref(p,"reduction");atomic_json(r/"manifest.json",m)
atomic_json(job["result"]["path"],{"schema_version":1,"job_id":job["job_id"],"status":"completed","summary":"Persistentgateway actual13attempts/11complete"+str(outputs)+"outputs and healthy sameepoch cleanrestart independentlyaudited; nativefault/unclean CPU42 scope separate","execution":{"inner_exit_code":0,"acceptance":"passed","processes":[]},"findings":[],"evidence":[ref(p,"reduction")],"unknowns":out["limits"],"decision_request":None,"next_check_at":None})
print(json.dumps(ref(p,"reduction")))
