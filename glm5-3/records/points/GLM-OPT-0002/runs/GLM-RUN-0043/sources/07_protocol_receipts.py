"""Offline native wire contract receipts. This validates observed attempts, not capacity."""
import hashlib,json
from pathlib import Path
from sse_observer import NativeSSEObserver
def verify_protocol_receipts(trace_path,expected_count,prompt_tokens,output_tokens,required_replicas=None):
 events=[json.loads(x) for x in Path(trace_path).read_text().splitlines() if x.strip()]
 leases=[x for x in events if x["event"]=="lease_acquired" and x.get("method")=="POST" and x.get("path")=="/v1/chat/completions" and x.get("output_budget")==output_tokens]
 if len(leases)!=expected_count:raise ValueError("native inference attempt count differs from contract")
 if required_replicas is not None and {x["replica"] for x in leases}!=set(required_replicas):raise ValueError("warmup did not cover each resident replica")
 rows=[]
 for lease in leases:
  matching=[x for x in events if x.get("lease_id")==lease["lease_id"]]
  contracts=[x for x in matching if x["event"]=="upstream_stream_contract"]
  release=[x for x in matching if x["event"]=="lease_released"]
  headers=[x for x in matching if x["event"]=="upstream_headers"]
  if len(contracts)!=1 or len(release)!=1 or len(headers)!=1:raise ValueError("native wire/headers/release receipt incomplete")
  wire=contracts[0];c=wire["contract"];u=c.get("usage")
  if headers[0]["status"]!=200 or wire["audit_error"] is not None or c.get("unknown",True) or c.get("native_error",True) or c.get("done") is not True:raise ValueError("native protocol failed/unknown/incomplete")
  if not isinstance(u,dict) or type(u.get("prompt_tokens")) is not int or type(u.get("completion_tokens")) is not int or u["prompt_tokens"]!=prompt_tokens or u["completion_tokens"]!=output_tokens:raise ValueError("native usage contract differs")
  if c.get("finish_reasons")!={"0":"length"} or not release[0]["released"] or release[0]["backend_failure"]:raise ValueError("native finish/release invalid")
  p=Path(wire["wire_path"]);digest=hashlib.sha256();size=0;observer=NativeSSEObserver(collect_contract=True)
  with p.open('rb') as stream:
   while block:=stream.read(65536):digest.update(block);size+=len(block);observer.feed(block)
  sha=digest.hexdigest()
  if size!=wire["wire_bytes"] or sha!=wire["wire_sha256"]:raise ValueError("native raw bytes/hash incomplete")
  if observer.contract()!=c:raise ValueError("raw native wire contract differs from live receipt")
  rows.append({"lease_id":lease["lease_id"],"replica":lease["replica"],"body_sha256":lease["body_sha256"],"usage":u,"finish_reasons":c["finish_reasons"],"done":True,"wire":{"path":str(p),"bytes":size,"sha256":sha},"valid":True})
 return {"measurement_valid":True,"requests":rows,"successful_inference_requests":len(rows),"effective_output_tokens":sum(x["usage"]["completion_tokens"] for x in rows),"limits":["wire receipts do not establish model quality or stable service capacity"]}
