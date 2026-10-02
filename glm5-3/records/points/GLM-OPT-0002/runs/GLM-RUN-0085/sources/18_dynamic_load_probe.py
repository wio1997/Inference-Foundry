"""Controlled open arrivals through an unmodified native-compatible gateway.

Raw token IDs and wire are authoritative. Finite draining windows are diagnostic,
not a stable online capacity or hardware bound.
"""
from pathlib import Path
import concurrent.futures,hashlib,json,threading,time,urllib.request,urllib.error
from phase_runner import atomic_json,utc
from sse_observer import NativeSSEObserver

def run_probe(plan,root,base_url,guard,metrics=None):
 root=Path(root);cases=plan["cases"];assert 0<len(cases)<=128
 assert len({x["id"]for x in cases})==len(cases)
 assert all(x["arrival_s"]>=0 and type(x["expected_outputs"])is int and x["expected_outputs"]>0 for x in cases)
 lock=threading.Lock();rows={};origin=time.monotonic();sampler_stop=threading.Event();samples=[]
 def save():
  with lock:atomic_json(root/"attempts.json",list(rows.values()))
 def sample():
  while not sampler_stop.is_set():
   try:
    guard();a=metrics();a.update(elapsed_s=time.monotonic()-origin,at=utc())
    samples.append(a)
    with(root/"native_samples.jsonl").open("a")as f:f.write(json.dumps(a)+"\n")
   except BaseException as error:
    atomic_json(root/"sample_error.json",dict(type=type(error).__name__,error=str(error)));return
   sampler_stop.wait(1)
 sampler=threading.Thread(target=sample,daemon=True)if metrics else None
 if sampler:sampler.start()
 def execute(case):
  remaining=origin+case["arrival_s"]-time.monotonic()
  while remaining>0:
   time.sleep(min(.02,remaining));remaining=origin+case["arrival_s"]-time.monotonic()
  guard();body=(root/case["body"]).read_bytes();start=time.monotonic()
  row=dict(id=case["id"],input_kind=case["input_kind"],expected_prompt_tokens=case["expected_prompt_tokens"],expected_outputs=case["expected_outputs"],scheduled_arrival_s=case["arrival_s"],actual_dispatch_s=start-origin,attempted_at=utc(),body_sha256=hashlib.sha256(body).hexdigest(),completed=False,effective_public_output_credit=0)
  with lock:rows[case["id"]]=row
  save();raw=b"";pending=b"";first=None;ids=[];points=[];obs=NativeSSEObserver(max_bytes=2097152,collect_contract=True)
  try:
   req=urllib.request.Request(base_url+"/v1/chat/completions",data=body,headers={"Content-Type":"application/json","X-Request-ID":plan["run_id"]+"-"+case["id"]})
   with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req,timeout=plan.get("request_timeout_s",900))as response:
    with lock:row["http_status"]=response.status;row["headers"]=list(response.headers.items())
    assert response.status==200
    while True:
     block=response.read1(65536)
     if not block:break
     now=time.monotonic();raw+=block;obs.feed(block);pending+=block
     while b"\n\n"in pending:
      frame,pending=pending.split(b"\n\n",1);data=b"\n".join(l[5:].removeprefix(b" ")for l in frame.splitlines()if l.startswith(b"data:"))
      if not data or data==b"[DONE]":continue
      value=json.loads(data);assert not value.get("error")
      for choice in value.get("choices",[]):
       delta=choice.get("delta")or{}
       if first is None and any(delta.get(k)for k in["content","reasoning","reasoning_content","tool_calls"]):first=now
       committed=choice.get("token_ids")or[]
       if committed:
        assert all(type(t)is int for t in committed);ids.extend(committed);points.append(dict(elapsed_s=now-start,origin_s=now-origin,committed_token_count=len(ids),chunk_commits=len(committed)))
    end=time.monotonic()
   con=obs.contract();assert con["done"]and not con["native_error"]and not con["unknown"]and con["finish_reasons"]=={"0":"length"}
   usage=con["usage"];assert usage["completion_tokens"]==len(ids)==case["expected_outputs"]and usage["prompt_tokens"]==case["expected_prompt_tokens"]and usage["total_tokens"]==usage["prompt_tokens"]+usage["completion_tokens"]
   assert first is not None and points
   gaps=[points[i]["elapsed_s"]-points[i-1]["elapsed_s"]for i in range(1,len(points))]
   with lock:row.update(completed=True,usage=usage,ttft_s=first-start,wall_s=end-start,dispatch_to_first_committed_s=points[0]["elapsed_s"],mean_committed_arrival_interval_s=(points[-1]["elapsed_s"]-points[0]["elapsed_s"])/(len(ids)-1)if len(ids)>1 else None,mean_after_first_output_s=(end-first)/(len(ids)-1)if len(ids)>1 else None,max_native_chunk_gap_s=max(gaps or[0]),committed_token_ids=ids,native_token_chunk_points=points,effective_public_output_credit=len(ids),contract=con,finished_origin_s=end-origin,finished_at=utc())
  except BaseException as error:
   if isinstance(error,urllib.error.HTTPError):
    raw+=error.read()
    with lock:row["http_status"]=error.code
   with lock:row.update(error_type=type(error).__name__,error=str(error))
   raise
  finally:
   wire=root/(case["id"]+".wire");wire.write_bytes(raw)
   with lock:row["wire"]=dict(path=str(wire),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
   save()
 errors=[]
 try:
  with concurrent.futures.ThreadPoolExecutor(max_workers=len(cases))as pool:
   futures=[pool.submit(execute,c)for c in cases]
   for future in futures:
    try:future.result()
    except BaseException as error:errors.append(dict(type=type(error).__name__,error=str(error)))
 finally:
  sampler_stop.set()
  if sampler:sampler.join(timeout=20)
 elapsed=time.monotonic()-origin;ordered=[rows[c["id"]]for c in cases]
 out=dict(at=utc(),functional_acceptance=not errors and len(ordered)==len(cases)and all(x["completed"]for x in ordered),requests=ordered,actual_new_requests=len(cases),effective_public_output_tokens=sum(x["effective_public_output_credit"]for x in ordered),elapsed_s=elapsed,finite_effective_output_tps=sum(x["effective_public_output_credit"]for x in ordered)/elapsed,errors=errors,metrics_samples=len(samples),limits=["Controlled finite open arrivals and subsequent drain; effectiveTPS is not stable online capacity","Native committedchunk timestamps includeMTP/transport; meaninterval is client observation, not GPUper-token time","Unique cache salts/input-output shapes/dispatch schedule are part of workload; not original81932to61440aisbench comparison"])
 atomic_json(root/"arrival_summary.json",out)
 if errors:raise RuntimeError("arrival workload failed: "+json.dumps(errors))
 return out
