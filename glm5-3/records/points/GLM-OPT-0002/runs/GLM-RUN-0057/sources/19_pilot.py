import json,sys,time,hashlib,urllib.request,urllib.error,re,subprocess,shlex
from pathlib import Path
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc
from sse_observer import NativeSSEObserver
from owner_guard import start_watchdog,guard
start_watchdog()
r=Path(__file__).parent;http=urllib.request.build_opener(urllib.request.ProxyHandler({}));rows=[]
owners=json.loads((r/"adopted_model_identities.json").read_text())
def ownercheck():
 for x in owners.values():
  code="import pathlib,json;p=pathlib.Path('/proc/"+str(x["pid"])+"');s=(p/'stat').read_text();print(json.dumps({'boot_id':pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'start_ticks':s[s.rfind(')')+2:].split()[19],'argv':[x.decode()for x in(p/'cmdline').read_bytes().split(bytes([0]))if x]}))";args=["python3","-c",code]
  if x["host"]=="167":args=["ssh","-o","BatchMode=yes","root@172.16.10.167",shlex.join(args)]
  p=subprocess.run(args,capture_output=True,timeout=20);p.check_returncode();i=json.loads(p.stdout);assert {k:i[k]for k in ["boot_id","start_ticks"]}==x["identity"]and i["argv"]==x["argv"]
def snapshot(label):
 guard();ownercheck()
 for key,x in owners.items():
  for ep in ["health","metrics"]:
   with http.open("http://172.16.10."+x["host"]+":"+str(x["port"])+"/"+ep,timeout=10)as z:
    assert z.status==200;(r/(label+"_"+key+"."+ep)).write_bytes(z.read())
def req(name,key,body,kind,expected,min_prompt=None):
 guard();ownercheck();x=owners[key];f=r/(name+".body.json");atomic_json(f,body);raw=b"";start=time.monotonic();first=None
 row={"name":name,"owner":key,"kind":kind,"attempted_at":utc(),"request_body_sha256":hashlib.sha256(f.read_bytes()).hexdigest(),"expected_output_tokens":expected,"completed":False,"effective_public_output_credit":0};rows.append(row);atomic_json(r/"attempts.json",rows)
 request=urllib.request.Request("http://172.16.10."+x["host"]+":"+str(x["port"])+"/v1/chat/completions",data=f.read_bytes(),headers={"Content-Type":"application/json","X-Request-ID":"GLM-RUN-0057-"+name})
 try:
  with http.open(request,timeout=240)as response:
   row["http_status"]=response.status
   while True:
    block=response.read1(65536)
    if not block:break
    raw+=block
    if first is None and body.get("stream"):
     for frame in raw.replace(b"\r\n",b"\n").split(b"\n\n"):
      data=b"\n".join(v[5:].removeprefix(b" ")for v in frame.splitlines()if v.startswith(b"data:"))
      try:o=json.loads(data)
      except (ValueError,UnicodeDecodeError):continue
      for c in o.get("choices",[]):
       d=c.get("delta")or{}
       if any(d.get(z)for z in ["content","reasoning","reasoning_content","tool_calls"]):first=time.monotonic();break
  assert row["http_status"]==200
  if body.get("stream"):
   obs=NativeSSEObserver(collect_contract=True);obs.feed(raw);contract=obs.contract();assert contract["done"]and not contract["native_error"]and not contract["unknown"];usage=contract["usage"];assert contract["finish_reasons"].get("0")=="length";value=None
  else:
   value=json.loads(raw);assert not value.get("error");usage=value["usage"];assert len(value["choices"])==1 and value["choices"][0]["finish_reason"]=="length";contract={"nativeJSON":True,"finish_reason":"length"}
  assert type(usage["completion_tokens"])is int and usage["completion_tokens"]==expected and usage["total_tokens"]==usage["prompt_tokens"]+usage["completion_tokens"]
  if min_prompt is not None:assert min_prompt<usage["prompt_tokens"]<144352
  row.update(completed=True,usage=usage,contract=contract,wall_s=time.monotonic()-start,ttft_s=None if first is None else first-start,effective_public_output_credit=0 if kind=="internal_P_KV_helper"else expected)
  return value
 except urllib.error.HTTPError as e:
  raw=e.read();row.update(http_status=e.code,error="nativeHTTPerror");raise
 except BaseException as e:row.update(error_type=type(e).__name__,error=str(e));raise
 finally:
  f=r/(name+".wire");f.write_bytes(raw);row["wire"]={"path":str(f),"bytes":len(raw),"sha256":hashlib.sha256(raw).hexdigest()};atomic_json(r/"attempts.json",rows)
try:
 snapshot("initial")
 small={"model":"glm-52","messages":[{"role":"user","content":"List three properties of a correct inference service."}],"temperature":0,"seed":20260930,"ignore_eos":True,"max_tokens":32,"stream":True,"stream_options":{"include_usage":True}}
 for key in ["D0","D1"]:req("local_"+key,key,small,"public_D_localprefill",32)
 canonical=json.loads((r/"canonical_81932.body.json").read_text());canonical["max_tokens"]=64
 for n,pkey,dkey in [(0,"P0","D1"),(1,"P1","D0"),(2,"P1","D1"),(3,"P0","D0")]:
  before="pd"+str(n)+"_before";snapshot(before)
  pb=dict(canonical);pb.update(stream=False,max_tokens=1,min_tokens=1,kv_transfer_params={"do_remote_decode":True,"do_remote_prefill":False,"remote_engine_id":None,"remote_block_ids":None,"remote_host":None,"remote_port":None});pb.pop("stream_options",None)
  response=req("prefill_"+str(n),pkey,pb,"internal_P_KV_helper",1)
  kv=response.get("kv_transfer_params");assert isinstance(kv,dict)and kv.get("do_remote_prefill")and kv.get("remote_block_ids")and kv.get("remote_dcp_size")==16 and kv.get("remote_pcp_size")==1
  atomic_json(r/("transfer_metadata_"+str(n)+".json"),kv)
  db=dict(canonical);db["kv_transfer_params"]=kv;req("decode_"+str(n),dkey,db,"public_PD_decode",64)
  snapshot("pd"+str(n)+"_after")
 # InternalP maxlen restriction does notrestrict actualnativeD localprefill compatibility.
 large=dict(small);content=canonical["messages"][0]["content"];large["messages"]=[{"role":"user","content":content+content[:len(content)//5]}];req("D0_fullinput_local_fallback","D0",large,"public_D_input_exceeds_P_limit",32,min_prompt=81933)
 snapshot("final")
 public=[x for x in rows if x["kind"]!="internal_P_KV_helper"];helpers=[x for x in rows if x["kind"]=="internal_P_KV_helper"]
 assert len(public)==7 and all(x["completed"]for x in public)and len(helpers)==4 and all(x["completed"]for x in helpers)
 out={"at":utc(),"measurement_valid":True,"new_native_inference_attempts":len(rows),"new_native_completed_requests":len(rows),"native_committed_output_tokens":sum(x["usage"]["completion_tokens"]for x in rows),"public_completed_requests":len(public),"effective_public_output_tokens":sum(x["effective_public_output_credit"]for x in rows),"internal_P_helper_commits":sum(x["usage"]["completion_tokens"]for x in helpers),"requests":rows,"limits":["DirectnativeP/Ddiagnostic, no publicproxy routing/PDfullcompatibility/capacity/KEEP claim","Pinternalhelpertokens separatefrompubliceffectivecredits; allnativeattempts retained","ConfiguredDlocal fullinput request>81933 provesnativeconsumerfallback only, not publicgateway routing","Samecheckpoint/nativeoperators, differentparallel/memory/cache/SDKlifetime/control composition; noisolatedgain","Native KVbytes/shards/transfer completion verifiedby nativeinfo/counters/logsaudit pending"]}
 atomic_json(r/"pilot_summary.json",out);print(json.dumps({"public_requests":len(public),"public_output":out["effective_public_output_tokens"],"native_inferences":len(rows)}))
except BaseException as e:
 atomic_json(r/"pilot_summary.json",{"at":utc(),"measurement_valid":False,"error_type":type(e).__name__,"error":str(e),"requests":rows,"effective_public_output_tokens":sum(x["effective_public_output_credit"]for x in rows),"limits":["Allattempts/raw retained, not fullPDperformance/capacity verdict"]});raise
