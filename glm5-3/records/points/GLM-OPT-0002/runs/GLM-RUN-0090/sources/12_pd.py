from pathlib import Path
import json,sys,time,hashlib,urllib.request,urllib.error,re,subprocess
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc
from sse_observer import NativeSSEObserver
from owner_guard import start_watchdog,guard
start_watchdog();r=Path(__file__).parent;owners=json.loads((r/"adopted_model_identities.json").read_text());opener=urllib.request.build_opener(urllib.request.ProxyHandler({}));rows=[]
def ref(f):
 b=f.read_bytes();return dict(path=str(f),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def snap(label):
 guard()
 for key,o in owners.items():
  with opener.open("http://172.16.10."+o["host"]+":"+str(o["port"])+"/metrics",timeout=10)as res:b=res.read();assert res.status==200
  (r/(label+"_"+key+".metrics")).write_bytes(b)
def rpc(label,key,path,body=None,expected_status=200,stream=False):
 guard();o=owners[key];f=r/(label+".body.json");raw=b"";t=time.monotonic();row=dict(name=label,owner=key,path=path,at=utc(),completed=False,effective_public_output_credit=0);rows.append(row)
 if body is not None:atomic_json(f,body);row["body"]=ref(f);data=f.read_bytes()
 else:data=None
 atomic_json(r/"attempts.json",rows);req=urllib.request.Request("http://172.16.10."+o["host"]+":"+str(o["port"])+path,data=data,headers={"Content-Type":"application/json","X-Request-ID":r.name+"-"+label})
 try:
  try:
   with opener.open(req,timeout=600)as res:
    status=res.status
    while True:
     b=res.read1(65536)
     if not b:break
     raw+=b
     if stream and "ttft_s"not in row:
      for frame in raw.replace(b"\r\n",b"\n").split(b"\n\n"):
       ds=b"\n".join(v[5:].removeprefix(b" ")for v in frame.splitlines()if v.startswith(b"data:"))
       try:x=json.loads(ds)
       except (ValueError,UnicodeDecodeError):continue
       if any(any((c.get("delta")or{}).get(k)for k in["content","reasoning","reasoning_content","tool_calls"])for c in x.get("choices",[])):row["ttft_s"]=time.monotonic()-t;break
  except urllib.error.HTTPError as e:status=e.code;raw=e.read()
  row.update(http_status=status,wall_s=time.monotonic()-t);assert status==expected_status,(label,status,raw[:400])
  if stream:
   ob=NativeSSEObserver(max_bytes=2097152,collect_contract=True);ob.feed(raw);v=ob.contract();assert v["done"]and not v["native_error"]and not v["unknown"]
  else:v=json.loads(raw)
  row.update(completed=True,response=v);return v,row
 except BaseException as e:row.update(error_type=type(e).__name__,error=str(e));raise
 finally:
  f=r/(label+".wire");f.write_bytes(raw);row["wire"]=ref(f);atomic_json(r/"attempts.json",rows)
def kvcheck(v,label):
 kv=v.get("kv_transfer_params");assert isinstance(kv,dict)and kv.get("do_remote_prefill")is True and kv.get("remote_block_ids")and kv.get("remote_dcp_size")==16 and kv.get("remote_pcp_size")==1
 assert kv["remote_host"]=="172.16.10.166"and int(kv["remote_port"])==28000 and kv["remote_engine_id"].startswith("GLM89-P-")
 atomic_json(r/(label+".metadata.json"),kv);return kv
def response_text(v):
 return "".join(c.get("text","")for x in v.get("output",[])for c in x.get("content",[])if c.get("type")=="output_text")
def response_valid(v,maximum,answer=None):
 u=v["usage"];assert type(u["input_tokens"])is int and u["input_tokens"]>128 and type(u["output_tokens"])is int and 0<u["output_tokens"]<=maximum and u["total_tokens"]==u["input_tokens"]+u["output_tokens"]
 if answer is not None:assert v["status"]=="completed"and response_text(v).strip()==answer,(v["status"],response_text(v))
 return u
def idle():
 for _ in range(100):
  snap("final");active=[]
  for key in owners:
   b=(r/("final_"+key+".metrics")).read_text();vals=[float(v)for v in re.findall(r"^vllm:num_requests_(?:running|waiting)(?:\{[^\n]*\})?\s+([0-9.eE+-]+)",b,re.M)];assert len(vals)>=2;active+=vals
  if all(x==0 for x in active):return
  time.sleep(.2)
 raise RuntimeError("NativePD terminalidle deadline")
remote=dict(do_remote_decode=True,do_remote_prefill=False,remote_engine_id=None,remote_block_ids=None,remote_host=None,remote_port=None)
try:
 snap("initial")
 canonical=json.loads((r/"canonical_81932.body.json").read_text());canonical.update(max_tokens=64,stream=True,stream_options=dict(include_usage=True),return_token_ids=True,cache_salt=r.name+"-NEW-cold-chat");canonical.pop("kv_transfer_params",None)
 begin=time.monotonic();helper=dict(canonical);helper.update(max_tokens=1,min_tokens=1,stream=False,kv_transfer_params=remote);helper.pop("stream_options",None)
 p,pr=rpc("chat_helper","D0","/v1/chat/completions",helper);assert p["usage"]["prompt_tokens"]==81932 and p["usage"]["completion_tokens"]==1;pr.update(usage=p["usage"],internal_helper=True)
 kv=kvcheck(p,"chat");snap("chat_helper_done");public=dict(canonical,kv_transfer_params=kv)
 d,dr=rpc("chat_public","D1","/v1/chat/completions",public,stream=True);assert d["usage"]==dict(prompt_tokens=81932,total_tokens=81996,completion_tokens=64)and d["finish_reasons"]=={"0":"length"};dr.update(usage=d["usage"],effective_public_output_credit=64,e2e_s=time.monotonic()-begin);snap("chat_done")
 input_text=("This is a padding line for the inference state test.\n"*1024)+"\nCompute 17 + 25. Reply with only the number."
 base=dict(model="glm-52",input=input_text,temperature=0,seed=20260930,max_output_tokens=32,stream=False,store=True,background=False,request_id="resp_glm_run90_base",cache_salt=r.name+"-NEW-responses",chat_template_kwargs=dict(enable_thinking=False))
 helper=dict(base,max_output_tokens=1,store=False,request_id="resp_glm_run90_helper",ignore_eos=True,kv_transfer_params=remote)
 begin=time.monotonic();p,pr=rpc("responses_helper","D0","/v1/responses",helper);pu=response_valid(p,1);assert pu["output_tokens"]==1;pr.update(usage=pu,internal_helper=True)
 kv=kvcheck(p,"responses");snap("responses_helper_done");public=dict(base,kv_transfer_params=kv)
 d,dr=rpc("responses_public","D1","/v1/responses",public);du=response_valid(d,32,"42");assert du["input_tokens"]==pu["input_tokens"]and d["id"]==base["request_id"];dr.update(usage=du,effective_public_output_credit=du["output_tokens"],e2e_s=time.monotonic()-begin);snap("responses_done")
 retrieved,rr=rpc("responses_retrieve","D1","/v1/responses/"+d["id"]);assert retrieved==d
 rpc("responses_wrong_store","D0","/v1/responses/"+d["id"],expected_status=404)
 rpc("helper_not_stored","D0","/v1/responses/"+helper["request_id"],expected_status=404)
 previous=dict(model="glm-52",input="What was the number in your previous answer? Reply only with that number.",previous_response_id=d["id"],temperature=0,seed=20260930,max_output_tokens=32,stream=False,store=True,request_id="resp_glm_run90_prev",cache_salt=r.name+"-previous",chat_template_kwargs=dict(enable_thinking=False))
 v,vr=rpc("responses_previous","D1","/v1/responses",previous);u=response_valid(v,32,"42");assert v["id"]==previous["request_id"];vr.update(usage=u,effective_public_output_credit=u["output_tokens"],local_stateful_D_request=True);idle()
 z=subprocess.run([sys.executable,str(r/"epoch_check.py")],capture_output=True,timeout=240);(r/"final_epoch.stdout").write_bytes(z.stdout);(r/"final_epoch.stderr").write_bytes(z.stderr);z.check_returncode()
 out=dict(at=utc(),functional_acceptance=True,measurement_valid=True,verdict="INCONCLUSIVE",effective_public_output_tokens=sum(x["effective_public_output_credit"]for x in rows),internal_helper_commits=2,requests=rows,PD_Chat_cold81932_64=True,PD_native_Responses_exact42=True,Responses_store_retrieve_previous42_wrongdomain404_helper_storeFalse404=True,API2_NPU32_same=True,limits=["DirectnativeP89/D88 diagnostic, no publicproxy/typedResponsesSSE/background/fullcompatibility/dynamiccapacity/KEEP claim","Allnativehelpercommits2 charged separately fromeffectivepublicoutputs; finiteE2E includesPhelper andDdecode, no isolatedtransfer bandwidth bound","Statefulprevious usesnativeDlocalcontinuation; no Phelper reconstruction or state replication; originalnativefield schemas/protocols/operators preserved"])
 atomic_json(r/"pd_summary.json",out);print(json.dumps({k:out[k]for k in["functional_acceptance","effective_public_output_tokens","internal_helper_commits"]}))
except BaseException as e:
 atomic_json(r/"pd_summary.json",dict(at=utc(),functional_acceptance=False,measurement_valid=False,error_type=type(e).__name__,error=str(e),requests=rows,effective_public_output_tokens=sum(x["effective_public_output_credit"]for x in rows),limits=["Rawallactualattempts preserved; no completePD/capacity claim, inspectnativecounts/modelidentities/currentresources before nextwork"]));raise
