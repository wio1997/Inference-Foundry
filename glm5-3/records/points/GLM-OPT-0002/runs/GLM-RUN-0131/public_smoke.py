from pathlib import Path
import sys,json,urllib.request,urllib.error,hashlib,subprocess,time
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc
from owner_guard import start_watchdog,guard
from vllm.entrypoints.openai.responses.protocol import ResponsesResponse
start_watchdog();r=Path(__file__).parent;http=urllib.request.build_opener(urllib.request.ProxyHandler({}));rows=[]
def req(name,method,path,body=None,status=200):
 guard();rawbody=None if body is None else json.dumps(body).encode()
 if body is not None:(r/(name+".body.json")).write_bytes(rawbody)
 request=urllib.request.Request("http://127.0.0.1:8000"+path,data=rawbody,method=method,headers={"Content-Type":"application/json"})
 start=time.monotonic()
 try:
  with http.open(request,timeout=120)as response:code=response.status;raw=response.read()
 except urllib.error.HTTPError as e:code=e.code;raw=e.read()
 f=r/(name+".wire");f.write_bytes(raw);rows.append(dict(at=utc(),name=name,method=method,path=path,status=code,wall_s=time.monotonic()-start,body_sha256=None if rawbody is None else hashlib.sha256(rawbody).hexdigest(),wire=dict(path=str(f),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())))
 atomic_json(r/"public_smoke_attempts.json",rows);assert code==status,(name,code);return json.loads(raw)
body=dict(model="glm-52",messages=[dict(role="user",content="List three properties of a correct inference service.")],max_tokens=32,ignore_eos=True,temperature=0,seed=20260930,cache_salt=r.name+"-public-chat",stream=False)
chat=req("public_chat","POST","/v1/chat/completions",body)
assert chat["usage"]["completion_tokens"]==32and chat["choices"][0]["finish_reason"]=="length"
basebody=dict(model="glm-52",input="Explain one concrete GLM serving example.",max_output_tokens=32,ignore_eos=True,temperature=0,seed=20260930,store=True,request_id="resp_glm_run131_base",cache_salt=r.name+"-response")
base=req("public_response","POST","/v1/responses",basebody);typed=ResponsesResponse.model_validate(base);assert typed.id=="resp_glm_run131_base"and typed.usage.output_tokens==32and typed.status=="incomplete"
assert req("public_retrieve","GET","/v1/responses/"+base["id"])==base
child=req("public_previous","POST","/v1/responses",dict(basebody,request_id="resp_glm_run131_child",previous_response_id=base["id"],max_output_tokens=16))
typed=ResponsesResponse.model_validate(child);assert typed.usage.output_tokens==16and typed.previous_response_id==base["id"]
# Physical engines were replaced; old bindings must fail before any native replay.
for ident in["resp_glm_run125_base","resp_glm_run125_other"]:
 req("retired_"+ident,"GET","/v1/responses/"+ident,status=503)
z=subprocess.run([sys.executable,str(r/"epoch_check.py"),"public_smoke_final"],capture_output=True,timeout=240);(r/"public_smoke_final.stdout").write_bytes(z.stdout);(r/"public_smoke_final.stderr").write_bytes(z.stderr);z.check_returncode()
placement=req("public_placement","GET","/control/replicas")
assert len(placement["replicas"])==1and placement["replicas"][0]["active_requests"]==0and not placement["replicas"][0]["group_faulted"]
atomic_json(r/"public_smoke_summary.json",dict(at=utc(),functional_acceptance=True,effective_output_tokens=80,new_inference_requests=3,old_bindings_retired_preRPC=True,typed_Responses=True,Chat32=True,previous_response=True,retained_public8000=True,native_NPU32_same=True,limits=["Smallpubliccontract smoke, fulltypedSSE/tools/background/nativeerror coverage stillneeds newphysicalclass execution","Physicalnative stores were replaced; oldbinding failclosed ratherthan replicate"]))
print("public coupled smoke80 valid")
