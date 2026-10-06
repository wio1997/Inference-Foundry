from pathlib import Path
import sys,os,asyncio,json,hashlib
j=Path(__file__).parent;b=j/"prototype_bundle";sys.path.insert(0,str(b))
for k in list(os.environ):
 if k.startswith("GLM_"):os.environ.pop(k)
import httpx
from response_affinity_gateway_v11 import create_app
def ref(f):
 raw=Path(f).read_bytes();return dict(path=str(f),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
records=[]
async def native(request):
 raw=await request.aread();records.append(dict(host=request.url.host,path=request.url.path,method=request.method,body=raw.hex()))
 try:v=json.loads(raw)if raw else {}
 except ValueError:v={}
 if v.get("CPU_error"):
  return httpx.Response(v["CPU_error"],stream=httpx.ByteStream(b'{"error":{"message":"native mock error"}}'),headers={"content-type":"application/json"})
 if request.url.path.startswith("/v1/responses"):
  ident=(v.get("request_id")or "CPU_native_new_"+str(len(records)))if request.method=="POST"and request.url.path=="/v1/responses"else request.url.path.rsplit("/",1)[-1]
  wire=json.dumps(dict(object="response",id=ident,status="completed",output=[])).encode()
 elif v.get("stream"):
  wire=b'data: {"choices":[{"delta":{"content":"CPU"},"token_ids":[7],"finish_reason":null}]}\n\ndata: {"choices":[{"delta":{},"token_ids":[],"finish_reason":"stop"}],"usage":{"completion_tokens":1}}\n\ndata: [DONE]\n\n'
  return httpx.Response(200,stream=httpx.ByteStream(wire),headers={"content-type":"text/event-stream","x-native":"CPU"})
 else:wire=b'{"choices":[{"message":{"content":"CPU"},"token_ids":[7],"finish_reason":"stop"}]}'
 return httpx.Response(200,stream=httpx.ByteStream(wire),headers={"content-type":"application/json","x-native":"CPU"})
replicas=[dict(id="D0",url="http://v2.invalid"),dict(id="D1",url="http://v1.invalid")]
groups=[dict(id="V2",epoch="a"*64,members=["D0"]),dict(id="V1",epoch="b"*64,members=["D1"])]
shape=dict(input_threshold_bytes=512,prefill_members=["D1"],decode_members=["D0"])
compat={k:["D1"]for k in ["structured_output","thinking_budget","response_lifecycle"]}
checks=[]
async def main():
 app=create_app(config=replicas,transport=httpx.MockTransport(native),policy="shape_split_idle_spill",groups=groups,
 response_owner_state_path=str(j/"CPU_owners.json"),shape_config=shape,compatibility_config=compat)
 async with app.router.lifespan_context(app):
  async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url="http://gateway.invalid")as c:
   async def send(path,value,host,code=200):
    raw=json.dumps(value,separators=(",",":")).encode();n=len(records)
    out=await c.post(path,content=raw,headers={"content-type":"application/json","x-request-id":"CPU"})
    assert out.status_code==code,(out.status_code,out.text)
    if host is None:assert len(records)==n
    else:assert len(records)==n+1 and records[-1]["host"]==host and bytes.fromhex(records[-1]["body"])==raw
    assert not app.state.placement.leases
    return out
   out=await send("/v1/chat/completions",dict(tool_choice="required",stream=True),"v1.invalid")
   assert out.content.endswith(b"data: [DONE]\n\n")and out.headers["x-native"]=="CPU"
   await send("/v1/chat/completions",dict(tool_choice="auto"),"v2.invalid")
   await send("/v1/chat/completions",dict(thinking_token_budget=True,CPU_error=400),"v1.invalid",400)
   rows=(await c.get("/control/replicas")).json()["replicas"];assert not any(x["group_faulted"]for x in rows)
   legacy=await app.state.placement.acquire_for_owner(8,1)
   assert legacy.replica.key=="D0";app.state.response_owner_index.bind("CPU_legacy_v2",legacy);await app.state.placement.release(legacy,False)
   await send("/v1/responses",dict(request_id="CPU_v2"),"v1.invalid")
   await send("/v1/responses",dict(previous_response_id="CPU_v2",tool_choice="required"),"v1.invalid")
   await send("/v1/responses",dict(previous_response_id="CPU_legacy_v2",tool_choice="required"),None,503)
   await send("/v1/responses",dict(previous_response_id="CPU_legacy_v2",thinking_token_budget=None),"v2.invalid")
   await send("/v1/responses",dict(previous_response_id="CPU_legacy_v2",thinking_token_budget=-1),"v2.invalid")
   await send("/v1/responses",dict(request_id="CPU_v1",tool_choice="required"),"v1.invalid")
   await send("/v1/responses",dict(previous_response_id="CPU_v1",request_id="CPU_v1_next",tool_choice="required"),"v1.invalid")
   n=len(records);out=await c.get("/v1/responses/CPU_v1");assert out.status_code==200 and len(records)==n+1 and records[-1]["host"]=="v1.invalid"
   await c.delete("/control/replicas/D1")
   await send("/v1/chat/completions",dict(tool_choice="required"),None,503)
   await c.post("/control/replicas",json=replicas[1])
   await send("/v1/chat/completions",dict(tool_choice="required",CPU_error=500),"v1.invalid",500)
   rows=(await c.get("/control/replicas")).json()["replicas"];assert next(x for x in rows if x["id"]=="D1")["group_faulted"]
   await send("/v1/chat/completions",dict(tool_choice="required"),None,503)
 checks.extend(["ASGI structured SSE native bytes/header unchanged and lease released","plain shape choice intact; invalid thinking field/native400 remains native, no quarantine","native STORE owner affinity persists and V2 structured followup fails before RPC","drain/full-group500 excludes V1 with no incompatible replay"])
asyncio.run(main())
f=j/"CPU_native_mock_trace.json";f.write_text(json.dumps(records,indent=2)+"\n")
out=dict(CPU_gateway_VALID=True,checks=checks,mocked_upstream_calls=len(records),actual_native_inference_or_SDK_calls=0,body_exact=True,CPU_only=True,active_runtime_changed=False,sources=[ref(b/n)for n in ["capability_placement.py","response_affinity_gateway_v11.py","native_engines_service_config.py"]],limits=["ASGI/mock protocol evidence only; not native feature/capacity proof","All new Responses begin in V1; existing native owners never migrate, their true feature requirements still determine compatibility. Native fullAPI and cost remain E2E pending."])
f=j/"reduction.json";f.write_text(json.dumps(out,indent=2)+"\n")
v=dict(schema_version=1,job_id=j.name,status="completed",summary="V14 CPU ASGI/mock lifecycle compatibility wire/error/owner/drain/group-fault checks passed, no native calls",execution=dict(inner_exit_code=0,acceptance="passed",processes=[]),findings=[],evidence=[dict(id="reduction",**ref(f),locator="CPU ASGI/mock protocol")],unknowns=out["limits"],decision_request=None,next_check_at=None);(j/"result.json").write_text(json.dumps(v,indent=2)+"\n");print(json.dumps(dict(CPU_VALID=True,calls=len(records))))
