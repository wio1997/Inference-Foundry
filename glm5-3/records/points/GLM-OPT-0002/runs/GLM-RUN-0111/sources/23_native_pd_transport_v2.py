"""Optional native P->D transport; native D remains protocol/state authority.

Selection is conservative; unsupported requests use the original native D path.
A selected request gets exactly one helper and one D dispatch, never a retry.
The helper output is internal cost; no helper bytes become user output.
"""
import asyncio,hashlib,json,time,uuid,os
from pathlib import Path
import httpx

REMOTE=dict(do_remote_decode=True,do_remote_prefill=False,remote_engine_id=None,
            remote_block_ids=None,remote_host=None,remote_port=None)
COMMON={"model","temperature","top_p","top_k","min_p","seed","stream","cache_salt",
        "chat_template_kwargs","presence_penalty","frequency_penalty","repetition_penalty",
        "ignore_eos","stop","stop_token_ids","logit_bias","request_id"}
CHAT=COMMON|{"messages","max_tokens","min_tokens","stream_options","return_token_ids"}
RESPONSES=COMMON|{"input","instructions","max_output_tokens","store","background"}
def helper_body(path,body,minimum_bytes):
    try:data=json.loads(body)
    except (ValueError,UnicodeError):return None
    if not isinstance(data,dict):return None
    allowed=CHAT if path=="/v1/chat/completions" else RESPONSES if path=="/v1/responses" else set()
    if not allowed or set(data)-allowed:return None
    if not isinstance(data.get("model"),str) or not data["model"]:return None
    if data.get("background",False) is not False:return None
    if path=="/v1/responses":
        prompt=data.get("input")
        if not isinstance(prompt,str):return None
        if "instructions"in data and not isinstance(data["instructions"],str):return None
        maximum=data.get("max_output_tokens")
    else:
        messages=data.get("messages")
        if not isinstance(messages,list)or not messages:return None
        if any(not isinstance(m,dict)or set(m)-{"role","content"}or m.get("role")not in {"system","user","assistant","developer"}or not isinstance(m.get("content"),str)for m in messages):return None
        prompt="".join(m["content"]for m in messages)
        maximum=data.get("max_tokens")
    if type(maximum)is not int or maximum<=0 or len(prompt.encode())<minimum_bytes:return None
    out=dict(data);out["stream"]=False;out["ignore_eos"]=True
    out["kv_transfer_params"]=dict(REMOTE)
    # Sampling/validation remains native. The helper is a separate internal commit.
    if path=="/v1/responses":
        out.update(max_output_tokens=1,store=False,background=False,
                   request_id="resp_glm_pdhelper_"+uuid.uuid4().hex)
    else:
        out.update(max_tokens=1,min_tokens=1)
        out.pop("stream_options",None)
        # Caller identity belongs to D; P helper has a unique private identity.
        if "request_id"in out:out["request_id"]="glm_pdhelper_"+uuid.uuid4().hex
    return data,out

def valid_metadata(value,producer):
    kv=value.get("kv_transfer_params")if isinstance(value,dict)else None
    if not isinstance(kv,dict)or kv.get("do_remote_prefill")is not True:return False
    if kv.get("remote_host")!=producer["remote_host"]or kv.get("remote_port")!=producer["remote_port"]:return False
    if not isinstance(kv.get("remote_engine_id"),str)or not kv["remote_engine_id"]:return False
    if kv.get("remote_dcp_size")!=producer["dcp_size"]or kv.get("remote_pcp_size")!=producer.get("pcp_size",1):return False
    blocks=kv.get("remote_block_ids")
    return isinstance(blocks,list)and bool(blocks)and all(isinstance(row,list)and row and all(type(x)is int and x>=0 for x in row)for row in blocks)

class NativePDTransport(httpx.AsyncBaseTransport):
    def __init__(self,producers,base=None,trace=None,minimum_bytes=4096,helper_timeout=600,audit_dir=None):
        self.base=base or httpx.AsyncHTTPTransport()
        self.producers={k.rstrip("/"):dict(v)for k,v in producers.items()}
        self.trace=trace or (lambda event,**fields:None)
        self.minimum_bytes=minimum_bytes;self.helper_timeout=helper_timeout
        self.audit_dir=Path(audit_dir)if audit_dir else None
        if self.audit_dir:self.audit_dir.mkdir(parents=True,exist_ok=True)
        if type(minimum_bytes)is not int or minimum_bytes<0:raise ValueError("minimum_bytes")
        for origin,p in self.producers.items():
            if not p.get("url")or not p.get("remote_host")or type(p.get("remote_port"))is not int or type(p.get("dcp_size"))is not int:raise ValueError("producer mapping")
    def artifact(self,operation,label,raw):
        if self.audit_dir is None:return None
        p=self.audit_dir/(operation+"."+label)
        try:
            fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
            with os.fdopen(fd,"wb")as stream:stream.write(raw)
            return dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
        except OSError as error:return dict(path=str(p),audit_error=str(error))
    async def handle_async_request(self,request):
        origin=str(request.url.copy_with(path="/",query=None,fragment=None)).rstrip("/")
        producer=self.producers.get(origin)
        if producer is None or request.method!="POST":return await self.base.handle_async_request(request)
        body=await request.aread();selected=helper_body(request.url.path,body,self.minimum_bytes)
        if selected is None:return await self.base.handle_async_request(request)
        data,helper=selected;operation=uuid.uuid4().hex;started=time.monotonic()
        headers=[(k,v)for k,v in request.headers.raw if k.lower()not in {b"host",b"content-length",b"x-request-id"}]
        headers.append((b"x-request-id",("glm-pdhelper-"+operation).encode()))
        helper_raw=json.dumps(helper,separators=(",",":"),ensure_ascii=False).encode()
        hp=httpx.Request("POST",producer["url"].rstrip("/")+request.url.path,headers=headers,content=helper_raw,extensions=dict(request.extensions))
        self.trace("pd_helper_started",operation=operation,producer=producer["url"],decoder=origin,
                   original_body_sha256=hashlib.sha256(body).hexdigest(),helper_body_sha256=hashlib.sha256(helper_raw).hexdigest(),
                   original_body_artifact=self.artifact(operation,"original.body",body),
                   helper_body_artifact=self.artifact(operation,"helper.body",helper_raw))
        response=None
        try:
            async def fetch():
                nonlocal response
                response=await self.base.handle_async_request(hp)
                raw=await response.aread()
                self.trace("pd_helper_wire",operation=operation,status=response.status_code,artifact=self.artifact(operation,"helper.wire",raw))
                if response.status_code in {400,422}:return None,raw
                if response.status_code!=200:raise ValueError("native helper HTTP "+str(response.status_code))
                value=json.loads(raw)
                if not valid_metadata(value,producer):raise ValueError("native helper KV metadata")
                usage=value.get("usage")or{}
                output=usage.get("output_tokens"if request.url.path=="/v1/responses"else"completion_tokens")
                if output!=1:raise ValueError("native helper must commit exactly one internal token")
                return value,raw
            value,raw=await asyncio.wait_for(fetch(),self.helper_timeout)
        except asyncio.CancelledError:
            self.trace("pd_helper_cancelled",operation=operation,helper_wall_s=time.monotonic()-started)
            raise
        except (httpx.HTTPError,ValueError,TypeError,KeyError,asyncio.TimeoutError)as error:
            self.trace("pd_helper_failed",operation=operation,error_type=type(error).__name__,error=str(error),
                       helper_wall_s=time.monotonic()-started,KV_release="native lifecycle; unknown after failed helper")
            content=json.dumps({"error":{"type":"pd_helper_error","message":str(error)}}).encode()
            return httpx.Response(502,headers={"content-type":"application/json"},stream=httpx.ByteStream(content),request=request)
        finally:
            if response is not None:await response.aclose()
        if value is None:
            # A helper validation failure cannot replace the public native contract.
            # Original D dispatch occurs once, before any D response bytes exist.
            self.trace("pd_helper_native_rejection",operation=operation,helper_status=response.status_code,
                       helper_wall_s=time.monotonic()-started,public_output_credit=0)
            self.trace("pd_native_fallback_dispatch",operation=operation,
                       original_body_sha256=hashlib.sha256(body).hexdigest())
            return await self.base.handle_async_request(request)
        self.trace("pd_helper_completed",operation=operation,helper_wall_s=time.monotonic()-started,
                   internal_output_tokens=1,usage=value["usage"],kv_transfer_params=value["kv_transfer_params"],
                   helper_wire_bytes=len(raw),helper_wire_sha256=hashlib.sha256(raw).hexdigest())
        native=dict(data,kv_transfer_params=value["kv_transfer_params"])
        native_raw=json.dumps(native,separators=(",",":"),ensure_ascii=False).encode()
        headers=[(k,v)for k,v in request.headers.raw if k.lower()not in {b"host",b"content-length"}]
        outgoing=httpx.Request(request.method,request.url,headers=headers,content=native_raw,extensions=dict(request.extensions))
        self.trace("pd_decoder_dispatch",operation=operation,native_body_sha256=hashlib.sha256(native_raw).hexdigest(),
                   public_output_budget=data.get("max_tokens",data.get("max_output_tokens")),
                   native_body_artifact=self.artifact(operation,"native.body",native_raw),
                   helper_wall_s=time.monotonic()-started)
        # The native response stream, headers, usage and IDs are returned untouched.
        return await self.base.handle_async_request(outgoing)
    async def aclose(self):await self.base.aclose()
