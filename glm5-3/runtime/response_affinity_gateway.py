"""Native Responses owner affinity plus complete-request GLM routing; opt-in experiment.

Engines remain the state/KV/MTP owners. Payloads, output bytes, native error
status and sampling parameters pass through. No post-first-byte retry occurs.
The current implementation has one scheduling process; launch one worker.
"""
import argparse,asyncio,hashlib,json,os,time
from contextlib import asynccontextmanager
from placement import estimate_request
from response_affinity import ResponseAffinityPlacement,ResponseOwnerIndex,ResponsesOwnerObserver,request_owner_key
from sse_observer import NativeSSEObserver

def forwarded_headers(headers):
    extra=set()
    for key,value in headers:
        if key.lower()==b'connection':extra.update(x.strip().lower() for x in value.split(b','))
    return [(key,value) for key,value in headers if key.lower() not in HOP|extra]

HOP={b'host',b'connection',b'keep-alive',b'proxy-authenticate',b'proxy-authorization',b'te',b'trailer',b'transfer-encoding',b'upgrade'}

def create_app(config=None,transport=None,policy=None,groups=None,fault_state_path=None,response_owner_state_path=None):
    import httpx
    from fastapi import FastAPI,Request
    from fastapi.responses import JSONResponse,StreamingResponse
    audit_dir=os.environ.get('GLM_ROUTER_AUDIT_DIR')
    if audit_dir:os.makedirs(audit_dir,exist_ok=True)
    replicas=config if config is not None else json.loads(os.environ['GLM_REPLICAS'])
    response_owner_state_path=response_owner_state_path or os.environ.get("GLM_RESPONSE_OWNER_STATE_PATH")
    if response_owner_state_path and not fault_state_path and not os.environ.get("GLM_GROUP_FAULT_STATE_PATH"):fault_state_path=response_owner_state_path+".fault"
    placement=ResponseAffinityPlacement(replicas,policy or os.environ.get("GLM_PLACEMENT_POLICY","active_count"),groups if groups is not None else json.loads(os.environ.get("GLM_EXECUTION_GROUPS","[]")),fault_state_path or os.environ.get("GLM_GROUP_FAULT_STATE_PATH"))
    owner_enabled=response_owner_state_path is not None
    try:owner_index=ResponseOwnerIndex(placement,response_owner_state_path)
    except BaseException:
        if placement.journal_fd is not None:os.close(placement.journal_fd);placement.journal_fd=None
        raise
    if response_owner_state_path and any(key not in placement.member_group for key in placement.replicas):
        owner_index.close();raise ValueError("every stored-response endpoint requires declared native owner epoch")
    trace_fd=None
    def trace(event,lease=None,**fields):
        if trace_fd is None:return
        if lease:fields.update(lease_id=lease.lease_id,replica=lease.replica.key,generation=lease.replica.generation)
        try:os.write(trace_fd,(json.dumps({'event':event,'monotonic_ns':time.monotonic_ns(),'pid':os.getpid(),**fields},separators=(',',':'))+'\n').encode())
        except OSError:pass
    @asynccontextmanager
    async def lifespan(app):
        nonlocal trace_fd
        trace_path=os.environ.get('GLM_ROUTER_TRACE_PATH')
        if trace_path:trace_fd=os.open(trace_path,os.O_CREAT|os.O_APPEND|os.O_WRONLY,0o600)
        app.state.cleanup_tasks=set()
        app.state.client=httpx.AsyncClient(timeout=httpx.Timeout(connect=10,read=None,write=30,pool=30),trust_env=False,transport=transport)
        yield
        if app.state.cleanup_tasks:await asyncio.gather(*app.state.cleanup_tasks,return_exceptions=True)
        await app.state.client.aclose()
        await placement.close()
        owner_index.close()
        if trace_fd is not None:os.close(trace_fd);trace_fd=None
    app=FastAPI(lifespan=lifespan);app.state.placement=placement;app.state.response_owner_index=owner_index
    @app.get('/healthcheck')
    async def healthcheck():
        entries=await placement.snapshot();health=[]
        for replica in entries:
            try:
                response=await app.state.client.get(replica['url']+'/health',timeout=3);ready=response.status_code==200
            except httpx.HTTPError:ready=False
            health.append({'id':replica['id'],'ready':ready,'draining':replica['draining']})
        available=any(x['ready'] and not x['draining'] and not r['temporarily_unhealthy'] for x,r in zip(health,entries))
        return JSONResponse({'status':'ok' if available else 'unavailable','request_num':len(placement.leases),'replicas':health},status_code=200 if available else 503)
    @app.get('/control/replicas')
    async def inspect():return {'replicas':await placement.snapshot()}
    @app.post('/control/replicas')
    async def add(request:Request):
        try:await placement.add(await request.json());return {'replicas':await placement.snapshot()}
        except (ValueError,KeyError,TypeError) as error:return JSONResponse({'error':str(error)},status_code=409)
    @app.delete('/control/replicas/{key}')
    async def remove(key:str):
        try:await placement.remove(key);return {'replicas':await placement.snapshot()}
        except KeyError:return JSONResponse({'error':'unknown replica'},status_code=404)
    @app.api_route('/{path:path}',methods=['GET','POST','PUT','PATCH','DELETE','OPTIONS','HEAD'])
    async def forward(path:str,request:Request):
        body=await request.body();budget,size=estimate_request(body)
        owner_key=request_owner_key(request.url.path,request.method,body)
        binding=owner_index.resolve(owner_key) if owner_enabled and owner_key is not None else None
        creates_response=owner_enabled and request.method=='POST' and request.url.path=='/v1/responses'
        if creates_response and owner_index.error:return JSONResponse({'error':'response owner journal unavailable'},status_code=503)
        try:lease=await placement.acquire_for_owner(budget,size,binding)
        except RuntimeError as error:return JSONResponse({'error':str(error)},status_code=503)
        trace('lease_acquired',lease,path=request.url.path,method=request.method,output_budget=budget,input_bytes=size,body_sha256=hashlib.sha256(body).hexdigest() if trace_fd is not None else None,request_header_id=request.headers.get("x-request-id"),response_owner_key=owner_key,affinity_applied=binding is not None)
        response=None;cleanup_task=None
        async def close(failure=False):
            nonlocal cleanup_task
            if cleanup_task is None:
                async def cleanup():
                    try:
                        if response is not None:await response.aclose()
                    finally:
                        released=await placement.release(lease,backend_failure=failure)
                        trace('lease_released',lease,backend_failure=failure,released=released)
                cleanup_task=asyncio.create_task(cleanup())
                app.state.cleanup_tasks.add(cleanup_task)
                cleanup_task.add_done_callback(app.state.cleanup_tasks.discard)
            await asyncio.shield(cleanup_task)
        headers=forwarded_headers(request.headers.raw)
        try:
            raw_path=request.scope.get('raw_path',('/'+path).encode())
            native_path=raw_path.decode('ascii') if raw_path.isascii() else '/'+path
            url=lease.replica.url+native_path
            if request.url.query:url+='?'+request.url.query
            outgoing=app.state.client.build_request(request.method,url,content=body,headers=headers)
            response=await app.state.client.send(outgoing,stream=True)
            trace('upstream_headers',lease,status=response.status_code)
        except asyncio.CancelledError:await close();raise
        except httpx.HTTPError as error:
            await close(True);return JSONResponse({'error':{'type':'upstream_error','message':str(error)}},status_code=502)
        except BaseException:
            await close();raise
        buffered=None
        content_type=response.headers.get("content-type","").split(";")[0].strip().lower()
        owner_observer=ResponsesOwnerObserver() if creates_response and response.status_code<400 and content_type=="text/event-stream" else None
        owner_decoder=response._get_content_decoder() if owner_observer is not None else None
        if creates_response and response.status_code<400 and content_type=="application/json":
            try:
                # Nonstream Responses already complete natively; retain raw encoding,
                # then persist native owner before any response ID reaches the caller.
                buffered=b"".join([block async for block in response.aiter_raw()])
                obj=httpx.Response(response.status_code,headers=response.headers,content=buffered).json()
                if isinstance(obj,dict)and obj.get("object")=="response"and isinstance(obj.get("id"),str):
                    bound=owner_index.bind(obj["id"],lease)
                    trace("response_owner_bound",lease,response_id=obj["id"],new_binding=bound)
            except asyncio.CancelledError:await close();raise
            except httpx.HTTPError as error:
                await close(True);return JSONResponse({'error':{'type':'upstream_error','message':str(error)}},status_code=502)
            except OSError:
                await close();return JSONResponse({'error':'response owner journal unavailable'},status_code=503)
            except BaseException:
                await close();raise
        async def raw_blocks():
            if buffered is not None:
                yield buffered
            else:
                async for block in response.aiter_raw():yield block
        async def stream():
            failure=response.status_code>=500;first=True;output_observed=False
            observe=NativeSSEObserver(collect_contract=bool(audit_dir)) if response.headers.get("content-type","").split(";")[0].strip().lower()=="text/event-stream" and response.headers.get("content-encoding","identity")=="identity" else None
            audit_file=None;audit_path=None;audit_error=None;wire_bytes=0;wire_hash=hashlib.sha256() if audit_dir else None
            if audit_dir:
                audit_path=os.path.join(audit_dir,lease.lease_id+'.sse')
                try:audit_file=open(audit_path,'xb')
                except OSError as error:audit_error=str(error)
            try:
                async for block in raw_blocks():
                    if owner_observer is not None:
                        for response_id in owner_observer.feed(owner_decoder.decode(block)):
                            bound=owner_index.bind(response_id,lease)
                            trace('response_owner_bound',lease,response_id=response_id,new_binding=bound)
                        if owner_observer.error:failure=True
                    if wire_hash is not None:
                        wire_hash.update(block);wire_bytes+=len(block)
                        if audit_file is not None:
                            try:audit_file.write(block)
                            except OSError as error:
                                audit_error=str(error)
                                try:audit_file.close()
                                except OSError:pass
                                audit_file=None
                    if first and block:trace('upstream_first_bytes',lease,bytes=len(block));first=False
                    if observe is not None and observe.feed(block):
                        failure=True
                        trace('upstream_native_error',lease,classification='server_error_in_sse')
                    if observe is not None and observe.output_started and not output_observed:
                        output_observed=True
                        transitioned=await placement.output_started(lease)
                        trace('upstream_first_output',lease,prefill_transitioned=transitioned)
                    yield block
            except httpx.HTTPError:failure=True;raise
            finally:
                if audit_file is not None:
                    try:audit_file.close()
                    except OSError as error:audit_error=str(error)
                if audit_dir:
                    contract=observe.contract() if observe is not None else {'unknown':True}
                    trace('upstream_stream_contract',lease,contract=contract,wire_path=audit_path,wire_bytes=wire_bytes,wire_sha256=wire_hash.hexdigest(),audit_error=audit_error)
                await close(failure)
        # Preserve duplicate end-to-end headers and raw encoding unchanged.
        class LeasedResponse(StreamingResponse):
            async def __call__(self,scope,receive,send):
                try:return await super().__call__(scope,receive,send)
                finally:await close(response.status_code>=500)
        result=LeasedResponse(stream(),status_code=response.status_code)
        result.raw_headers=forwarded_headers(response.headers.raw)
        return result
    return app

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--host',default='127.0.0.1');p.add_argument('--port',type=int,default=8002);a=p.parse_args()
    import uvicorn
    uvicorn.run('response_affinity_gateway:create_app',host=a.host,port=a.port,workers=1,factory=True,app_dir=os.path.dirname(__file__))
