"""Transparent complete-request GLM routing; explicit experimental deployment.

Engines remain the state/KV/MTP owners. Payloads, output bytes, native error
status and sampling parameters pass through. No post-first-byte retry occurs.
The current implementation has one scheduling process; launch one worker.
"""
import argparse,asyncio,hashlib,json,os,time
from contextlib import asynccontextmanager
from placement import Placement,estimate_request
from sse_observer import NativeSSEObserver

def forwarded_headers(headers):
    extra=set()
    for key,value in headers:
        if key.lower()==b'connection':extra.update(x.strip().lower() for x in value.split(b','))
    return [(key,value) for key,value in headers if key.lower() not in HOP|extra]

HOP={b'host',b'connection',b'keep-alive',b'proxy-authenticate',b'proxy-authorization',b'te',b'trailer',b'transfer-encoding',b'upgrade'}

def create_app(config=None,transport=None):
    import httpx
    from fastapi import FastAPI,Request
    from fastapi.responses import JSONResponse,StreamingResponse
    replicas=config if config is not None else json.loads(os.environ['GLM_REPLICAS'])
    placement=Placement(replicas)
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
        if trace_fd is not None:os.close(trace_fd);trace_fd=None
    app=FastAPI(lifespan=lifespan);app.state.placement=placement
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
        try:lease=await placement.acquire(budget,size)
        except RuntimeError as error:return JSONResponse({'error':str(error)},status_code=503)
        trace('lease_acquired',lease,path=request.url.path,method=request.method,output_budget=budget,input_bytes=size,body_sha256=hashlib.sha256(body).hexdigest() if trace_fd is not None else None)
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
        async def stream():
            failure=response.status_code>=500;first=True
            observe=NativeSSEObserver() if response.headers.get("content-type","").split(";")[0].strip().lower()=="text/event-stream" and response.headers.get("content-encoding","identity")=="identity" else None
            try:
                async for block in response.aiter_raw():
                    if first and block:trace('upstream_first_bytes',lease,bytes=len(block));first=False
                    if observe is not None and observe.feed(block):
                        failure=True
                        trace('upstream_native_error',lease,classification='server_error_in_sse')
                    yield block
            except httpx.HTTPError:failure=True;raise
            finally:await close(failure)
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
    uvicorn.run('replica_gateway:create_app',host=a.host,port=a.port,workers=1,factory=True,app_dir=os.path.dirname(__file__))
