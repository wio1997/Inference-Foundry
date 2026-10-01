"""Transparent complete-request GLM routing; explicit experimental deployment.

Engines remain the state/KV/MTP owners. Payloads, output bytes, native error
status and sampling parameters pass through. No post-first-byte retry occurs.
The current implementation has one scheduling process; launch one worker.
"""
import argparse,asyncio,json,os
from contextlib import asynccontextmanager
from placement import Placement,estimate_request

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
    @asynccontextmanager
    async def lifespan(app):
        app.state.cleanup_tasks=set()
        app.state.client=httpx.AsyncClient(timeout=httpx.Timeout(connect=10,read=None,write=30,pool=30),trust_env=False,transport=transport)
        yield
        if app.state.cleanup_tasks:await asyncio.gather(*app.state.cleanup_tasks,return_exceptions=True)
        await app.state.client.aclose()
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
        response=None;cleanup_task=None
        async def close(failure=False):
            nonlocal cleanup_task
            if cleanup_task is None:
                async def cleanup():
                    try:
                        if response is not None:await response.aclose()
                    finally:await placement.release(lease,backend_failure=failure)
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
        except asyncio.CancelledError:await close();raise
        except httpx.HTTPError as error:
            await close(True);return JSONResponse({'error':{'type':'upstream_error','message':str(error)}},status_code=502)
        except BaseException:
            await close();raise
        async def stream():
            failure=response.status_code>=500
            try:
                async for block in response.aiter_raw():yield block
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
