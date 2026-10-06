import asyncio,json,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'runtime'))
from placement import Placement,Lease,estimate_request
from replica_gateway import create_app
import httpx

class Stream(httpx.AsyncByteStream):
    def __init__(self,chunks):self.chunks=chunks;self.closed=False
    async def __aiter__(self):
        for chunk in self.chunks:yield chunk
    async def aclose(self):self.closed=True

class PlacementContracts(unittest.IsolatedAsyncioTestCase):
    async def test_uncalibrated_spread_and_exact_release(self):
        p=Placement([{'id':'P','url':'http://p'},{'id':'D','url':'http://d'}]);a=await p.acquire(100,20);b=await p.acquire(1,20)
        self.assertNotEqual(a.replica.key,b.replica.key);self.assertTrue(await p.release(a));self.assertFalse(await p.release(a));await p.release(b);self.assertEqual(len(p.leases),0)
    async def test_drain_generation_and_forged_lease(self):
        p=Placement([{'id':'P','url':'http://p'}]);lease=await p.acquire(100,20);fake=Lease(lease.lease_id,lease.replica,100,20)
        with self.assertRaises(ValueError):await p.release(fake)
        self.assertIs(p.leases[lease.lease_id],lease)
        await p.remove('P')
        with self.assertRaises(RuntimeError):await p.acquire(1,1)
        with self.assertRaises(ValueError):await p.add({'id':'P','url':'http://p'})
        await p.release(lease);await p.add({'id':'P','url':'http://p'});self.assertGreater(p.replicas['P'].generation,lease.replica.generation)
        self.assertFalse(await p.release(lease));self.assertEqual(len(p.replicas['P'].active),0)
    async def test_mixed_calibration_never_mixes_units(self):
        p=Placement([{'id':'P','url':'http://p','decode_tps':10000},{'id':'D','url':'http://d'}]);a=await p.acquire(1,1);b=await p.acquire(10000,1);self.assertNotEqual(a.replica.key,b.replica.key)
        await p.release(a);await p.release(b)
    async def test_calibrated_outstanding_work(self):
        p=Placement([{'id':'P','url':'http://p','decode_tps':10},{'id':'D','url':'http://d','decode_tps':10}]);a=await p.acquire(1000,1);b=await p.acquire(1,1);c=await p.acquire(1,1);self.assertEqual(b.replica.key,c.replica.key)
        for lease in [a,b,c]:await p.release(lease)
    def test_sampling_hints_do_not_mutate_payload(self):
        body=b'{"max_completion_tokens":32,"n":2,"tools":[{"type":"function"}],"stream":true}';self.assertEqual(estimate_request(body),(64,len(body)));self.assertEqual(estimate_request(b'{oops'),(16,5))

class ForwardingContracts(unittest.IsolatedAsyncioTestCase):
    async def exercise(self,status,chunks,body):
        observed=[];stream=Stream(chunks)
        async def upstream(request):
            observed.append(request);self.assertEqual(await request.aread(),body)
            return httpx.Response(status,stream=stream,headers=[(b'content-type',b'text/event-stream'),(b'x-native',b'value'),(b'set-cookie',b'a=1'),(b'set-cookie',b'b=2')])
        app=create_app([{'id':'P','url':'http://p'}],httpx.MockTransport(upstream))
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://gateway') as client:
                response=await client.post('/v1/chat/completions?feature=1',content=body,headers={'content-type':'application/json','x-request-id':'request','connection':'x-hop','x-hop':'remove'})
            self.assertEqual(response.status_code,status);self.assertEqual(response.content,b''.join(chunks));self.assertEqual(response.headers.get_list('set-cookie'),['a=1','b=2']);self.assertTrue(stream.closed);self.assertEqual(len(app.state.placement.leases),0)
            self.assertEqual(observed[0].url.query,b'feature=1');self.assertEqual(observed[0].headers['x-request-id'],'request');self.assertNotIn('x-hop',observed[0].headers);self.assertEqual(len(observed),1)
    async def test_stream_and_all_payload_fields(self):
        await self.exercise(200,[b'data: {"choices":',b'[]}\n\ndata: [DONE]\n\n'],b'{"model":"glm-53","messages":[],"tools":[],"n":2,"seed":42,"stream":true,"top_p":0.9}')
    async def test_native_400_not_retried(self):
        await self.exercise(400,[b'{"error":"maximum context length"}'],b'{"max_tokens":999999}')
    async def test_header_disconnect_releases_unstarted_stream(self):
        stream=Stream([b'unstarted'])
        async def upstream(request):return httpx.Response(200,stream=stream)
        app=create_app([{'id':'P','url':'http://p'}],httpx.MockTransport(upstream));messages=[{'type':'http.request','body':b'{}','more_body':False}]
        async def receive():
            if messages:return messages.pop(0)
            await asyncio.sleep(60)
        async def send(message):
            if message['type']=='http.response.start':raise asyncio.CancelledError()
        scope={'type':'http','asgi':{'version':'3.0','spec_version':'2.4'},'http_version':'1.1','method':'POST','scheme':'http','path':'/v1/chat/completions','raw_path':b'/v1/chat/completions','query_string':b'','headers':[],'client':('127.0.0.1',1),'server':('127.0.0.1',2)}
        async with app.router.lifespan_context(app):
            with self.assertRaises(asyncio.CancelledError):await app(scope,receive,send)
            if app.state.cleanup_tasks:await asyncio.gather(*app.state.cleanup_tasks)
            self.assertEqual(len(app.state.placement.leases),0);self.assertTrue(stream.closed)
    async def test_transport_failure_cleans_and_avoids_failed_replica(self):
        called=[]
        async def upstream(request):
            called.append(request.url.host)
            if request.url.host=='p':raise httpx.ConnectError('fixture failure',request=request)
            return httpx.Response(200,stream=Stream([b'ok']))
        app=create_app([{'id':'P','url':'http://p'},{'id':'D','url':'http://d'}],httpx.MockTransport(upstream))
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://gateway') as client:
                first=await client.post('/v1/completions',content=b'{}');second=await client.post('/v1/completions',content=b'{}')
            self.assertEqual(first.status_code,502);self.assertEqual(second.content,b'ok');self.assertEqual(called,['p','d']);self.assertEqual(len(app.state.placement.leases),0)
    async def test_interrupted_stream_no_retry_and_release(self):
        calls=[]
        class Broken(Stream):
            async def __aiter__(self):
                yield b'data: partial\n\n'
                raise httpx.ReadError('fixture interrupted')
        stream=Broken([])
        async def upstream(request):calls.append(request);return httpx.Response(200,stream=stream)
        app=create_app([{'id':'P','url':'http://p'}],httpx.MockTransport(upstream));messages=[{'type':'http.request','body':b'{}','more_body':False}];sent=[]
        async def receive():
            if messages:return messages.pop(0)
            await asyncio.sleep(60)
        async def send(message):sent.append(message)
        scope={'type':'http','asgi':{'version':'3.0','spec_version':'2.4'},'http_version':'1.1','method':'POST','scheme':'http','path':'/v1/completions','raw_path':b'/v1/completions','query_string':b'','headers':[],'client':('127.0.0.1',1),'server':('127.0.0.1',2)}
        async with app.router.lifespan_context(app):
            with self.assertRaises(httpx.ReadError):await app(scope,receive,send)
            if app.state.cleanup_tasks:await asyncio.gather(*app.state.cleanup_tasks)
            self.assertEqual(len(calls),1);self.assertTrue(stream.closed);self.assertEqual(len(app.state.placement.leases),0)
            self.assertEqual(sent[0]['status'],200);self.assertEqual(sent[1]['body'],b'data: partial\n\n')
if __name__=='__main__':unittest.main()
