import asyncio,json,sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"runtime"))
import httpx
from sse_observer import NativeSSEObserver
from replica_gateway import create_app
from loadgen import execute

ERROR=b'data: {"error": {"message": "EngineCore encountered an issue. See stack trace (above) for the root cause.", "type": "InternalServerError", "param": null, "code": 500}}\n\n'
class Stream(httpx.AsyncByteStream):
    def __init__(self,chunks):self.chunks=chunks;self.closed=False
    async def __aiter__(self):
        for chunk in self.chunks:yield chunk
    async def aclose(self):self.closed=True

class ObserverContracts(unittest.TestCase):
    def test_real_error_at_every_chunk_boundary(self):
        for cut in range(len(ERROR)+1):
            observer=NativeSSEObserver();observer.feed(ERROR[:cut]);observer.feed(ERROR[cut:]);self.assertTrue(observer.failed);self.assertEqual(observer.errors,1)
    def test_multiline_crlf_and_content_not_error(self):
        observer=NativeSSEObserver()
        for byte in b'data: {"error":\r\ndata: {"code": 500}}\r\n\r\n':observer.feed(bytes([byte]))
        self.assertTrue(observer.failed)
        other=NativeSSEObserver();other.feed(b'data: {"choices":[{"delta":{"content":"error"}}]}\n\ndata: {"error":{"code":400}}\n\n')
        self.assertFalse(other.failed)
    def test_oversized_frame_bounded_and_next_frame_recovers(self):
        observer=NativeSSEObserver(256);observer.feed(b'data: '+b'x'*10000)
        self.assertLessEqual(len(observer.line),256);observer.feed(b'\n\n'+ERROR)
        self.assertTrue(observer.failed);self.assertEqual(observer.errors,1)

class InBandContracts(unittest.IsolatedAsyncioTestCase):
    async def test_native_error_unchanged_and_fault_isolated_without_retry(self):
        calls=[];first=Stream([ERROR[:19],ERROR[19:]+b'data: [DONE]\n\n'])
        async def upstream(request):
            calls.append(request.url.host)
            return httpx.Response(200,stream=first if request.url.host=="p" else Stream([b'ok']),headers={"content-type":"text/event-stream"})
        app=create_app([{"id":"P","url":"http://p"},{"id":"D","url":"http://d"}],httpx.MockTransport(upstream))
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url="http://gateway") as client:
                response=await client.post("/v1/chat/completions",content=b'{"stream":true}')
                state=await app.state.placement.snapshot()
                self.assertTrue(next(r for r in state if r["id"]=="P")["temporarily_unhealthy"])
                second=await client.post("/v1/chat/completions",content=b'{"stream":true}')
            self.assertEqual(response.status_code,200);self.assertEqual(response.content,ERROR+b'data: [DONE]\n\n')
            self.assertEqual(second.content,b'ok');self.assertEqual(calls,["p","d"]);self.assertTrue(first.closed);self.assertEqual(len(app.state.placement.leases),0)
    async def test_native_error_never_credits_usage_even_if_stream_finishes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);body=root/"body.json";body.write_text('{"stream":true}')
            trailing=b'data: {"choices":[{"index":0,"finish_reason":"length"}],"usage":{"completion_tokens":3}}\n\ndata: [DONE]\n\n'
            async def upstream(request):return httpx.Response(200,stream=Stream([ERROR,trailing]))
            result=await execute({"endpoint":"http://native","requests":[{"id":"a","arrival_s":0,"body_path":str(body)}]},root,httpx.MockTransport(upstream))
            self.assertFalse(result["valid"]);self.assertEqual(result["effective_output_tokens"],0)
            row=result["requests"][0];self.assertEqual(row["outcome"],"native_error");self.assertEqual(row["error"]["code"],500);self.assertTrue(row["done"])
if __name__=="__main__":unittest.main()
