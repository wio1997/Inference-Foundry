import asyncio,json,sys,tempfile,time,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'runtime'))
import httpx
from loadgen import execute
class Stream(httpx.AsyncByteStream):
    def __init__(self,delay=0):self.delay=delay
    async def __aiter__(self):
        await asyncio.sleep(self.delay)
        yield b'data: {"choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":null}]}\n\n'
        yield b'data: {"choices":[{"index":0,"delta":{},"finish_reason":"length"}],"usage":{"prompt_tokens":2,"completion_tokens":3,"total_tokens":5}}\n\n'
        yield b'data: [DONE]\n\n'
class LoadContracts(unittest.IsolatedAsyncioTestCase):
    async def test_open_arrivals_independent_of_previous_completion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);body=root/'body.json';body.write_text(json.dumps({'stream':True,'seed':7,'max_tokens':3,'tools':[]}));times=[]
            async def upstream(request):times.append(time.monotonic());self.assertEqual(json.loads(await request.aread())['seed'],7);return httpx.Response(200,stream=Stream(.1))
            plan={'endpoint':'http://fixture/v1/chat/completions','requests':[{'id':'a','arrival_s':0,'body_path':str(body),'expected':{'output_tokens':3}},{'id':'b','arrival_s':.01,'body_path':str(body),'expected':{'output_tokens':3}}]}
            result=await execute(plan,root,httpx.MockTransport(upstream));self.assertTrue(result['valid']);self.assertLess(times[1]-times[0],.08);self.assertEqual(result['effective_output_tokens'],6)
    async def test_truncated_stream_not_effective_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);body=root/'body.json';body.write_text('{"stream":true}')
            class Truncated(httpx.AsyncByteStream):
                async def __aiter__(self):yield b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'
            async def upstream(request):return httpx.Response(200,stream=Truncated())
            result=await execute({'endpoint':'http://fixture/v1/chat/completions','requests':[{'id':'a','arrival_s':0,'body_path':str(body)}]},root,httpx.MockTransport(upstream));self.assertFalse(result['valid']);self.assertEqual(result['effective_output_tokens'],0)
    async def test_negative_contract_excluded_from_inference_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);body=root/'body.json';body.write_text('{"stream":true}')
            async def upstream(request):return httpx.Response(400,json={'error':'fixture'})
            result=await execute({'endpoint':'http://fixture/v1/chat/completions','requests':[{'id':'a','arrival_s':0,'body_path':str(body),'expected':{'http_status':400}}]},root,httpx.MockTransport(upstream));self.assertTrue(result['valid']);self.assertEqual(result['successful_inference_requests'],0);self.assertEqual(result['expected_rejections'],1)
    async def test_completion_stream_first_text_is_ttft(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);body=root/'body.json';body.write_text('{"stream":true}')
            class Completion(httpx.AsyncByteStream):
                async def __aiter__(self):
                    yield b'data: {"choices":[{"index":0,"text":"hello","finish_reason":null}]}\n\n'
                    yield b'data: {"choices":[{"index":0,"text":"","finish_reason":"length"}],"usage":{"completion_tokens":2}}\n\n'
                    yield b'data: [DONE]\n\n'
            async def upstream(request):return httpx.Response(200,stream=Completion())
            result=await execute({'endpoint':'http://fixture/v1/completions','requests':[{'id':'a','arrival_s':0,'body_path':str(body)}]},root,httpx.MockTransport(upstream));self.assertTrue(result['valid']);self.assertIsNotNone(result['requests'][0]['ttft_s'])
    async def test_native_reasoning_is_first_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);body=root/'body.json';body.write_text('{"stream":true}')
            class Reasoning(httpx.AsyncByteStream):
                async def __aiter__(self):
                    yield b'data: {"choices":[{"index":0,"delta":{"reasoning":"thinking"}}]}\n\n'
                    yield b'data: {"choices":[{"index":0,"delta":{},"finish_reason":"length"}],"usage":{"completion_tokens":2}}\n\n'
                    yield b'data: [DONE]\n\n'
            async def upstream(request):return httpx.Response(200,stream=Reasoning())
            result=await execute({'endpoint':'http://fixture/v1/chat/completions','requests':[{'id':'a','arrival_s':0,'body_path':str(body)}]},root,httpx.MockTransport(upstream));self.assertTrue(result['valid']);self.assertIsNotNone(result['requests'][0]['ttft_s'])
if __name__=='__main__':unittest.main()
