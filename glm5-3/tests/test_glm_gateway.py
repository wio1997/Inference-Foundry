import asyncio,json,sys,tempfile,types,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'runtime'))
from glm_gateway import StageTraceMiddleware,TraceSink,instrument_stock,trace_context

class GatewayContracts(unittest.IsolatedAsyncioTestCase):
    async def compare_messages(self,enabled,code,chunks):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'events.jsonl';sink=TraceSink(str(path) if enabled else None)
            received={'type':'http.request','body':b'{"messages":[]}', 'more_body':False};forwarded=[]
            messages=[{'type':'http.response.start','status':code,'headers':[(b'content-type',b'text/event-stream')]},*[{'type':'http.response.body','body':chunk,'more_body':i<len(chunks)-1} for i,chunk in enumerate(chunks)]]
            async def receive():return received
            async def send(message):forwarded.append(message)
            async def native(scope,r,s):
                self.assertIs(await r(),received)
                for message in messages:await s(message)
            await StageTraceMiddleware(native,sink)({'type':'http','path':'/v1/chat/completions','method':'POST'},receive,send)
            self.assertEqual(forwarded,messages)
            for original,actual in zip(messages,forwarded):self.assertIs(original,actual)
            self.assertIsNone(trace_context.get())
            if enabled:
                events=[json.loads(l) for l in path.read_text().splitlines()];self.assertEqual(events[0]['event'],'request_received');self.assertEqual(events[-1]['event'],'request_finished');self.assertEqual(len({e['trace_id'] for e in events}),1)
                self.assertNotIn('messages',path.read_text());__import__('os').close(sink.fd)
    async def test_sse_bytes_and_status_unchanged(self):
        await self.compare_messages(True,200,[b'data: {"choices":',b'[]}\n\ndata: [DONE]\n\n'])
    async def test_upstream_error_unchanged(self):
        await self.compare_messages(True,400,[b'{"error":"maximum context length"}'])
    async def test_disabled_path_unchanged(self):
        await self.compare_messages(False,200,[b'{}'])
    async def test_disconnect_and_cancellation_propagate(self):
        with tempfile.TemporaryDirectory() as tmp:
            sink=TraceSink(str(Path(tmp)/'trace'));message={'type':'http.disconnect'}
            async def receive():return message
            async def send(_message):raise AssertionError('should not send')
            async def native(_scope,r,_s):
                self.assertIs(await r(),message);raise asyncio.CancelledError()
            with self.assertRaises(asyncio.CancelledError):
                await StageTraceMiddleware(native,sink)({'type':'http'},receive,send)
            events=[json.loads(l) for l in (Path(tmp)/'trace').read_text().splitlines()]
            self.assertIn('client_disconnect',[e['event'] for e in events]);self.assertEqual(events[-1]['event'],'request_finished');self.assertIsNone(trace_context.get());__import__('os').close(sink.fd)
    async def test_native_stream_retry_payload_and_close(self):
        with tempfile.TemporaryDirectory() as tmp:
            sink=TraceSink(str(Path(tmp)/'trace'));closed=[];args_seen=[]
            async def stream(*args,**kwargs):
                args_seen.append((args,kwargs))
                try:
                    yield b'one';yield b'two'
                finally:closed.append(True)
            async def native(*args,**kwargs):return types.SimpleNamespace(request_id='backend',prefiller_key='P',decoder_key='D',prefiller_cached_tokens=32)
            stock=types.SimpleNamespace(assign_instances=native,send_request_to_service=native,_finish_instance=native,stream_service_response=stream)
            instrument_stock(stock,sink);token=trace_context.set({'trace_id':'test','trace_write_failed':False})
            try:
                gen=stock.stream_service_response('client','api',{'max_tokens':2},request_id='request',max_retries=3)
                self.assertEqual(await gen.__anext__(),b'one');await gen.aclose()
                self.assertEqual(closed,[True]);self.assertEqual(args_seen[0][1],{'request_id':'request','max_retries':3});self.assertEqual(args_seen[0][0][2],{'max_tokens':2})
            finally:trace_context.reset(token);__import__('os').close(sink.fd)
if __name__=='__main__':unittest.main()
