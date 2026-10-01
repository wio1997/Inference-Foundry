import hashlib,json,os,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"runtime"))
import httpx
from sse_observer import NativeSSEObserver
from replica_gateway import create_app
RAW=(b'data: {"choices":[{"index":0,"delta":{"reasoning":"x"},"finish_reason":null}]}\n\n'
 b'data: {"choices":[{"index":0,"delta":{},"finish_reason":"length"}]}\n\n'
 b'data: {"choices":[],"usage":{"prompt_tokens":5,"completion_tokens":3,"total_tokens":8}}\n\n'
 b'data: [DONE]\n\n')
class Stream(httpx.AsyncByteStream):
 async def __aiter__(self):
  for i in range(0,len(RAW),7):yield RAW[i:i+7]
class AuditContracts(unittest.TestCase):
 def test_all_byte_boundaries_retain_complete_contract(self):
  for cut in range(len(RAW)+1):
   o=NativeSSEObserver(collect_contract=True);o.feed(RAW[:cut]);o.feed(RAW[cut:]);c=o.contract()
   self.assertTrue(c["done"]);self.assertEqual(c["usage"]["completion_tokens"],3);self.assertEqual(c["finish_reasons"],{"0":"length"});self.assertFalse(c["unknown"]);self.assertFalse(c["native_error"])
 def test_oversized_frame_cannot_be_certified_by_later_usage_done(self):
  o=NativeSSEObserver(256,collect_contract=True);o.feed(b'data: '+b'x'*10000+b'\n\n'+RAW)
  self.assertTrue(o.contract()["unknown"]);self.assertTrue(o.done)
 def test_error_or_post_done_data_remains_uncertifiable(self):
  o=NativeSSEObserver(collect_contract=True);o.feed(b'data: {"error":{"code":400}}\n\n'+RAW);self.assertTrue(o.contract()["native_error"])
  other=NativeSSEObserver(collect_contract=True);other.feed(RAW+b'data: {"choices":[]}\n\n');self.assertTrue(other.contract()["unknown"])
class GatewayAudit(unittest.IsolatedAsyncioTestCase):
 async def exercise(self,fail_write):
  with tempfile.TemporaryDirectory() as t:
   r=Path(t);trace=r/"trace.jsonl";calls=[]
   async def upstream(req):calls.append(req);return httpx.Response(200,stream=Stream(),headers={"content-type":"text/event-stream"})
   with patch.dict(os.environ,{"GLM_ROUTER_AUDIT_DIR":str(r/"raw"),"GLM_ROUTER_TRACE_PATH":str(trace)}):
    app=create_app([{"id":"P","url":"http://p"}],httpx.MockTransport(upstream))
    async with app.router.lifespan_context(app):
     async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url="http://g") as client:
      if fail_write:
       with patch("builtins.open",side_effect=OSError("audit unavailable")):response=await client.post("/v1/chat/completions",content=b'{"max_tokens":3}')
      else:response=await client.post("/v1/chat/completions",content=b'{"max_tokens":3}')
     self.assertFalse(app.state.placement.leases)
   self.assertEqual(response.content,RAW);self.assertEqual(len(calls),1)
   e=[json.loads(x) for x in trace.read_text().splitlines() if json.loads(x)["event"]=="upstream_stream_contract"];self.assertEqual(len(e),1)
   e=e[0];self.assertEqual(e["wire_sha256"],hashlib.sha256(RAW).hexdigest());self.assertEqual(e["wire_bytes"],len(RAW));self.assertTrue(e["contract"]["done"]);self.assertEqual(e["contract"]["finish_reasons"],{"0":"length"})
   if fail_write:self.assertTrue(e["audit_error"])
   else:self.assertIsNone(e["audit_error"]);self.assertEqual(Path(e["wire_path"]).read_bytes(),RAW)
 async def test_raw_audit_and_forwarding_match(self):await self.exercise(False)
 async def test_audit_io_failure_does_not_change_native_bytes(self):await self.exercise(True)
if __name__=="__main__":unittest.main()
