import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"runtime"))
from placement import Placement,Lease
from sse_observer import NativeSSEObserver
class PhaseContracts(unittest.IsolatedAsyncioTestCase):
    async def test_prefill_aware_avoids_busy_prefill_despite_fewer_leases(self):
        p=Placement([{"id":"P","url":"http://p"},{"id":"D","url":"http://d"}],"prefill_aware")
        first=await p.acquire(2048,100);second=await p.acquire(2048,100)
        await p.output_started(first)
        third=await p.acquire(4096,100);self.assertEqual(third.replica.key,"P");await p.output_started(third)
        short=await p.acquire(128,20);self.assertEqual(short.replica.key,"P")
        for x in [first,second,third,short]:await p.release(x)
        self.assertFalse(await p.output_started(first));self.assertTrue(all(r["prefilling_requests"]==0 for r in await p.snapshot()))
    async def test_default_active_count_preserved_and_progress_identity(self):
        p=Placement([{"id":"P","url":"http://p"},{"id":"D","url":"http://d"}]);first=await p.acquire(1,1);second=await p.acquire(1,1)
        await p.output_started(first)
        third=await p.acquire(1,1);self.assertEqual(third.replica.key,"P")
        fourth=await p.acquire(1,1);self.assertEqual(fourth.replica.key,"D")
        fake=Lease(second.lease_id,second.replica,1,1)
        with self.assertRaises(ValueError):await p.output_started(fake)
        self.assertIn(second.lease_id,second.replica.prefilling)
        for x in [first,second,third,fourth]:await p.release(x)
    async def test_late_progress_cannot_change_new_generation(self):
        p=Placement([{"id":"P","url":"http://p"}],"prefill_aware");old=await p.acquire(1,1);await p.remove("P");await p.release(old);await p.add({"id":"P","url":"http://p"})
        new=await p.acquire(1,1);self.assertFalse(await p.output_started(old));self.assertIn(new.lease_id,new.replica.prefilling)
        self.assertTrue(await p.output_started(new));self.assertFalse(await p.output_started(new));await p.release(new)
class FirstOutputContracts(unittest.TestCase):
    def test_role_empty_is_not_output_then_reasoning_is(self):
        p=NativeSSEObserver();p.feed(b'data: {"choices":[{"delta":{"role":"assistant","content":""}}]}\n\n');self.assertFalse(p.output_started)
        block=b'data: {"choices":[{"delta":{"reasoning":"thinking"}}]}\n\n'
        for x in block:p.feed(bytes([x]))
        self.assertTrue(p.output_started)
    def test_malformed_native_shapes_remain_unknown(self):
        for block in [b'data: {"choices":null}\n\n',b'data: {"choices":[null,"text",{"delta":"invalid"}]}\n\n']:
            p=NativeSSEObserver();p.feed(block);self.assertFalse(p.output_started);self.assertFalse(p.failed)
    def test_tool_text_and_error(self):
        for block in [b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"name":"f"}}]}}]}\n\n',b'data: {"choices":[{"text":"hello"}]}\n\n']:
            p=NativeSSEObserver();p.feed(block);self.assertTrue(p.output_started)
        p=NativeSSEObserver();p.feed(b'data: {"error":{"code":500}}\n\n');self.assertFalse(p.output_started);self.assertTrue(p.failed)
if __name__=="__main__":unittest.main()
