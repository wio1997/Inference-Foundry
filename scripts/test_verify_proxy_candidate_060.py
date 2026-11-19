#!/usr/bin/env python3
"""Synthetic AST guards; no historical proxy or HTTP payloads."""
import unittest
import verify_proxy_candidate_060 as s


SOURCE='''
def build_prefill_request(req_data):
    payload = req_data.copy()
    payload["stream"] = False
    payload["max_tokens"] = 1
    payload["min_tokens"] = 1
    return payload
async def assign_instances():
    response = await send_request_to_service()
    decoder = await runtime.schedule("pick_decoder")
    return response, decoder
async def handle_completions_impl():
    assigned = await assign_instances()
    stream = stream_service_response()
    chunk = await stream.__anext__()
    return assigned, chunk
'''


class Tests(unittest.TestCase):
 def test_expected(self):self.assertTrue(s.facts(SOURCE)['decode_stream_open_precedes_first_chunk_await_in_source'])
 def test_wrong_order(self):
  changed=SOURCE.replace('    response = await send_request_to_service()\n    decoder = await runtime.schedule("pick_decoder")','    decoder = await runtime.schedule("pick_decoder")\n    response = await send_request_to_service()')
  with self.assertRaises(ValueError):s.facts(changed)
 def test_wrong_payload(self):
  with self.assertRaises(ValueError):s.facts(SOURCE.replace('payload["max_tokens"] = 1','payload["max_tokens"] = 2'))
 def test_nested_sha(self):self.assertEqual(s.count_value({'a':[{'b':'same'},'same']},'same'),2)


if __name__=='__main__':unittest.main(verbosity=2)
