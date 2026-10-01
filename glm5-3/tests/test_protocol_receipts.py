import hashlib,json,sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"runtime"))
from protocol_receipts import verify_protocol_receipts
from sse_observer import NativeSSEObserver
RAW=b'data: {"choices":[{"index":0,"delta":{"reasoning":"x"},"finish_reason":"length"}],"usage":{"prompt_tokens":5,"completion_tokens":3,"total_tokens":8}}\n\ndata: [DONE]\n\n'
class Receipts(unittest.TestCase):
 def fixture(self,root,nodes=("P166",)):
  events=[]
  for i,node in enumerate(nodes):
   p=root/(str(i)+".sse");p.write_bytes(RAW);o=NativeSSEObserver(collect_contract=True);o.feed(RAW)
   common={"lease_id":str(i),"replica":node}
   events.extend([{**common,"event":"lease_acquired","method":"POST","path":"/v1/chat/completions","output_budget":3,"body_sha256":"body"+str(i)},{**common,"event":"upstream_headers","status":200},{**common,"event":"upstream_stream_contract","contract":o.contract(),"audit_error":None,"wire_path":str(p),"wire_bytes":len(RAW),"wire_sha256":hashlib.sha256(RAW).hexdigest()},{**common,"event":"lease_released","released":True,"backend_failure":False}])
  trace=root/"trace.jsonl";trace.write_text("".join(json.dumps(x)+"\n" for x in events));return trace,events
 def save(self,p,rows):p.write_text("".join(json.dumps(x)+"\n" for x in rows))
 def test_complete_native_wires_account_exactly_once(self):
  with tempfile.TemporaryDirectory() as t:
   trace,_=self.fixture(Path(t),("P166","D167"));r=verify_protocol_receipts(trace,2,5,3,["P166","D167"]);self.assertEqual(r["effective_output_tokens"],6);self.assertEqual(len(r["requests"]),2)
 def test_success_metadata_cannot_certify_missing_done(self):
  with tempfile.TemporaryDirectory() as t:
   trace,rows=self.fixture(Path(t));p=Path(rows[2]["wire_path"]);data=RAW.split(b'data: [DONE]')[0];p.write_bytes(data);rows[2]["wire_bytes"]=len(data);rows[2]["wire_sha256"]=hashlib.sha256(data).hexdigest();self.save(trace,rows)
   with self.assertRaises(ValueError):verify_protocol_receipts(trace,1,5,3)
 def test_duplicate_or_unreleased_attempts_never_hide(self):
  for mode in ["duplicate","unreleased"]:
   with tempfile.TemporaryDirectory() as t:
    trace,rows=self.fixture(Path(t))
    if mode=="duplicate":rows.append(dict(rows[0]))
    else:rows[-1]["released"]=False
    self.save(trace,rows)
    with self.assertRaises(ValueError):verify_protocol_receipts(trace,1,5,3)
 def test_both_warmup_attempts_on_one_node_are_rejected(self):
  with tempfile.TemporaryDirectory() as t:
   trace,_=self.fixture(Path(t),("P166","P166"))
   with self.assertRaises(ValueError):verify_protocol_receipts(trace,2,5,3,["P166","D167"])
 def test_corrupted_wire_or_forged_usage_rejected(self):
  for mode in ["bytes","usage"]:
   with tempfile.TemporaryDirectory() as t:
    trace,rows=self.fixture(Path(t))
    if mode=="bytes":Path(rows[2]["wire_path"]).write_bytes(RAW+b"x")
    else:rows[2]["contract"]["usage"]["total_tokens"]=9;self.save(trace,rows)
    with self.assertRaises(ValueError):verify_protocol_receipts(trace,1,5,3)
if __name__=="__main__":unittest.main()
