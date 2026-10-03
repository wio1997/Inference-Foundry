"""Actual AISBench adapter with phase-scoped raw receipts on retained public service."""
import json,os,sys,hashlib
from pathlib import Path
g=Path("/data/tiankuan/wio/Inference-Foundry/glm5-3");sys.path.insert(0,str(g/"runtime"));sys.path.insert(0,str(g/"adapters/aisbench"))
import prefix_bench
from protocol_receipts import verify_protocol_receipts
original=prefix_bench.validate_phase_details
def validate(log_dir,phase,expected_count,output_len):
 acceptance=original(log_dir,phase,expected_count,output_len)
 cursor=json.loads(Path(os.environ["GLM_FORMAL_TRACE_CURSOR"]).read_text());source=Path(cursor["path"]);raw=source.read_bytes()
 assert hashlib.sha256(raw[:cursor["bytes"]]).hexdigest()==cursor["sha256"]and raw[cursor["bytes"]-1:cursor["bytes"]]==b"\n"
 scoped=Path(os.environ["GLM_FORMAL_ROUTER_TRACE_PATH"]);scoped.write_bytes(raw[cursor["bytes"]:])
 receipt=verify_protocol_receipts(scoped,expected_count,73740 if phase=="warmup"else 81932,output_len,["D0","D1"])
 counts={key:sum(x["replica"]==key for x in receipt["requests"])for key in ["D0","D1"]}
 assert counts=={"D0":1,"D1":1}if phase=="warmup"else counts=={"D0":2,"D1":2}
 (Path(os.environ["GLM_FORMAL_RECEIPTS_PATH"])/(phase+"_wire_acceptance.json")).write_text(json.dumps(receipt,indent=2)+"\n")
 return acceptance
prefix_bench.validate_phase_details=validate
if __name__=="__main__":sys.exit(prefix_bench.main())
