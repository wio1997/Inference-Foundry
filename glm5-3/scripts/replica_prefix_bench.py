"""AISBench entry with native wire receipts checked after each actual phase."""
import json,os,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"runtime"))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"adapters/aisbench"))
import prefix_bench
from protocol_receipts import verify_protocol_receipts
original=prefix_bench.validate_phase_details
def validate(log_dir,phase,expected_count,output_len):
 acceptance=original(log_dir,phase,expected_count,output_len)
 receipt=verify_protocol_receipts(os.environ["GLM_FORMAL_ROUTER_TRACE_PATH"],expected_count,73740 if phase=="warmup" else 81932,output_len,["P166","D167"] if phase=="warmup" else None)
 p=Path(os.environ["GLM_FORMAL_RECEIPTS_PATH"])/(phase+"_wire_acceptance.json")
 p.write_text(json.dumps(receipt,indent=2)+"\n")
 return acceptance
prefix_bench.validate_phase_details=validate
if __name__=="__main__":sys.exit(prefix_bench.main())
