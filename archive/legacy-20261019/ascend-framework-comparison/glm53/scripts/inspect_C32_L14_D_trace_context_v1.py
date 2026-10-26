"""One-shot read-only L14 D existing-trace context collector; fixed reviewed entry only."""
import hashlib,importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
RUN_ID='20261019T-L14-sglang-existing-D-trace-context-v1'
REVIEW='results/C32-L14-D-trace-context-actual-review-v1.json'
COLLECTOR_SHA='440e5755323a45968aacf460735a1e6a2f718ecf717d58ea389e35c3b838855a'

def main():
 p=ROOT/'scripts/glm53_C32_L14_D_trace_context_v1.py';assert hashlib.sha256(p.read_bytes()).hexdigest()==COLLECTOR_SHA
 spec=importlib.util.spec_from_file_location('l14_trace_context',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 assert m.RUN_ID==RUN_ID and m.REVIEW==REVIEW
 m.execute(RUN_ID,REVIEW)
if __name__=='__main__':main()
