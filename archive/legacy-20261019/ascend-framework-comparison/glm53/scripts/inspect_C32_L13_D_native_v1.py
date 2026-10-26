"""One-shot read-only L13 D two-step native capture; fixed reviewed entry only."""
import hashlib,importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
CAPTURE_ID='20261019T-L13-sglang-C32-D-two-step-native-v1'
DIAG_RID='20261019T-L13-sglang-C32-D-native-decode-two-step-profile-v1'
REVIEW='results/C32-L13-D-native-actual-review-v1.json'
CAPTURE_SHA='0ab06cc7f3e15e69f51be364d0aace56be1a7f126e740baf2dc0be70f6f07f81'

def main():
 p=ROOT/'scripts/glm53_C32_L13_D_native_capture_v1.py';assert hashlib.sha256(p.read_bytes()).hexdigest()==CAPTURE_SHA
 spec=importlib.util.spec_from_file_location('l13_capture',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 assert m.CAPTURE_ID==CAPTURE_ID and m.DIAG_RID==DIAG_RID
 m.execute(CAPTURE_ID,REVIEW)
if __name__=='__main__':main()
