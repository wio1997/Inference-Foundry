"""Read-only native and completeD16metrics capture after fresh L04 C32 control."""
from pathlib import Path
import hashlib,importlib.util
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=ROOT/'scripts/glm53_sglang_C32_L09_L04_after_control_native_capture_v1.py';assert hashlib.sha256(p.read_bytes()).hexdigest()=='e1f27b40dd07109c500b346c7dd2c8388643fdfbe9e95cc4d2d36141a93139b9'
 spec=importlib.util.spec_from_file_location('capture',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 m.execute('config/C32-packed-normal-runtime-v1/formal-C32-L04-after-control-v1.json','d6a9bbba88a63258e5d4c97f3afa28756251c07bac48a714364f283ed2bd1407','20261018T-L09-sglang-L04-after-control-post-native-v1',mode='formal')
if __name__=='__main__':main()
