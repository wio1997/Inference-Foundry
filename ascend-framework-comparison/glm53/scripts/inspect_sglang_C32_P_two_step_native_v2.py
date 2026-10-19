"""Frozen C32 diagnostic; exact once through ZCode only."""
import hashlib,importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=ROOT/'scripts/glm53_C32_P_native_profile_capture_v2.py';assert hashlib.sha256(p.read_bytes()).hexdigest()=='22a9868de8d52da60f53dd573b232a113b71198aff4e7a96a6295a9e24d7ae6c'
 s=importlib.util.spec_from_file_location('controller',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
 m.execute('config/C32-target-graph-runtime-v3/P-two-step-diagnostic-v2.json','3726d079a2a81898ac809185809cb581c263fa5788b46e10d8022c2371f555dc','20261018T-L11-sglang-C32-P-two-step-native-v1')
if __name__=='__main__':main()
