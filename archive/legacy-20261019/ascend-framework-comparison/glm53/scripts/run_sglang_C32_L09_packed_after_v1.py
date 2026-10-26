"""Exact once C32 L09 candidate measurement, real semantics/native gates required."""
import hashlib,importlib.util,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=ROOT/'results/sglang-C32-packed-normal-native-readiness-v1.json';assert hashlib.sha256(p.read_bytes()).hexdigest()=='5a74c4283fc454d9572e74eaf36497286580dc50ad8f0f96b15cd0b138f207c0'
 native=json.loads(p.read_text());assert native['pass'] and native['successful_functional_backend_inferences']==76 and native['all_DP_short_and_80K_semantic_verified'] and native['native_cache_and_SDMA_verified']
 p=ROOT/'scripts/glm53_sglang_metadata_launch_v1.py';assert hashlib.sha256(p.read_bytes()).hexdigest()=='c57a105a1803ca1d28bcd098f6ea722c77883757ecbd3180c00ae5798b8a4b2d'
 spec=importlib.util.spec_from_file_location('operation',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 m.execute('config/C32-packed-normal-runtime-v1/formal-C32-packed-after-v1.json','522402248e183a0e46a08346f67c451bb1bcde9e3f9fb22aad9db809e00d1ca5')
if __name__=='__main__':main()
