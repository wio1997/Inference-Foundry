"""Exact-once C32 native HiCache measurement after complete native qualification."""
import hashlib,importlib.util,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=ROOT/'results/sglang-C32-L12C-native-readiness-v1.json';assert hashlib.sha256(p.read_bytes()).hexdigest()=='8b8314fc99dcbb0232c2c86c5ff1b6830b7ae623f7d045154b9f11e93fb14f93'
 q=json.loads(p.read_text());assert q['pass'] and q['successful_functional_backend_inferences']==76 and q['all32_P_actual_native_packed_MTP_firstuse_verified'] and q['native_cache_and_SDMA_verified']
 p=ROOT/'scripts/glm53_C32_L12C_metadata_launch_v1.py';assert hashlib.sha256(p.read_bytes()).hexdigest()=='b68a1ac2ed6e13790ce48a421facdbb4f196a7baaccf8d2394f8b7685d402802'
 spec=importlib.util.spec_from_file_location('frozen_operation',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.execute('config/C32-L12C-native-runtime-v1/formal-C32-native-v1.json','356383a1f4a258352a762bc51e9ac875b2230e861026e0c47e53ea83357e5215')
if __name__=='__main__':main()
