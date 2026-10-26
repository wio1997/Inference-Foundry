"""Exact once C32 L10 target full Graph candidate measurement, real semantics/native gates required."""
import hashlib,importlib.util,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=ROOT/'results/sglang-C32-target-graph-native-readiness-v1.json';assert hashlib.sha256(p.read_bytes()).hexdigest()=='bbb3eb81390e4aeca548fe6c055379c256a6122c3b73f2adcf4e8be5b1e3d52f'
 native=json.loads(p.read_text());assert native['pass'] and native['successful_functional_backend_inferences']==76 and native['all_DP_short_and_80K_semantic_verified'] and native['native_cache_and_SDMA_verified']
 p=ROOT/'scripts/glm53_sglang_metadata_launch_v1.py';assert hashlib.sha256(p.read_bytes()).hexdigest()=='c57a105a1803ca1d28bcd098f6ea722c77883757ecbd3180c00ae5798b8a4b2d'
 spec=importlib.util.spec_from_file_location('operation',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 m.execute('config/C32-target-graph-runtime-v3/formal-C32-target-graph-v1.json','69d59821b32dfb03ac21bd4380bb0ebf7e753c43c04f4ab8b5af98ab7df4bfe1')
if __name__=='__main__':main()
