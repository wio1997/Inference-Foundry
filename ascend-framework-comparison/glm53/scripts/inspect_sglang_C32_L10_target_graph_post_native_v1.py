"""Read-only post-formal native capture; retain raw failure/nulls and original D metrics."""
from pathlib import Path
import hashlib,importlib.util
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=ROOT/'scripts/glm53_sglang_C32_target_graph_native_capture_v2.py';assert hashlib.sha256(p.read_bytes()).hexdigest()=='d8ce3ece25497949f45a5033f73a8dedd103c7e2141c64c1f3cc40cb3eb60971'
 spec=importlib.util.spec_from_file_location('capture',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 m.execute('config/C32-target-graph-runtime-v3/formal-C32-target-graph-v1.json','69d59821b32dfb03ac21bd4380bb0ebf7e753c43c04f4ab8b5af98ab7df4bfe1','20261018T-L10-sglang-target-graph-post-native-v1',mode='formal')
if __name__=='__main__':main()
