"""Read-only capture after real C32 short and two80K semantic cases; no new requests."""
from pathlib import Path
import hashlib,importlib.util,json
ROOT=Path(__file__).resolve().parents[1]
def main():
 short=json.loads((ROOT/'results/sglang-C32-target-graph-functional-v5.json').read_text());assert short['pass'] and short['successful_backend_inferences']==64
 long=json.loads((ROOT/'runs/20261018T-L10-long-completed-artifact-recovery-v1/receipt.json').read_text());assert long['rc']==0 and long['collection_only_recovered'] and long['original_result_verified'] and long['new_model_requests']==0 and long['original_client_rc']==0
 p=ROOT/'scripts/glm53_sglang_C32_target_graph_native_capture_v2.py';assert hashlib.sha256(p.read_bytes()).hexdigest()=='d8ce3ece25497949f45a5033f73a8dedd103c7e2141c64c1f3cc40cb3eb60971'
 spec=importlib.util.spec_from_file_location('capture',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 m.execute('config/C32-target-graph-runtime-v3/native-correctness-settings-v1.json','4b299f96610a80da55b8a2739c93faa9832b6a2aba12cf7919910782d09c91d7','20261018T-L10-sglang-target-graph-native-qualification-v1',mode='baseline')
if __name__=='__main__':main()
