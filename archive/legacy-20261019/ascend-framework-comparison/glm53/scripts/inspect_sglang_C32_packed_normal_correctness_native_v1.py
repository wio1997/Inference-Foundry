"""Read-only capture after real C32 short and two80K semantic cases; no new requests."""
from pathlib import Path
import hashlib,importlib.util,json
ROOT=Path(__file__).resolve().parents[1]
def main():
 short=json.loads((ROOT/'results/sglang-C32-packed-normal-functional-v1.json').read_text());assert short['pass'] and short['successful_backend_inferences']==64
 long=json.loads((ROOT/'runs/20261018T-L09-sglang-packed-long-v1/receipt.json').read_text());assert long['rc']==0
 p=ROOT/'scripts/glm53_sglang_C32_packed_normal_native_capture_v1.py';assert hashlib.sha256(p.read_bytes()).hexdigest()=='89b278ede3f7774460ea26bba8de8f711abc38b09fc0279d0776fd3d18fb144d'
 spec=importlib.util.spec_from_file_location('capture',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 m.execute('config/C32-packed-normal-runtime-v1/native-correctness-settings-v1.json','ded73807d8565eb9b63efdd2979f000d2e8e28f2f1ef1b5b96afca222437ed22','20261018T-L09-sglang-packed-native-qualification-v1',mode='baseline')
if __name__=='__main__':main()
