"""Read-only capture after actual restored-runtime short/two80K guards."""
import hashlib,importlib.util,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 short=json.loads((ROOT/'results/sglang-C32-L12B-restored-L10-functional-v3.json').read_text());assert short['pass'] and short['successful_backend_inferences']==64
 long=json.loads((ROOT/'runs/20261019T-L12B-sglang-restored-L10-long-v1/receipt.json').read_text());assert long['rc']==0
 p=ROOT/'scripts/glm53_C32_L12B_restored_native_capture_v2.py';assert hashlib.sha256(p.read_bytes()).hexdigest()=='73cef8615a534fcb744dfd5ecd6a9ff771f2c15612dec0e4fa78f7003bb2ce73';s=importlib.util.spec_from_file_location('cap',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);m.execute('20261019T-L12B-sglang-restored-L10-functional-native-v1','results/C32-L12B-restored-post-functional-native-review-v1.json')
if __name__=='__main__':main()
