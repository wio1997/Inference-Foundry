"""C32-only actual native SDK/SDMA counter baseline; zero model/SDK calls."""
import hashlib,importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=ROOT/'scripts/glm53_sglang_C32_L04_native_capture_v1.py';assert hashlib.sha256(p.read_bytes()).hexdigest()=='91a9b273904338516c8be93d0e953a977bece82709deaa10346147a588701a6b'
 spec=importlib.util.spec_from_file_location('capture',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 m.execute('config/C32-only-L04-restore/runtime.json','1100bc9f5c4a938becd0233c00d513e2e1a3384adc95ea18f257f30f215fd526','20261017T2050-sglang-C32-L04-before-control-native-v1',mode='baseline')
if __name__=='__main__':main()
