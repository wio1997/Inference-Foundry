"""Read-only full native capture after P-only startup; zero model requests."""
import hashlib,importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=ROOT/'scripts/glm53_C32_L12B_native_capture_v1.py';assert hashlib.sha256(p.read_bytes()).hexdigest()=='f16ac81adfe93a3ab724c769d6124fe2f7d06da2e3688c222bb9eb2bc53de5b2';s=importlib.util.spec_from_file_location('cap',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);m.execute('20261019T-L12B-sglang-startup-native-v1','results/C32-L12B-startup-native-capture-actual-review-v1.json')
if __name__=='__main__':main()
