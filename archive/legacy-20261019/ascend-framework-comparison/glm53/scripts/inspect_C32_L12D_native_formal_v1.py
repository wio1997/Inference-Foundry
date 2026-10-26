"""Read-only native evidence after the once-only C32 candidate formal client."""
import hashlib,importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=ROOT/'scripts/glm53_C32_L12D_formal_native_capture_v1.py';assert hashlib.sha256(p.read_bytes()).hexdigest()=='548833e16e2cf31dd985f54b13821b33eb54f73ef65b5a662fd9e9af2528562a'
 spec=importlib.util.spec_from_file_location('native',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.execute('20261019T-L12C-sglang-native-formal-capture-v1','results/C32-L12D-native-formal-capture-review-v1.json')
if __name__=='__main__':main()
