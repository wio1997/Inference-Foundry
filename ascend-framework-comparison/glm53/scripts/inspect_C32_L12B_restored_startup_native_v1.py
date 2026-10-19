"""Read-only mixed-root native capture after tested L10 P restore."""
import hashlib,importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=ROOT/'scripts/glm53_C32_L12B_restored_native_capture_v1.py';assert hashlib.sha256(p.read_bytes()).hexdigest()=='1d1832c03cfc1e54a1daa30682c977e1ee3fccbfa8fc7b8d07b92b1d65ed4c2a';s=importlib.util.spec_from_file_location('cap',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);m.execute('20261019T-L12B-sglang-restored-L10-startup-native-v1','results/C32-L12B-restored-startup-native-capture-review-v1.json')
if __name__=='__main__':main()
