"""Read-only mixed-root native capture after tested L10 P restore."""
import hashlib,importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=ROOT/'scripts/glm53_C32_L12C_native_capture_v1.py';assert hashlib.sha256(p.read_bytes()).hexdigest()=='e08773c054c1e6a636e9ff6de3e5a0f22257c14ddd6b16e53b0bf36a31d86610';s=importlib.util.spec_from_file_location('cap',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);m.execute('20261019T-L12C-sglang-native-startup-v1','results/C32-L12C-native-startup-capture-review-v1.json')
if __name__=='__main__':main()
