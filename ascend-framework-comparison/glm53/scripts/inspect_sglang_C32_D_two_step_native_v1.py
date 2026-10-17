"""Frozen C32 diagnostic; exact once through ZCode only."""
import hashlib,importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=ROOT/'scripts/glm53_C32_D_native_profile_capture_v1.py';assert hashlib.sha256(p.read_bytes()).hexdigest()=='e1b5595e5b9ff29c43df4ef6ef68f90f583342e009717d3d545094476da785c6'
 s=importlib.util.spec_from_file_location('controller',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
 m.execute('config/C32-only-L04-restore/D-two-step-diagnostic-v1.json','f2ab656c9e4b5fde35448decbff6b13cff209f5b76fa91a0577bbb91b73efc60','20261017T2060-sglang-C32-D-two-step-native-v1')
if __name__=='__main__':main()
