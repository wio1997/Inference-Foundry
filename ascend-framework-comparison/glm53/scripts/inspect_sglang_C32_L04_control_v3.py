"""Frozen C32 L04 control; execute via ZCode exactly once."""
import hashlib,importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=ROOT/'scripts/glm53_sglang_C32_L04_native_capture_v1.py';assert hashlib.sha256(p.read_bytes()).hexdigest()=='91a9b273904338516c8be93d0e953a977bece82709deaa10346147a588701a6b'
 s=importlib.util.spec_from_file_location('operation',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
 m.execute('config/C32-only-L04-restore/formal-C32-control-v3.json','922bd03e18e5463c74cbf3ff340670f03af9a7d08b9a1b7d283441ecac94bc4e','20261017T2054-sglang-C32-L04-control-native-v3')
if __name__=='__main__':main()
