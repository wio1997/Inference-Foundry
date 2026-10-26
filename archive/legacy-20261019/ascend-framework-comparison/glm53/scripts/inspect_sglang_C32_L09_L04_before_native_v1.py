"""Exact once C32L09beforecontrol wrapper; ZCode execution only."""
import hashlib,importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=ROOT/'scripts/glm53_sglang_C32_L09_L04_before_native_capture_v1.py';assert hashlib.sha256(p.read_bytes()).hexdigest()=='4765127f3133ca949627d97624a2da132787cdd9f579ad14de76e49935b1e93a'
 s=importlib.util.spec_from_file_location('operation',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
 m.execute('config/C32-packed-normal-runtime-v1/formal-C32-L04-before-v1.json','bc152c7e3997422ef46eafe04f0da492bab81d4d93ef5353e4bd16fbcea54503','20261018T-L09-sglang-L04-before-native-v1',mode="baseline")
if __name__=='__main__':main()
