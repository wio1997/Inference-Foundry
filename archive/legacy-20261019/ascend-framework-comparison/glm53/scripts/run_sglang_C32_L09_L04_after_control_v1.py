"""Exact once fresh L04 C32 temporal control, same canonical fullclock protocol."""
from pathlib import Path
import hashlib,importlib.util,json
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=ROOT/'results/C32-L09-L04-after-control-native-baseline-v1.json';assert hashlib.sha256(p.read_bytes()).hexdigest()=='c0676c2651c47306edf0cd47f97f4cb75669a59f2b0f54049973873223332c07';assert json.loads(p.read_text())['pass']
 p=ROOT/'scripts/glm53_sglang_metadata_launch_v1.py';assert hashlib.sha256(p.read_bytes()).hexdigest()=='c57a105a1803ca1d28bcd098f6ea722c77883757ecbd3180c00ae5798b8a4b2d'
 spec=importlib.util.spec_from_file_location('operation',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 m.execute('config/C32-packed-normal-runtime-v1/formal-C32-L04-after-control-v1.json','d6a9bbba88a63258e5d4c97f3afa28756251c07bac48a714364f283ed2bd1407')
if __name__=='__main__':main()
