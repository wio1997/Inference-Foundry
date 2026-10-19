"""Exact once C32L09beforecontrol wrapper; ZCode execution only."""
import hashlib,importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 import json
 baseline=json.loads((ROOT/'results/C32-L09-L04-before-native-audit-v1.json').read_text());assert baseline['pass'] and baseline['run_id']=='20261018T-L09-sglang-L04-before-native-v1'
 p=ROOT/'scripts/glm53_sglang_metadata_launch_v1.py';assert hashlib.sha256(p.read_bytes()).hexdigest()=='c57a105a1803ca1d28bcd098f6ea722c77883757ecbd3180c00ae5798b8a4b2d'
 s=importlib.util.spec_from_file_location('operation',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
 m.execute('config/C32-packed-normal-runtime-v1/formal-C32-L04-before-v1.json','bc152c7e3997422ef46eafe04f0da492bab81d4d93ef5353e4bd16fbcea54503')
if __name__=='__main__':main()
