"""Frozen C32 L04 control; execute via ZCode exactly once."""
import hashlib,importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=ROOT/'scripts/glm53_sglang_metadata_launch_v1.py';assert hashlib.sha256(p.read_bytes()).hexdigest()=='c57a105a1803ca1d28bcd098f6ea722c77883757ecbd3180c00ae5798b8a4b2d'
 s=importlib.util.spec_from_file_location('operation',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
 assert 32==32; m.execute('config/C32-only-L04-restore/formal-C32-control-v3.json','922bd03e18e5463c74cbf3ff340670f03af9a7d08b9a1b7d283441ecac94bc4e')
if __name__=='__main__':main()
