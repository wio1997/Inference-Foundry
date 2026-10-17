"""Frozen L04 trial operation; actual execution via ZCode only."""
import hashlib,importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 path=ROOT/'scripts/glm53_sglang_metadata_launch_v1.py';assert hashlib.sha256(path.read_bytes()).hexdigest()=='c57a105a1803ca1d28bcd098f6ea722c77883757ecbd3180c00ae5798b8a4b2d'
 spec=importlib.util.spec_from_file_location('operation',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 m.execute('config/optimization-overlays/routing-metadata-v1/formal-C64-v2.json','a714ac75525bdd00812d757dcb8233481298d94d2748d3e914f3f6ed34c25761')
if __name__=='__main__':main()
