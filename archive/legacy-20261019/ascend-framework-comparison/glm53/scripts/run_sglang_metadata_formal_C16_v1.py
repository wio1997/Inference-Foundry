"""Frozen L04 trial operation; actual execution via ZCode only."""
import hashlib,importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 path=ROOT/'scripts/glm53_sglang_paired_launch_v1.py';assert hashlib.sha256(path.read_bytes()).hexdigest()=='b227e29c24f0f85e4462457db3623a421c35cd1d3a165392e91f96bfd91f077f'
 spec=importlib.util.spec_from_file_location('operation',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 m.execute('config/optimization-overlays/routing-metadata-v1/formal-C16.json','7213e135f3ccc44b6a34f937e0f0e5e4738beb5156663c302d8ab1a0d15f233f')
if __name__=='__main__':main()
