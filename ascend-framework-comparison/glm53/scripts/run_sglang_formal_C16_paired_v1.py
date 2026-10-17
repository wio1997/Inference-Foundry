"""Frozen paired-stage SG operation; actual execution via ZCode only."""
import hashlib,importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 path=ROOT/'scripts/glm53_sglang_paired_launch_v1.py';assert hashlib.sha256(path.read_bytes()).hexdigest()=='b227e29c24f0f85e4462457db3623a421c35cd1d3a165392e91f96bfd91f077f'
 spec=importlib.util.spec_from_file_location('operation',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 m.execute('config/formal-runs-v2/sglang-C16.json','349315219c75a96d86a083c479c11c77a3f3811a8bdb29e9523795458a1201f6')
if __name__=='__main__':main()
