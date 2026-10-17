"""Frozen paired-stage SG operation; actual execution via ZCode only."""
import hashlib,importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 path=ROOT/'scripts/glm53_sglang_formal_native_capture_v1.py';assert hashlib.sha256(path.read_bytes()).hexdigest()=='89373b845e5623f41fa444d1052389bda4385d99a4bc81fc48e511a3ea25a71a'
 spec=importlib.util.spec_from_file_location('operation',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 m.execute('config/formal-runs-v2/sglang-C16.json','349315219c75a96d86a083c479c11c77a3f3811a8bdb29e9523795458a1201f6','20261017T1894-sglang-formal-C16-native-v4')
if __name__=='__main__':main()
