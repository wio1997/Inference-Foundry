"""Frozen L04 trial operation; actual execution via ZCode only."""
import hashlib,importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 path=ROOT/'scripts/glm53_sglang_metadata_native_capture_v1.py';assert hashlib.sha256(path.read_bytes()).hexdigest()=='520bb2ef5cca591218912fccea2a494cbbb6b6eec9b8e54d3a152fb8342dba14'
 spec=importlib.util.spec_from_file_location('operation',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 m.execute('config/optimization-overlays/routing-metadata-v1/formal-C32-v2.json','a99a84aa4895e4d4f3e2fcbba948658509cc7f840d8ac86a46eedd31f43e61ec','20261017T1992-sglang-L04-C32-native-v2')
if __name__=='__main__':main()
