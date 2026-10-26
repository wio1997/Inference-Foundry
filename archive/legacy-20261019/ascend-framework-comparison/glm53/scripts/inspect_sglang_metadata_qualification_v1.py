"""Fresh SG1954 functional/native qualification capture; ZCode only."""
import hashlib,importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 path=ROOT/'scripts/glm53_sglang_metadata_native_capture_v1.py'
 assert hashlib.sha256(path.read_bytes()).hexdigest()=='520bb2ef5cca591218912fccea2a494cbbb6b6eec9b8e54d3a152fb8342dba14'
 spec=importlib.util.spec_from_file_location('capture',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 m.execute('config/optimization-overlays/routing-metadata-v1/runtime.json','3818f8b619234767815380604f727af854a57e68ed17ff8aa4819ef636698856','20261017T1966-sglang-metadata-native-qualification-v1',mode='qualification')
if __name__=='__main__':main()
