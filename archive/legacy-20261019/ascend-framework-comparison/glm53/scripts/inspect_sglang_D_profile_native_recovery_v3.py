"""Recover failed1924 captures with one SSH at a time; preserve original failure."""
import hashlib,importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 path=ROOT/'scripts/glm53_sglang_D_profile_native_recovery_v3.py';assert hashlib.sha256(path.read_bytes()).hexdigest()=='6310fd7b0cd8dcb7aa767412d7dbf30e0b96c3027c9ab34f653978cc67b569ef'
 spec=importlib.util.spec_from_file_location('recovery',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 m.execute('config/diagnostic-runs-v1/sglang-C16-D-profile.json','5a91c2804522b430827da0338825dcd6f549db90bfdd0bdaee10a4c97aac6a24','20261017T1940-sglang-D-profile-native-recovery-v2')
if __name__=='__main__':main()
