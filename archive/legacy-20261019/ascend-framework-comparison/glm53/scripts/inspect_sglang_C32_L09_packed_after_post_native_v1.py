"""Read-only post-formal native capture; retain raw failure/nulls and original D metrics."""
from pathlib import Path
import hashlib,importlib.util
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=ROOT/'scripts/glm53_sglang_C32_packed_normal_native_capture_v1.py';assert hashlib.sha256(p.read_bytes()).hexdigest()=='89b278ede3f7774460ea26bba8de8f711abc38b09fc0279d0776fd3d18fb144d'
 spec=importlib.util.spec_from_file_location('capture',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 m.execute('config/C32-packed-normal-runtime-v1/formal-C32-packed-after-v1.json','522402248e183a0e46a08346f67c451bb1bcde9e3f9fb22aad9db809e00d1ca5','20261018T-L09-sglang-packed-after-post-native-v1',mode='formal')
if __name__=='__main__':main()
