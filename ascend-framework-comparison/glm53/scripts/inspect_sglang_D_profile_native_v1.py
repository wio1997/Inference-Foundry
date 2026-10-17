"""Capture completed diagnostic and node-side trace manifests only; no trace download."""
import hashlib,importlib.util,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 run=ROOT/'runs/20261017T1920-sglang-formal-C16-D-profile-v1'
 rpc=json.loads((run/'controller-rpc-state.json').read_text());assert rpc['done_received'] and rpc['connection_rc'] is not None
 assert (run/'D-profile-control-receipt.json').is_file(),'Wait for profile control SSH to close before capture'
 path=ROOT/'scripts/glm53_sglang_D_profile_native_capture_v1.py';assert hashlib.sha256(path.read_bytes()).hexdigest()=='4dc0727dffc582d6dd2bfeae86d0484d877fe95d462da6f314d58d34dfb4cce1'
 spec=importlib.util.spec_from_file_location('capture',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 module.execute('config/diagnostic-runs-v1/sglang-C16-D-profile.json','5a91c2804522b430827da0338825dcd6f549db90bfdd0bdaee10a4c97aac6a24','20261017T1924-sglang-D-profile-native-v1')
if __name__=='__main__':main()
