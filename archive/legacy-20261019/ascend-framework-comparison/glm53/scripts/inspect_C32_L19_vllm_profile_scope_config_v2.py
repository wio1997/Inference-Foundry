"""New two-file read with a freshly authenticated master; preserve transport timeout v1."""
import hashlib
import json
from pathlib import Path
import inspect_C32_L19_vllm_profile_scope_config_v1 as prior
core=prior.core
core.RID='20261019T-L19-vllm-profile-scope-config-source-v2'
core.REVIEW='results/C32-L19-vllm-profile-scope-config-review-v2.json'
core.SOCKET='/Users/wio/work/sglang-glm53-pd/ssh-control-l20-20261019'
if __name__=='__main__':
    review=json.loads((core.ROOT/core.REVIEW).read_bytes())
    assert review['launcher_sha256']==hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    core.main()
