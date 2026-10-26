"""Resolve the actual hybrid-DP profile control scope from installed config code."""
import hashlib
import json
from pathlib import Path
import inspect_C32_L19_vllm_profile_fanout_v1 as prior

core=prior.core
core.RID='20261019T-L19-vllm-profile-scope-config-source-v1'
core.REVIEW='results/C32-L19-vllm-profile-scope-config-review-v1.json'
lines=core.PROBE.splitlines()
assert lines[1].startswith('paths=')
lines[1]="paths=['/vllm-workspace/vllm/vllm/config/parallel.py','/vllm-workspace/vllm/vllm/engine/arg_utils.py']"
core.PROBE='\n'.join(lines).replace('four fixed installed source files only','two fixed installed source files only')+'\n'

if __name__=='__main__':
    review=json.loads((core.ROOT/core.REVIEW).read_bytes())
    assert review['launcher_sha256']==hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    core.main()
