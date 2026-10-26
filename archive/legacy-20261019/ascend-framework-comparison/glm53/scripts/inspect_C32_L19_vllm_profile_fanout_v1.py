"""Read actual profile fanout/worker path before deciding which native endpoint to start."""
import hashlib
import json
from pathlib import Path
import inspect_C32_L19_vllm_profile_readiness_v1 as core

core.RID='20261019T-L19-vllm-profile-fanout-source-v1'
core.REVIEW='results/C32-L19-vllm-profile-fanout-review-v1.json'
core.PROBE=r'''import pathlib,json,hashlib,gzip,base64,sys
paths=['/vllm-workspace/vllm/vllm/v1/engine/async_llm.py','/vllm-workspace/vllm/vllm/v1/engine/core_client.py','/vllm-workspace/vllm-ascend/vllm_ascend/platform.py','/vllm-workspace/vllm-ascend/vllm_ascend/worker/worker.py']
out={'project_id':'glm53','files':[],'missing':[],'worker_processes':[],'existing_profile_traces':[],'scope':'four fixed installed source files only; process/profile discovery not repeated','model_requests':0,'SDK_init_calls':0};total=0
for name in paths:
 p=pathlib.Path(name)
 if not p.is_file():out['missing'].append(name);continue
 assert not p.is_symlink() and any(p.resolve().is_relative_to(pathlib.Path(r).resolve()) for r in ['/vllm-workspace/vllm','/vllm-workspace/vllm-ascend'])
 raw=p.read_bytes();total+=len(raw);assert len(raw)<=2*1024**2 and total<=4*1024**2
 out['files'].append({'path':name,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'source':raw.decode()})
assert not any(k.startswith(('torch','vllm','sglang','memcache_hybrid')) for k in sys.modules)
raw=json.dumps(out).encode();print(json.dumps({'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'gzip_base64':base64.b64encode(gzip.compress(raw,mtime=0)).decode()}))
'''
original_plans=core.plans


def plans():
    inv,items=original_plans()
    return inv,{43:items[43]}


core.plans=plans
if __name__=='__main__':
    review=json.loads((core.ROOT/core.REVIEW).read_bytes())
    assert review['launcher_sha256']==hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    core.main()
