#!/usr/bin/env python3
"""Read exact TokenDispatcher source and MC2 call site, with no imports."""
import hashlib,json
from pathlib import Path
paths=[Path('/vllm-workspace/vllm-ascend/vllm_ascend/ops/fused_moe/token_dispatcher.py'),Path('/vllm-workspace/vllm-ascend/vllm_ascend/ops/fused_moe/moe_comm_method.py')]
out=[]
for p in paths:
 if not p.is_file():out.append({'path':str(p),'status':'MISSING'});continue
 raw=p.read_bytes()
 if len(raw)>400000:raise RuntimeError('file too large')
 out.append({'path':str(p),'status':'READ','sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),'text':raw.decode()})
print(json.dumps({'scope':'Two fixed same-image files; no imports or NPU use','files':out},ensure_ascii=False))
