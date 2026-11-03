#!/usr/bin/env python3
"""Read fixed group construction source from one container; no imports or device access."""
import hashlib,json
from pathlib import Path
paths=[Path('/vllm-workspace/vllm/vllm/distributed/parallel_state.py'),Path('/vllm-workspace/vllm-ascend/vllm_ascend/distributed/parallel_state.py')]
out=[]
for p in paths:
 if not p.is_file():out.append({'path':str(p),'status':'MISSING'});continue
 raw=p.read_bytes()
 if len(raw)>500000:raise RuntimeError('source file exceeds bound')
 out.append({'path':str(p),'status':'READ','sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),'text':raw.decode('utf8')})
print(json.dumps({'scope':'Two fixed installed source paths; no imports, no NPU','files':out},ensure_ascii=False))
