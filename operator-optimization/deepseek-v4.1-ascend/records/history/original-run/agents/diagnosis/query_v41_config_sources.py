#!/usr/bin/env python3
"""Read exact source files only; no imports, instantiation, model or NPU."""
import hashlib,json
from pathlib import Path
files={
 'v41_config':('/vllm-workspace/vllm/vllm/transformers_utils/configs/deepseek_v41.py','5e7052bca7ce79dfa801a3bff20fa2c287ec15d82cf999d003f309b30bf6c1f0',True),
 'config_loader':('/vllm-workspace/vllm/vllm/transformers_utils/config.py',None,True),
 'config_registry':('/vllm-workspace/vllm/vllm/transformers_utils/configs/__init__.py',None,False),
 'model_config':('/vllm-workspace/vllm/vllm/config/model.py',None,True),
 'transformers_base':('/usr/local/python3.12.13/lib/python3.12/site-packages/transformers/configuration_utils.py',None,True),
}
out={};total=0
for key,(name,expected,required) in files.items():
 p=Path(name)
 if not p.is_file():
  if required:raise RuntimeError('STOP missing required source '+key)
  out[key]={'path':name,'status':'NOT_VISIBLE'};continue
 if p.stat().st_size>1024*1024:raise RuntimeError('STOP oversized source '+key)
 data=p.read_bytes();digest=hashlib.sha256(data).hexdigest()
 if expected and digest!=expected:raise RuntimeError('STOP source mismatch '+key)
 text=data.decode('utf-8');total+=len(data)
 if total>3*1024*1024:raise RuntimeError('STOP oversized aggregate')
 out[key]={'path':name,'status':'OK','sha256':digest,'bytes':len(data),'source':text}
print(json.dumps({'sources':out,'no_device_work':True,'scope':'exact static source files; current image, no historic instance'},ensure_ascii=False))
