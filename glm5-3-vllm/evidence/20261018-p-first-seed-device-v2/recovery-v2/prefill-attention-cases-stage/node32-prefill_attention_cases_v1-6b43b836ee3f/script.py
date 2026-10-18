import hashlib,json
from pathlib import Path
old=Path('/data/wio/glm5-3-vllm-execution/20261018-state-tail/prefill-padding-live1-20261018/node32/prefill-padding-cases-v1.json')
new=Path('/data/wio/glm5-3-vllm-execution/20261018-state-tail/prefill-attention-live1-20261018/node32/prefill-padding-cases-v1.json')
b=old.read_bytes();assert hashlib.sha256(b).hexdigest()=='c82d7ecb6cf5d2a8786ce844d133e6e5627c743b714c17bf01055e34b3959906';assert new.parent.exists() and not new.exists();new.write_bytes(b)
print('GLM53_RESULT_JSON_BEGIN\n'+json.dumps({'staged':str(new),'sha256':hashlib.sha256(b).hexdigest(),'reuses_frozen_cases':True})+'\nGLM53_RESULT_JSON_END')
