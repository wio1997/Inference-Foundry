import hashlib,importlib.util,json
from pathlib import Path
root=Path('/vllm-workspace/vllm-ascend');spec=importlib.util.find_spec('batch_invariant_ops');files=[]
for p in sorted(root.rglob('*batch_invar*')):
 if p.is_file() and p.suffix in ('.py','.md','.sh'):
  data=p.read_bytes();files.append({'path':str(p),'sha256':hashlib.sha256(data).hexdigest(),'source':data.decode() if len(data)<80000 else None,'bytes':len(data)})
print('GLM53_RESULT_JSON_BEGIN\n'+json.dumps({'installed_spec':None if spec is None else str(spec.origin),'files':files})+'\nGLM53_RESULT_JSON_END')
