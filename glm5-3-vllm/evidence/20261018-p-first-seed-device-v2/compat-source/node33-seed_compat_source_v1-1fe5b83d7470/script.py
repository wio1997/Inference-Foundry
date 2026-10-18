import hashlib,json
from pathlib import Path
rows=[]
for p in Path('/vllm-workspace/vllm-ascend/vllm_ascend/patch').rglob('*.py'):
 s=p.read_text()
 if 'spec_token_ids' in s and 'ModelRunnerOutput' in s:
  rows.append({'path':str(p),'sha256':hashlib.sha256(s.encode()).hexdigest(),'source':s})
print('GLM53_RESULT_JSON_BEGIN\n'+json.dumps({'files':rows},indent=2)+'\nGLM53_RESULT_JSON_END')
