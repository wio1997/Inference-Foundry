import hashlib,json
from pathlib import Path
root=Path('/vllm-workspace/vllm-ascend/vllm_ascend');rows=[]
for p in sorted(root.rglob('*.py')):
 s=p.read_text();lines=s.splitlines();positions=[i for i,line in enumerate(lines) if 'VLLM_BATCH_INVARIANT' in line]
 if positions:
  selected=set(j for i in positions for j in range(max(0,i-3),min(len(lines),i+9)))
  rows.append({'path':str(p),'sha256':hashlib.sha256(s.encode()).hexdigest(),'snippets':[{str(i+1):lines[i]} for i in sorted(selected)]})
print('GLM53_RESULT_JSON_BEGIN\n'+json.dumps({'coverage':rows})+'\nGLM53_RESULT_JSON_END')
