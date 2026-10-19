import json,subprocess,socket
assert socket.gethostname()=='B4-33U-Atlas900RCKA3-32'
cid='4cb8a338da590f0d0fd624892b2af7a20884029c7418c05ada75f5dbf3bcb157'
code="""import json,pathlib,hashlib
roots=[pathlib.Path('/vllm-workspace/vllm/vllm/distributed/kv_transfer/kv_connector/v1'),pathlib.Path('/vllm-workspace/vllm-ascend/vllm_ascend/distributed/kv_transfer')]
files=[]
for root in roots:
 for p in root.rglob('*mooncake*.py'):
  text=p.read_text()
  assert len(text)<250000
  files.append({'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'text':text})
print(json.dumps({'files':files}))
"""
x=json.loads(subprocess.check_output(['docker','exec','-i',cid,'python3','-'],input=code.encode(),timeout=25));assert x['files']
print('GLM53_RESULT_JSON_BEGIN\n'+json.dumps(x)+'\nGLM53_RESULT_JSON_END')
