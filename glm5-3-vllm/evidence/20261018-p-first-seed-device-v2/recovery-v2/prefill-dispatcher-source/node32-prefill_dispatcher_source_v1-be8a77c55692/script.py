import json,subprocess,socket
assert socket.gethostname()=='B4-33U-Atlas900RCKA3-32'
code="""import pathlib,json,hashlib
files=[]
paths=[pathlib.Path('/vllm-workspace/vllm/vllm/v1/cudagraph_dispatcher.py')]
base=pathlib.Path('/vllm-workspace/vllm-ascend/vllm_ascend/patch')
for p in base.rglob('*.py'):
 text=p.read_text()
 if 'CudagraphDispatcher' in text or 'pad_for_cudagraph' in text:paths.append(p)
for p in paths:
 b=p.read_bytes();assert len(b)<200000
 files.append({'path':str(p),'sha256':hashlib.sha256(b).hexdigest(),'text':b.decode()})
print(json.dumps({'files':files}))
"""
x=json.loads(subprocess.check_output(['docker','exec','-i','4cb8a338da590f0d0fd624892b2af7a20884029c7418c05ada75f5dbf3bcb157','python3','-'],input=code.encode(),timeout=25))
print('GLM53_RESULT_JSON_BEGIN\n'+json.dumps(x)+'\nGLM53_RESULT_JSON_END')
