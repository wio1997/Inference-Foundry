import json,subprocess,socket
assert socket.gethostname()=='B4-33U-Atlas900RCKA3-32'
code="import pathlib,json,hashlib;p=pathlib.Path('/vllm-workspace/vllm-ascend/vllm_ascend/profiler/torch_npu_profiler.py');b=p.read_bytes();print(json.dumps({'path':str(p),'sha256':hashlib.sha256(b).hexdigest(),'text':b.decode()}))"
x=json.loads(subprocess.check_output(['docker','exec','-i','4cb8a338da590f0d0fd624892b2af7a20884029c7418c05ada75f5dbf3bcb157','python3','-'],input=code.encode(),timeout=20))
print('GLM53_RESULT_JSON_BEGIN\n'+json.dumps(x)+'\nGLM53_RESULT_JSON_END')
