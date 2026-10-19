from pathlib import Path
import hashlib,json,socket,subprocess
assert socket.gethostname()=='B5-5U-Atlas900RCKA3-33'
cid='163c7646c30ab819514f112e0af510934aa592305f3ec37bd7e3b77a3edc2bdc'
d=json.loads(subprocess.check_output(['docker','inspect',cid]))[0]
assert d['State']['Running'] and d['Name']=='/glm53-rope-step-reuse-33-live1-20261019'
paths=['/vllm-workspace/vllm-ascend/vllm_ascend/ops/fused_moe/shared_experts.py','/vllm-workspace/vllm-ascend/vllm_ascend/attention/utils.py']
out=Path(__file__).resolve().parent/'artifacts';out.mkdir(exist_ok=False);rows=[]
for p in paths:
 q=subprocess.run(['docker','exec',cid,'cat',p],capture_output=True,timeout=20)
 if q.returncode:rows.append({'path':p,'missing':True});continue
 dest=out/Path(p).name;dest.write_bytes(q.stdout);rows.append({'path':p,'bytes':len(q.stdout),'sha256':hashlib.sha256(q.stdout).hexdigest()})
print('GLM53_RESULT_JSON_BEGIN\n'+json.dumps({'completed':True,'CID':cid,'sources':rows,'scope':'Read exact source only; no device imports or execution.'})+'\nGLM53_RESULT_JSON_END')
