from pathlib import Path
import json,socket,subprocess
assert socket.gethostname()=='B5-5U-Atlas900RCKA3-33'
root=Path('/data/wio/glm5-3-vllm-execution/20261018-state-tail/rope-step-reuse-live1-20261019/node33')
cid='163c7646c30ab819514f112e0af510934aa592305f3ec37bd7e3b77a3edc2bdc'
d=json.loads(subprocess.check_output(['docker','inspect',cid]))[0];assert d['State']['Running'] and d['Id']==json.loads((root/'launch.json').read_text())['container_id']
probe=root/'rope-step-reuse-probe';assert not(probe/'enable').exists() and (probe/'disabled').exists()
(probe/'disabled').unlink()
print('GLM53_RESULT_JSON_BEGIN\n'+json.dumps(dict(completed=True,exact_D_CID=cid,observer_enabled=True,candidate_off=True,scope='Fresh same-config P smoke only; restore observer disabled before quality/performance.'))+'\nGLM53_RESULT_JSON_END')
