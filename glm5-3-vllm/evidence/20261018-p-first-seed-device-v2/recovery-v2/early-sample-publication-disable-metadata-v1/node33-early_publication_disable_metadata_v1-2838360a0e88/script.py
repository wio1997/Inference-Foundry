from pathlib import Path
import json,socket,subprocess
assert socket.gethostname()=='B5-5U-Atlas900RCKA3-33'
root=Path('/data/wio/glm5-3-vllm-execution/20261018-state-tail/early-publication-live1-20261019/node33')
cid='8d81520a203ef0baaf71860185d06fcf1d0d60e907ba1e0029852c7cb8f62cfd'
d=json.loads(subprocess.check_output(['docker','inspect',cid]))[0]
assert d['State']['Running'] and d['Id']==json.loads((root/'launch.json').read_text())['container_id']
probe=root/'early-sample-publication-probe'
assert not(probe/'enable').exists() and not(probe/'abort-test-once.json').exists()
assert json.loads((root/'early-sample-publication-native-abort-v1/result.json').read_text())['completed']
(probe/'disabled').touch()
print('GLM53_RESULT_JSON_BEGIN\n'+json.dumps(dict(completed=True,metadata_disabled=True,publication_default_off=True,late_bitwise_checks_remain_active=True))+'\nGLM53_RESULT_JSON_END')
