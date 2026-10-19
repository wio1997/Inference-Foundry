from pathlib import Path
import hashlib
import json
import socket
import subprocess

assert socket.gethostname() == 'B5-5U-Atlas900RCKA3-33'
root = Path('/data/wio/glm5-3-vllm-execution/20261018-state-tail/early-copy-shadow-live1-20261019/node33')
plan = json.loads((root/'plan.json').read_text()); launch = json.loads((root/'launch.json').read_text())
cid = '9b788715cb6e34139c2e9a70ef5bcf49009dda507164bf11e9e00cd07c163863'
d = json.loads(subprocess.check_output(['docker','inspect',cid]))[0]
assert d['State']['Running'] and d['Id']==launch['container_id'] and d['Name']=='/'+plan['container']['name'] and d['Image']==plan['image']['id']
env = dict(z.split('=',1) for z in d['Config']['Env'] if '=' in z)
assert env['GLM53_EARLY_SAMPLE_COPY_SHADOW']=='1' and env['GLM53_MOE_PREFIX_GMM2_COUNT_PROBE']=='0'
assert env['GLM53_ROPE_CACHE_LAYOUT_PROBE']=='1'
src=root.parent.parent/'prefix-count-live1-20261019/node33/prefill-padding-cases-v1.json'
data=src.read_bytes();assert hashlib.sha256(data).hexdigest()=='c82d7ecb6cf5d2a8786ce844d133e6e5627c743b714c17bf01055e34b3959906'
dest=root/'prefill-padding-cases-v1.json';assert not dest.exists();dest.write_bytes(data)
for name in ['rope-cache-layout-probe','early-sample-copy-shadow']:(root/name).mkdir(exist_ok=True)
(root/'rope-cache-layout-probe/enable').touch();(root/'rope-cache-layout-probe/disabled').touch()
assert not(root/'early-sample-copy-shadow/enable').exists() and not(root/'early-sample-copy-shadow/disabled').exists()
print('GLM53_RESULT_JSON_BEGIN\n'+json.dumps(dict(completed=True,CID=cid,RoPE_enabled_both_arms=True,all_MoE_native=True,shadow_default_off=True,fixture_sha256=hashlib.sha256(data).hexdigest()))+'\nGLM53_RESULT_JSON_END')
