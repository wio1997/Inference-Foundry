from pathlib import Path
import json
import socket
import subprocess
assert socket.gethostname()=='B5-5U-Atlas900RCKA3-33'
root=Path('/data/wio/glm5-3-vllm-execution/20261018-state-tail/early-copy-shadow-live1-20261019/node33')
cid='9b788715cb6e34139c2e9a70ef5bcf49009dda507164bf11e9e00cd07c163863'
d=json.loads(subprocess.check_output(['docker','inspect',cid]))[0]
assert d['State']['Running'] and d['Id']==json.loads((root/'launch.json').read_text())['container_id']
assert not(root/'early-sample-copy-shadow/enable').exists()
(root/'early-sample-copy-shadow/disabled').touch()
print('GLM53_RESULT_JSON_BEGIN\n'+json.dumps(dict(completed=True,shadow_default_off=True,metadata_marker_for_cached_contract=True,note='Shadow gate records remain required while selected; no formal performance phase permitted.'))+'\nGLM53_RESULT_JSON_END')
