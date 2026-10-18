import json
from pathlib import Path
import socket
import subprocess
assert socket.gethostname()=='B5-5U-Atlas900RCKA3-33'
root=Path('/data/wio/glm5-3-vllm-execution/20261018-state-tail/seed-live2-20261018/node33');cid=json.loads((root/'launch.json').read_text())['container_id'];plan=json.loads((root/'plan.json').read_text());ins=json.loads(subprocess.check_output(['docker','inspect',cid]))[0];assert cid=='cfa70a8cc3af142a57402d7e51a21677ff2f0a2fcae4915aedac691849c51189' and ins['State']['Running'] and ins['Name']=='/'+plan['container']['name']
p=root/'adopt-trace/disabled';assert p.is_file() and not p.is_symlink();p.unlink()
print('GLM53_RESULT_JSON_BEGIN\n'+json.dumps({'CID':cid,'D_host_trace_enabled':True,'model_restarted':False})+'\nGLM53_RESULT_JSON_END')
