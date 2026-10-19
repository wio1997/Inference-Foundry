from pathlib import Path
import json,socket
assert socket.gethostname()=='B5-5U-Atlas900RCKA3-33'
root=Path('/data/wio/glm5-3-vllm-execution/20261018-state-tail/rope-step-reuse-live1-20261019/node33')
assert not(root/'rope-step-reuse-probe/enable').exists()
(root/'rope-step-reuse-probe/disabled').touch()
print('GLM53_RESULT_JSON_BEGIN\n'+json.dumps(dict(completed=True,observer_disabled=True,marker_off=True))+'\nGLM53_RESULT_JSON_END')
