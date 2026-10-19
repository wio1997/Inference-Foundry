from pathlib import Path
import json,socket
assert socket.gethostname()=='B5-5U-Atlas900RCKA3-33'
root=Path('/data/wio/glm5-3-vllm-execution/20261018-state-tail/indexer-k-graph-live1-20261019/node33')
assert not(root/'indexer-k-graph-overlap-probe/enable').exists()
(root/'indexer-k-graph-overlap-probe/disabled').touch()
print('GLM53_RESULT_JSON_BEGIN\n'+json.dumps(dict(completed=True,observer_disabled=True,marker_off=True))+'\nGLM53_RESULT_JSON_END')
