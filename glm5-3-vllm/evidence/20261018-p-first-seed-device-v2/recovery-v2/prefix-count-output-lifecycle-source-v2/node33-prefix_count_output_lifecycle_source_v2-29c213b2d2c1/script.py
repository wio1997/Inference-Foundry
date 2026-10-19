"""Read exact loaded-image source without importing model modules or changing state."""
from pathlib import Path
import hashlib
import json
import socket
import subprocess

assert socket.gethostname() == 'B5-5U-Atlas900RCKA3-33'
cid = '8f434957c821dec55a77d352a68d16c14f2558d0b4c7e11287e0e37181111fdd'
d = json.loads(subprocess.check_output(['docker', 'inspect', cid]))[0]
assert d['Name'] == '/glm53-prefix-count-33-live1-20261019'
assert d['Image'] == 'sha256:090eeee66e347509127601e565b971a0801701e32e4e865d89cc8dec72ca7d83'
paths = [
    'vllm/vllm/v1/worker/gpu_input_batch.py',
    'vllm/vllm/v1/worker/gpu_worker.py',
    'vllm/vllm/v1/engine/core.py',
    'vllm/vllm/v1/executor/multiproc_executor.py',
    'vllm/vllm/v1/worker/kv_connector_model_runner_mixin.py',
    'vllm-ascend/vllm_ascend/spec_decode/llm_base_proposer.py',
    'vllm-ascend/vllm_ascend/spec_decode/eagle_proposer.py',
    'vllm-ascend/vllm_ascend/ops/rotary_embedding.py',
    'vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake/connector.py',
    'vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake/pull_worker.py',
]
workers = json.loads(subprocess.check_output(['docker', 'exec', cid, 'python3', '-c', "from pathlib import Path;import json;print(json.dumps([str(p.relative_to('/vllm-workspace')) for p in Path('/vllm-workspace/vllm-ascend/vllm_ascend/worker').glob('worker*.py')]))"], timeout=20))
assert workers
paths.extend(workers)
out = Path(__file__).resolve().parent / 'artifacts'
out.mkdir(exist_ok=False)
rows = []
for rel in paths:
    data = subprocess.check_output(['docker', 'exec', cid, 'cat', '/vllm-workspace/' + rel], timeout=20)
    assert len(data) < 2 * 1024**2
    target = out / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    rows.append({'path': rel, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()})
print('GLM53_RESULT_JSON_BEGIN\n' + json.dumps({'completed': True, 'CID': cid, 'files': rows, 'scope': 'Read-only source files; no model imports, device work, flags, stops or restarts.'}) + '\nGLM53_RESULT_JSON_END')
