"""Serving entrypoint for the frozen project overlay, executed only by ZCode."""
import hashlib
import json
import os
from pathlib import Path
import sys

root = Path('/opt/glm53/source')
manifest = json.loads((root/'manifest.json').read_text())
for entry in manifest['files']:
    assert hashlib.sha256(Path(entry['container_path']).read_bytes()).hexdigest() == entry['patched_sha256']
for entry in manifest['helpers']:
    assert hashlib.sha256((root/entry['local']).read_bytes()).hexdigest() == entry['sha256']
node = int(sys.argv[1])
config_path = Path(f'/opt/glm53/config/node-{node}.json')
config = json.loads(config_path.read_text())
assert node in (30,31,43,45) and config['node'] == node
assert config['kv_transfer']['kv_connector'] == 'MultiConnector'
assert config['parallel_intent']['role_die'] == 32
args = [sys.executable, '-m', 'vllm.entrypoints.openai.api_server',
        '--host', '0.0.0.0', '--port', str(config['api_port']), *config['launch_args']]
print('GLM53_SERVE_ENTRY=' + json.dumps({'project_id':'glm53','node':node,
      'pid':os.getpid(),'config_sha256':hashlib.sha256(config_path.read_bytes()).hexdigest(),
      'args':args,'phase':config['phase']}), flush=True)
os.execv(sys.executable, args)
