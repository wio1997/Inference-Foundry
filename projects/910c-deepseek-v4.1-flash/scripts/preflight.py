#!/usr/bin/env python3
"""Inspect checkpoint headers without reading all tensor payloads."""
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
from datetime import datetime

root = Path(os.environ['WIO_ROOT'])
project = root / 'projects/910c-deepseek-v4.1-flash'
model = root / 'DeepSeek-V4.1-Flash-1017-w8a8'
stamp = datetime.now().astimezone().strftime('%Y%m%dT%H%M%S%z')
out = project / 'evidence' / stamp
out.mkdir(parents=True, exist_ok=True)
def command(*args):
    p = subprocess.run(args, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return {'returncode': p.returncode, 'output': p.stdout}
config = json.loads((model / 'config.json').read_text())
index_path = model / 'quant_model_weights.safetensors.index.json'
index = json.loads(index_path.read_text())
weights = index['weight_map']
errors, shards = [], []
for name in sorted(set(weights.values())):
    path = model / name
    if not path.is_file():
        errors.append(f'missing shard: {name}')
        continue
    size = path.stat().st_size
    with path.open('rb') as f:
        header_size = struct.unpack('<Q', f.read(8))[0]
        if header_size > 100_000_000 or header_size + 8 > size:
            errors.append(f'invalid header size: {name}')
            continue
        header = json.loads(f.read(header_size))
    data_size = size - 8 - header_size
    for key, tensor in header.items():
        if key == '__metadata__':
            continue
        begin, end = tensor['data_offsets']
        if not 0 <= begin <= end <= data_size:
            errors.append(f'invalid offset: {name}:{key}')
    missing = [key for key, shard in weights.items() if shard == name and key not in header]
    if missing:
        errors.append(f'{name}: {len(missing)} index tensors absent')
    shards.append({'name': name, 'bytes': size, 'header_bytes': header_size})
npu = command('npu-smi', 'info')
(out / 'npu-smi.txt').write_text(npu['output'])
busy = [line for line in npu['output'].splitlines() if 'python' in line.lower() or 'vllm' in line.lower()]
report = {
    'time': datetime.now().astimezone().isoformat(),
    'host': command('hostname')['output'].strip(),
    'identity': command('whoami')['output'].strip(),
    'model_path': str(model), 'architectures': config.get('architectures'),
    'config_sha256': hashlib.sha256((model / 'config.json').read_bytes()).hexdigest(),
    'index_sha256': hashlib.sha256(index_path.read_bytes()).hexdigest(),
    'indexed_tensors': len(weights), 'shard_count': len(shards),
    'shard_bytes': sum(s['bytes'] for s in shards),
    'shards': shards, 'errors': errors,
    'verification_scope': 'all indexed shard headers and offsets; no full payload SHA256',
    'npu': npu, 'busy_lines': busy,
    'disk': command('df', '-hT', str(root)),
    'memory': command('free', '-h'),
}
(out / 'preflight.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({k: v for k, v in report.items() if k not in ('shards', 'npu')}, ensure_ascii=False, indent=2))
print('EVIDENCE_DIR=' + str(out))
sys.exit(1 if errors else 0)
