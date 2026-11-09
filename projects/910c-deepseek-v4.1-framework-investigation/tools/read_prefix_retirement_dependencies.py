#!/usr/bin/env python3
"""Read-only immediate cache/retirement callees, separate from primary pins."""
from hashlib import sha256
import json
from pathlib import Path
import subprocess
from read_prefix_retirement_identity import FROZEN, OUT, REMOTE

NAMES = [
    'vllm-ascend/vllm_ascend/core/single_type_kv_cache_manager.py',
    'vllm-ascend/vllm_ascend/core/kv_cache_interface.py',
    'vllm-ascend/vllm_ascend/worker/utils.py',
    'vllm-ascend/vllm_ascend/model_executor/offloader/base.py',
    'vllm/vllm/model_executor/offloader/base.py',
    'vllm/vllm/v1/kv_cache_interface.py',
    'vllm/vllm/v1/worker/gpu_model_runner.py',
    'vllm/vllm/config/vllm.py',
    'vllm/vllm/v1/request.py',
]


def main():
    old = []
    for p in OUT.glob('*identity.json'):
        try:
            old.extend(json.loads(p.read_text()).get('sources', []))
        except (ValueError, OSError):
            continue
    spec = []
    for name in NAMES:
        runtime = '/vllm-workspace/' + name
        found = next((r['local_path'] for r in old
                      if r.get('runtime_path') == runtime and r.get('local_path')
                      and Path(r['local_path']).is_file()), None)
        p = Path(found) if found else FROZEN / name
        spec.append(dict(runtime_path=runtime, local_path=str(p),
                         local_sha256=sha256(p.read_bytes()).hexdigest() if p.is_file() else None))
    r = subprocess.run(['/Users/wio/.local/bin/baai-ssh', '171', '--',
                        'python3', '-I', '-S', '-c', REMOTE, json.dumps(spec)],
                       capture_output=True, text=True, timeout=100)
    if r.returncode:
        raise RuntimeError(r.stderr)
    x = json.loads(r.stdout)
    for row in x['sources']:
        content = row.pop('content', None)
        if content is None:
            continue
        p = OUT / 'prefix-retirement-runtime-source' / row['runtime_path'].removeprefix('/vllm-workspace/')
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        assert sha256(p.read_bytes()).hexdigest() == row['runtime_sha256']
        row['local_path'] = str(p)
    x['reader_sha256'] = sha256(Path(__file__).read_bytes()).hexdigest()
    x['imported_primary_reader_sha256'] = sha256(Path('tools/read_prefix_retirement_identity.py').read_bytes()).hexdigest()
    (OUT / 'prefix-retirement-extra-dependencies.json').write_text(json.dumps(x, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(sources=len(x['sources']),
                          matched=sum(r.get('matches_local', False) for r in x['sources']),
                          missing=[r['runtime_path'] for r in x['sources'] if r.get('missing')],
                          new_NPU_work=0, remote_writes=[])))


if __name__ == '__main__':
    main()
