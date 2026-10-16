#!/usr/bin/env python3
"""Read image source identities and selected deployment/quantization evidence; no NPU imports."""
import hashlib
import importlib.metadata as md
import importlib.util
import json
from pathlib import Path
import re
import subprocess


def excerpt(p, pattern, before=2, after=5, limit=100):
    data = p.read_bytes()
    lines = data.decode(errors='replace').splitlines()
    selected = set()
    for i, line in enumerate(lines):
        if re.search(pattern, line, re.I):
            selected.update(range(max(0, i-before), min(len(lines), i+after)))
    return {'sha256': hashlib.sha256(data).hexdigest(), 'total_lines': len(lines),
            'matched_excerpt_truncated': len(selected) > limit,
            'lines': [{'line': i+1, 'text': lines[i]} for i in sorted(selected)[:limit]]}


def main():
    result = {'git': {}, 'modules': {}, 'files': {}, 'distribution_names': []}
    for d in md.distributions():
        name = d.metadata.get('Name', '')
        if any(x in name.lower() for x in ('moon', 'transfer', 'hccl', 'ascend')):
            result['distribution_names'].append({'name': name, 'version': d.version})
    for mod in ('mooncake', 'mooncake_transfer_engine'):
        s = importlib.util.find_spec(mod)
        result['modules'][mod] = None if s is None else s.origin
    for repo in ('vllm', 'vllm-ascend'):
        root = Path('/vllm-workspace') / repo
        result['git'][repo] = {}
        for name, args in [('sha', ['rev-parse', 'HEAD']), ('status', ['status', '--porcelain']),
                           ('diff', ['diff', 'HEAD'])]:
            p = subprocess.run(['git', '-C', str(root)] + args, capture_output=True, text=True, timeout=15)
            result['git'][repo][name] = {'rc': p.returncode, 'stdout': p.stdout[:65536], 'stderr': p.stderr[:2048]}
    roots = [Path('/vllm-workspace/vllm'), Path('/vllm-workspace/vllm-ascend')]
    for root in roots:
        candidates = set(root.glob('docs/**/GLM5.3*')) | set(root.glob('docs/**/glm5*'))
        if root.name == 'vllm':
            candidates |= {root/'vllm/envs.py', root/'vllm/config/vllm.py'}
        else:
            candidates |= set(root.glob('vllm_ascend/distributed/kv_transfer/**/*.py'))
            candidates |= set(root.glob('vllm_ascend/quantization/**/kv*.py'))
            candidates |= set(root.glob('vllm_ascend/quantization/**/cache*.py'))
            candidates |= set(root.glob('vllm_ascend/**/glm*py'))
        for p in sorted(candidates):
            if p.is_file() and p.stat().st_size < 300000:
                result['files'][str(p)] = excerpt(p, r'VLLM_USE_V2_MODEL_RUNNER|use_v2_model_runner|W8A8C8|kv_cache_dtype|INT8_DYNAMIC|mooncake|kv_connector|kv_role|tensor.parallel|data.parallel|C8|quantiz')
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
