#!/usr/bin/env python3
"""Read bounded source excerpts needed for an SDMA-only GLM launch; no framework import."""
import hashlib
import json
from pathlib import Path
import re


def main():
    root = Path('/vllm-workspace/vllm-ascend')
    candidates = list((root / 'vllm_ascend').glob('**/*moe*.py'))
    candidates += [root/'vllm_ascend/ascend_config.py', root/'vllm_ascend/config.py',
                   root/'vllm_ascend/distributed/parallel_state.py',
                   root/'docs/source/tutorials/models/GLM5.3.md']
    out = {}
    for p in candidates:
        if not p.is_file() or p.stat().st_size > 300000:
            continue
        data = p.read_bytes()
        lines = data.decode(errors='replace').splitlines()
        selected = set()
        pattern = r'enable_fused_mc2|enable_flashcomm|all2all_backend|all_to_all|all_gather|reduce_scatter|allgather_reducescatter|PD disagg|5\.1\.2|glm5.3.*\.sh|VLLM_USE_V2'
        if p.suffix == '.md':
            pattern = r'5\.1\.2|PD|prefill|decode|examples/|VLLM_USE_V2|kv.cache.dtype|indexer_kv|data.parallel|tensor.parallel'
        for i, line in enumerate(lines):
            if re.search(pattern, line, re.I):
                selected.update(range(max(0, i-2), min(len(lines), i+6)))
        if selected:
            out[str(p)] = {'sha256': hashlib.sha256(data).hexdigest(), 'line_count': len(lines),
                          'truncated': len(selected)>220,
                          'lines': [{'line': i+1, 'text': lines[i]} for i in sorted(selected)[:220]}
    out['example_paths'] = [str(p) for p in (root/'examples').rglob('*') if p.is_file() and
                            any('glm5.3' in x.lower() for x in p.parts) and p.suffix in ('.py','.sh','.yaml','.json')]
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
