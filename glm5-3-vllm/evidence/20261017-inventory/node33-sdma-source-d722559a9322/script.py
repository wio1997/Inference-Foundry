#!/usr/bin/env python3
"""Identify SDMA controls in the actual offline image, without device initialization."""
import importlib.util
import json
from pathlib import Path
import re
import subprocess


def main():
    out = {'top_dirs': [str(p) for p in Path('/').iterdir() if p.is_dir()],
           'source_matches': [], 'binary_controls': {}}
    spec = importlib.util.find_spec('mooncake')
    if spec and spec.submodule_search_locations:
        root = Path(next(iter(spec.submodule_search_locations)))
        out['mooncake_files'] = [str(p) for p in root.rglob('*') if p.is_file()]
        for p in root.glob('*.so'):
            proc = subprocess.Popen(['strings', str(p)], stdout=subprocess.PIPE, text=True)
            matches = set()
            for line in proc.stdout:
                if re.search(r'SDMA|MC_ASCEND|MC_.*TRANSPORT|ASCEND_.*TRANSPORT', line, re.I):
                    matches.add(line.strip()[:1000])
            proc.wait(timeout=20)
            out['binary_controls'][str(p)] = sorted(matches)[:160]
    roots = [Path('/vllm-workspace/vllm-ascend'), Path('/vllm-workspace/vllm')]
    roots += [p for p in Path('/').iterdir() if p.is_dir() and 'moon' in p.name.lower()]
    for root in roots:
        for p in root.rglob('*'):
            if not p.is_file() or p.suffix not in ('.py', '.md', '.sh', '.cpp', '.h', '.cc') or p.stat().st_size > 300000:
                continue
            if any(x in p.parts for x in ('.git', 'locale', 'node_modules')):
                continue
            lines = p.read_text(errors='replace').splitlines()
            selected = set()
            for i, line in enumerate(lines):
                if re.search(r'\bSDMA\b|HCCL_INTRA_ROCE_ENABLE|HCCL_INTRA_PCIE_ENABLE|MC_ASCEND', line, re.I):
                    selected.update(range(max(0, i-2), min(len(lines), i+4)))
            if selected:
                out['source_matches'].append({'file': str(p),
                    'lines': [{'line': i+1, 'text': lines[i]} for i in sorted(selected)[:100]]})
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
