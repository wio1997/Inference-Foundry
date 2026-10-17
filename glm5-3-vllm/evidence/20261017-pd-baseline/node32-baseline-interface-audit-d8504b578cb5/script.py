#!/usr/bin/env python3
"""Read the installed API/cache interfaces without importing or initializing NPU code."""
import hashlib
import importlib.util
import json
from pathlib import Path
import re


def main():
    root = Path(importlib.util.find_spec('vllm').origin).parent
    ascend = Path('/vllm-workspace/vllm-ascend/vllm_ascend')
    paths = list((root/'entrypoints').rglob('*.py'))
    paths += [ascend/'ascend_config.py']
    patterns = r'return_token_ids|enable_prompt_tokens_details|async def reset_prefix_cache|def reset_prefix_cache|reset_connector|mlapo_keep_prefill_weights'
    files = {}
    for p in paths:
        raw = p.read_bytes()
        lines = raw.decode().splitlines()
        selected = set()
        for i, line in enumerate(lines):
            if re.search(patterns, line):
                selected.update(range(max(0, i-3), min(len(lines), i+14)))
        if selected:
            files[str(p)] = {'sha256': hashlib.sha256(raw).hexdigest(),
                            'lines': [{'line': i+1, 'text': lines[i]} for i in sorted(selected)]}
    print(json.dumps({'files': files}, indent=2))


if __name__ == '__main__':
    main()
