#!/usr/bin/env python3
"""Freeze a minimal local-complete extension of the pinned official CPU proxy."""
import argparse
import ast
import hashlib
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--official', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    original = a.official.read_bytes()
    assert hashlib.sha256(original).hexdigest() == '85b8982948530758be655d3448ff10fc3979e56622221feace560272a81f1469'
    helper = Path(__file__).with_name('local_complete_proxy_helper.py').read_text()
    source = original.decode()
    needle = 'async def handle_completions_impl(api: str, request: Request):\n'
    assert source.count(needle) == 1
    source = source.replace(needle, needle + "    local_response = await _glm53_maybe_complete_at_P(api, request)\n    if local_response is not None:\n        return local_response\n")
    marker = 'if __name__ == "__main__":\n'
    assert source.count(marker) == 1
    source = source.replace(marker, helper + '\n\n' + marker)
    before, after = ast.parse(original.decode()), ast.parse(source)
    names = {}
    for node in before.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names[node.name] = node
    changed = {}
    for node in after.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name in names:
            if ast.dump(node) != ast.dump(names[node.name]):
                changed[node.name] = node
    assert set(changed) == {'handle_completions_impl'}
    node = changed['handle_completions_impl']
    node.body = node.body[2:]
    assert ast.dump(node) == ast.dump(names[node.name]), 'Native handler remainder changed'
    a.output.mkdir(exist_ok=False, parents=True)
    data = source.encode()
    (a.output / 'proxy.py').write_bytes(data)
    report = {'original_sha256': hashlib.sha256(original).hexdigest(), 'source_sha256': hashlib.sha256(data).hexdigest(),
              'helper_sha256': hashlib.sha256(helper.encode()).hexdigest(),
              'all_original_function_and_class_AST_unchanged_except_two_statement_guard': True,
              'native_handler_remainder_AST_unchanged': True,
              'scope': 'Independent CPU proxy; opt-in single-token local completion. Original official29880/experiment29881 and model code remain unchanged.'}
    (a.output / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
