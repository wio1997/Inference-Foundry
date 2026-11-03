#!/usr/bin/env python3
"""Recheck saved startup evidence without logging requests or server addresses."""
import hashlib
import json
import math
import re
from pathlib import Path
import sys

root = Path(__file__).resolve().parents[2]
log = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('/Users/wio/work/169-vllm-ascend-reproduce-172/benchmark-20261028/raw-results/service-log-tail.txt')
raw = log.read_bytes()
assert hashlib.sha256(raw).hexdigest() == '64bfa3d8523ad86e1c466fcca166cdcfe1dc4a3676573fc51b65279a69aa8aba'
serve = (root / 'evidence/frozen-serve.sh').read_bytes()
assert hashlib.sha256(serve).hexdigest() == 'c297201185b5676254af4d7693279f89bfa60f9c76eb340058a78ecf730b7bba'
text = raw.decode(errors='strict')
groups = {}
for line in text.splitlines():
    m = re.search(r'EngineCore_DP([01]).*kv cache group sizes \[([0-9, ]+)\]', line)
    if m:
        assert m[1] not in groups
        groups[m[1]] = [int(x.strip()) for x in m[2].split(',')]
assert set(groups) == {'0', '1'} and all(v == [128] * 12 for v in groups.values())
for phrase, expected in [('decode_context_parallel_size=', '1'), ('enable_prefix_caching=', 'True')]:
    hits = re.findall(re.escape(phrase) + r'([^, )]+)', text)
    assert hits and set(hits) == {expected}
command = serve.decode()
assert re.search(r'--block-size\s+128(?:\s|$)', command)
overrides = {name: bool(re.search(r'(?<!\S)' + re.escape(name) + r'(?:=|\s)', command))
             for name in ('--prefix-match-unit', '--prefix-caching-hash-algo')}
print(json.dumps({'group_sizes': groups, 'per_engine_gcd': {e: math.gcd(*v) for e, v in groups.items()},
                  'decode_context_parallel_size_logged': 1, 'prefix_caching_logged': True,
                  'explicit_serve_flags': overrides,
                  'limits': 'Conditional inference only: historical resolved match unit, runtime source, late overrides, active hasher, and per-request timing unbound.'}, indent=2))
