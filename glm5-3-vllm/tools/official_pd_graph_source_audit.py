#!/usr/bin/env python3
"""CPU-only pinned-image source audit before the bounded prefill graph experiment."""
import hashlib,json
from pathlib import Path
EXPECTED = {'/vllm-workspace/vllm-ascend/vllm_ascend/attention/sfa_v1.py': '9b94f67f30364d40caf4a7a67af0f0365d1ba0b5048939873ab0464e95b096dc', '/vllm-workspace/vllm-ascend/vllm_ascend/compilation/acl_graph.py': '8e1c269e331d005f1361acc5cec24402bff7ae39b75e952ea890da7812804955', '/vllm-workspace/vllm-ascend/vllm_ascend/platform.py': '0f4e619517058f36b700a521e93c449d78b269c1d29ed1a18cbc312f2d67b082', '/vllm-workspace/vllm-ascend/vllm_ascend/compilation/compiler_interface.py': '45f25b7f711a5c69e67646ae3323dd4bef3e5e9414a87b32cf9a61653ff750d6'}
observed={name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in EXPECTED}
report={'completed':observed==EXPECTED,'scope':'CPU-only exact module bytes; no torch/NPU import or execution. Capability source verification only, not model compatibility or performance.','expected':EXPECTED,'observed':observed}
print(json.dumps(report,indent=2))
assert report['completed'],'Image graph source differs from reviewed pinned commit'
