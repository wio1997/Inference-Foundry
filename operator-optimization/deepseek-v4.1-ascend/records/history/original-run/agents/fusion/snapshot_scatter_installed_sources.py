#!/usr/bin/env python3
"""Snapshot four already-inspected source files from the task container, verifying hashes."""
from pathlib import Path
import hashlib,shlex,subprocess
root=Path(__file__).resolve().parents[2]
base='/vllm-workspace/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/'
sources={
'op_impl/ai_core/tbe/custom_transformer_impl/ascendc/scatter_nd_update_sk/scatter_nd_update_sk.cpp':'7beb62e25c2c5f5e5059054da4f937747e925cf1dec46f460fd7ac39b791f57d',
'op_impl/ai_core/tbe/custom_transformer_impl/ascendc/scatter_nd_update_sk/arch22/scatter_nd_update_hp.h':'5741acc73f5ee48c1cfcde63cf1ebb7ddf2a748173c32ee45367a2a4198871bb',
'op_impl/ai_core/tbe/custom_transformer_impl/ascendc/scatter_nd_update_sk/arch22/scatter_nd_update_common.h':'b33f796fdd033582dee545d58d4c8f2f6e010a58c043083db15a575596e27b01',
'op_api/include/aclnnop/aclnn_scatter_nd_update_sk.h':'f5b0d646e33b251697dc97df6f18123923ef36ccdbd4a4884006393e5e7eeede'}
for rel,expected in sources.items():
 cmd=shlex.join(['docker','exec','wio-vllm-v41-repro169-stage-20261028','cat',base+rel])
 r=subprocess.run(['/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh',cmd],capture_output=True,timeout=40,check=True)
 assert hashlib.sha256(r.stdout).hexdigest()==expected,rel
 dest=root/'agents/fusion/frozen-source/cann-scatter-sk'/rel;dest.parent.mkdir(parents=True,exist_ok=True)
 if dest.exists():assert hashlib.sha256(dest.read_bytes()).hexdigest()==expected,rel
 else:dest.write_bytes(r.stdout)
 print(rel,'verified_sha256',expected)
