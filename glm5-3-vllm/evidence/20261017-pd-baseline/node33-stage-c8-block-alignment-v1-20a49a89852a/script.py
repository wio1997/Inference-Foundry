#!/usr/bin/env python3
"""Stage a guarded Python scheduler fix; no base image or operator modification."""
import ast,hashlib,json,socket,subprocess
from pathlib import Path
TARGET='/vllm-workspace/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py'
ORIGINAL_SHA='b57b77725787f04cf47916e520c58f31fb1ef6ce3addb8a8855d81cbe0026984'
PATCHED_SHA='48387ab028d93e20f9d4b42ffa53dd556c0edddc60f457e32ece4053d335b2e4'
OLD='        if len(self.cacheable_group_ids) != len(self.original_block_size):\n            # Private compressor state is rebuilt locally. Recompute the last\n            # whole token page rather than resume inside a compressed block.\n            num_external_hit_tokens = self._floor_to_cache_transfer_granularity(num_external_hit_tokens)\n'
NEW='        glm_c8_block_resume = (\n            self.backend_name == "memcache"\n            and not self.use_layerwise\n            and self._discard_partial_chunks\n            and getattr(self.vllm_config.model_config.hf_config, "model_type", None) == "glm_moe_dsa"\n            and self.vllm_config.cache_config.cache_dtype == "int8"\n        )\n        if glm_c8_block_resume or len(self.cacheable_group_ids) != len(self.original_block_size):\n            # MemCache moves whole chunks. For GLM C8 a fully cached prompt\n            # must not resume inside its final transferred block after the\n            # scheduler\'s one-token trim. Recompute the trailing whole block.\n            # Private compressor states already require the same alignment.\n            num_external_hit_tokens = self._floor_to_cache_transfer_granularity(num_external_hit_tokens)\n'
IMAGE='sha256:090eeee66e347509127601e565b971a0801701e32e4e865d89cc8dec72ca7d83'
def main():
 assert socket.gethostname() in ('B4-33U-Atlas900RCKA3-32','B5-5U-Atlas900RCKA3-33')
 root=Path('/data/wio/glm5-3-vllm-execution/20261017-pd-baseline/pool-alignment-v1');root.mkdir(exist_ok=False)
 command=['docker','run','--rm','--pull','never','--network','none','--read-only','--cap-drop','ALL','--entrypoint','python3',IMAGE,'-c','import pathlib,sys; sys.stdout.buffer.write(pathlib.Path('+repr(TARGET)+').read_bytes())']
 raw=subprocess.check_output(command,timeout=40);assert hashlib.sha256(raw).hexdigest()==ORIGINAL_SHA
 source=raw.decode();assert source.count(OLD)==1
 patched=source.replace(OLD,NEW);ast.parse(patched);assert hashlib.sha256(patched.encode()).hexdigest()==PATCHED_SHA
 (root/'original-pool_scheduler.py').write_bytes(raw);(root/'pool_scheduler.py').write_text(patched)
 result={'source':str(root/'pool_scheduler.py'),'target':TARGET,'original_sha256':ORIGINAL_SHA,'sha256':PATCHED_SHA,'scope':'MemCache, GLM-MoE-DSA, int8, non-layerwise full-block transfer only; whole trailing block recomputed; no kernel or precision change.'}
 (root/'manifest.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
