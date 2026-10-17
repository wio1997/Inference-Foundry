#!/usr/bin/env python3
"""Stage a guarded per-TP KV ownership experiment over the existing Python path."""
import ast
import hashlib
import json
from pathlib import Path
import socket
import subprocess

PREFIX = '/vllm-workspace/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/'
IMAGE = 'sha256:090eeee66e347509127601e565b971a0801701e32e4e865d89cc8dec72ca7d83'
GUARD = '''        if self.vllm_config.kv_transfer_config.kv_connector_extra_config.get("preserve_mla_tp_kv", False):
            if not (
                self.backend_name == "memcache"
                and not self.use_layerwise
                and self.use_mla
                and getattr(self.vllm_config.model_config.hf_config, "model_type", None) == "glm_moe_dsa"
                and self.vllm_config.cache_config.cache_dtype == "int8"
                and self.dcp_size == 1 and self.pcp_size == 1 and self.pp_size == 1
            ):
                raise ValueError("Per-TP GLM C8 KV experiment requires same-TP non-layerwise MemCache")
            # Retain the existing rank's quantized KV rather than substituting
            # a block saved by another nominal MLA replica. Operators unchanged.
            self.put_step = 1
'''


def main():
    assert socket.gethostname() in ('B4-33U-Atlas900RCKA3-32', 'B5-5U-Atlas900RCKA3-33')
    base = Path('/data/wio/glm5-3-vllm-execution/20261017-pd-baseline')
    root = base/'rank-preservation-v1'
    root.mkdir(exist_ok=False)
    scheduler = (base/'pool-alignment-v1/pool_scheduler.py').read_bytes()
    assert hashlib.sha256(scheduler).hexdigest() == '48387ab028d93e20f9d4b42ffa53dd556c0edddc60f457e32ece4053d335b2e4'
    worker = subprocess.check_output(['docker', 'run', '--rm', '--pull', 'never', '--network', 'none',
                                    '--read-only', '--cap-drop', 'ALL', '--entrypoint', 'python3', IMAGE,
                                    '-c', 'from pathlib import Path;import sys;sys.stdout.buffer.write(Path('+repr(PREFIX+'pool_worker.py')+').read_bytes())'], timeout=40)
    assert hashlib.sha256(worker).hexdigest() == '3d39b85efa8afb750abc179a030a82d2309fd7b0770eb9fcc0da073af0c141ff'
    result = []
    for name, raw, marker, addition in [
        ('pool_scheduler.py', scheduler, '        self.num_layers = vllm_config.model_config.get_num_layers(parallel_config)\n', GUARD),
        ('pool_worker.py', worker, '        if self.use_kvpp:\n            # Every owner saves all blocks of its layer shard, including its MTP replica.\n',
         GUARD+'            self.head_or_tp_rank = self.tp_rank\n'),
    ]:
        text = raw.decode()
        assert text.count(marker) == 1
        patched = text.replace(marker, addition+marker)
        ast.parse(patched)
        (root/name).write_text(patched)
        result.append({'source': str(root/name), 'target': PREFIX+name,
                       'original_sha256': hashlib.sha256(raw).hexdigest(),
                       'sha256': hashlib.sha256(patched.encode()).hexdigest()})
    report = {'patches': result, 'scope': 'Experimental opt-in preserve_mla_tp_kv; GLM C8 nonlayerwise same-TP only. Separate rank keys and all blocks written per rank; 16x logical pool writes/storage possible. No binary/kernel/precision change or performance acceptance.'}
    (root/'result.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
