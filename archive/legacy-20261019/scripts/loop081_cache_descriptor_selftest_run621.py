#!/usr/bin/env python3
"""Run621 no-service patch and CPU fake-cache observer selftest."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch

import loop081_cache_descriptor_patch_run621 as patch

OUT = Path('/data/wio/Inference_Foundry/evidence/20260928_loop081_bound/run621/preflight/selftest.json')
manifest, old, new = patch.prepared()
assert len(old) == len(new) == 9
assert all(hashlib.sha256(new[key]).hexdigest() == manifest['sources'][key]['patched_sha256']
           for key in new)
runner = new['runner'].decode()
dspark = new['dspark'].decode()
assert runner.count('EXTREME_LOOP081_RUN621_ARM') == 1
assert runner.count('EXTREME_LOOP081_RUN621_FLUSH_AFTER_EXISTING_SYNC') == 1
assert dspark.count('EXTREME_LOOP081_RUN621_CAPTURE_CPU_DESCRIPTORS') == 1
assert dspark.count('EXTREME_LOOP081_RUN621_CACHE_DESCRIPTOR_HELPER') == 1
assert all(token not in patch.HELPER for token in (
    '.cpu()', '.item()', '.synchronize()', 'torch.ops.', 'torch.distributed.'))

namespace = {'torch': torch}
exec(compile(patch.HELPER, '<run621-helper>', 'exec'), namespace)
capture = namespace['_p621_capture_cache_descriptors']
layers = {}
for layer in (43, 44, 45):
    cache = torch.empty((8, 16, 1, 192), dtype=torch.bfloat16)
    cache_layer = SimpleNamespace(kv_cache=cache, block_size=16,
                                  prefix=f'fake.layer.{layer}.swa')
    layers[str(layer)] = SimpleNamespace(self_attn=SimpleNamespace(
        dsa_attn=SimpleNamespace(swa_cache_layer=cache_layer)))
draft_model = SimpleNamespace(layers=layers)
common = SimpleNamespace(query_start_loc=torch.arange(13, dtype=torch.int32),
                         seq_lens=torch.arange(12, dtype=torch.int32),
                         slot_mapping=torch.arange(96, dtype=torch.int32),
                         block_table_tensor=torch.arange(12, dtype=torch.int32))
state = SimpleNamespace(cycle_index=64, target_slot_mapping=torch.arange(96),
                        target_positions=torch.arange(96),
                        target_input_ids=torch.arange(96))
handoff = SimpleNamespace(proposer=SimpleNamespace(model=SimpleNamespace(model=draft_model)),
                          common_attn_metadata=common, _p621_rank=0, _p621_cohort=5,
                          _p621_run_ts='RUN621-FAKE')
row1 = capture(handoff, state)
row2 = capture(handoff, state)
assert row1['status'] == 'descriptors_only_no_device_values'
assert len(row1['layers']) == 3 and row1['cycle'] == 64
assert [x['layer'] for x in row1['layers']] == ['43', '44', '45']
assert all(x['cache']['status'] == 'tensor' for x in row1['layers'])
assert [x['cache']['storage_cdata'] for x in row1['layers']] == [
    x['cache']['storage_cdata'] for x in row2['layers']]
assert all(x['cache']['format_id'] is None for x in row1['layers'])

missing = capture(SimpleNamespace(
    proposer=SimpleNamespace(model=None), common_attn_metadata=common), state)
assert missing['status'] == 'missing_draft_layers'
result = {
    'status': 'run621_patch_check_cpu_fake_descriptor_pass_not_live_ready',
    'source_manifest_sha256': hashlib.sha256(
        (OUT.parent / 'patch_check.json').read_bytes()).hexdigest(),
    'patch_sha256': hashlib.sha256(Path(patch.__file__).read_bytes()).hexdigest(),
    'source_count': len(new),
    'fake_layer_count': len(row1['layers']),
    'fake_cache_storage_stable': True,
    'scope': 'Pure source patch construction and CPU fake cache. No service, NPU, model, HCCL, live cache lifetime or strict Bound.',
}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({'status': result['status'], 'source_count': len(new)}))
