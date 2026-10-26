#!/usr/bin/env python3
"""Run620 isolated actual-format API and metadata-sized copy preflight.

NPU0 toy tensors only. No service, model, HCCL, or actual DSpark cache.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch_npu

OUT = Path('/data/wio/Inference_Foundry/evidence/20260928_loop081_bound/run620/format_api_smoke.json')


def descriptor(x: torch.Tensor) -> dict[str, object]:
    storage = x.untyped_storage()
    return {
        'format_id': int(torch_npu.get_npu_format(x)),
        'shape': list(x.shape),
        'stride': list(x.stride()),
        'dtype': str(x.dtype),
        'tensor_data_ptr': int(x.data_ptr()),
        'storage_base_ptr': int(storage.data_ptr()),
        'storage_cdata': int(storage._cdata),
        'storage_offset_elements': int(x.storage_offset()),
        'storage_nbytes': int(storage.nbytes()),
    }


cache = torch.empty((96, 192), device='npu:0', dtype=torch.bfloat16)
view = cache[1:2, 4:8]
metadata = torch.arange(1024, device='npu:0', dtype=torch.int32)
snapshot = torch.empty_like(metadata)
cache_d, view_d = descriptor(cache), descriptor(view)
metadata_d, snapshot_d = descriptor(metadata), descriptor(snapshot)
assert cache_d['storage_cdata'] == view_d['storage_cdata']
assert cache_d['format_id'] == view_d['format_id']
assert cache_d['storage_nbytes'] >= cache.numel() * cache.element_size()
assert metadata_d['storage_nbytes'] >= 4096 and snapshot_d['storage_nbytes'] >= 4096
snapshot.copy_(metadata, non_blocking=True)
metadata.fill_(-1)
torch.npu.synchronize()
assert torch.equal(snapshot.cpu(), torch.arange(1024, dtype=torch.int32))

result = {
    'status': 'isolated_format_api_metadata_4KiB_snapshot_pass',
    'scope': ('NPU0 toy allocation and same-stream metadata copy only. Format ID is '
              'reported as an integer; actual DSpark cache/metadata formats, '
              'binary identity and producer ordering remain unmeasured.'),
    'torch_version': torch.__version__,
    'torch_npu_version': torch_npu.__version__,
    'cache': cache_d,
    'view': view_d,
    'metadata': metadata_d,
    'snapshot': snapshot_d,
    'metadata_logical_bytes': 4096,
    'actual_W0_format_id': None,
    'strict_bound_change': None,
}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({'status': result['status'],
                  'cache_format_id': cache_d['format_id'],
                  'metadata_format_id': metadata_d['format_id']}))
