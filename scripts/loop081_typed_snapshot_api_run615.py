#!/usr/bin/env python3
"""Isolated NPU descriptor and original-stream snapshot API preflight.

No model/service/HCCL. The only synchronize is after both queued operations.
This proves API availability and toy stream FIFO, not live KV capture safety.
"""
import json
from pathlib import Path

import torch
import torch_npu

OUT = Path('/data/wio/Inference_Foundry/evidence/20260928_loop081_bound/run615')


def need(ok, why):
    if not ok:
        raise ValueError(why)


def descriptor(x):
    storage = x.untyped_storage()
    return {'device': str(x.device), 'dtype': str(x.dtype),
            'shape': list(x.shape), 'stride': list(x.stride()),
            'tensor_data_ptr': x.data_ptr(), 'storage_base_ptr': storage.data_ptr(),
            'storage_cdata': storage._cdata, 'storage_nbytes': storage.nbytes(),
            'storage_offset_elements': x.storage_offset()}


src = torch.arange(192, device='npu:0', dtype=torch.int32).reshape(96, 2)
snap = torch.empty_like(src)
cache = torch.empty((96, 192), device='npu:0', dtype=torch.bfloat16)
cache_view = cache[1:2, 4:8]
src_d = descriptor(src)
snap_d = descriptor(snap)
cache_d = descriptor(cache)
view_d = descriptor(cache_view)
need(src_d['storage_cdata'] != snap_d['storage_cdata'] and
     cache_d['storage_cdata'] == view_d['storage_cdata'] and
     cache_d['storage_base_ptr'] == view_d['storage_base_ptr'] and
     view_d['storage_offset_elements'] > 0,
     'storage identity/offset behavior')
scratch_bytes = src.numel() * src.element_size() + cache_view.numel() * cache_view.element_size()
need(scratch_bytes <= 128 * 1024, 'selected-row scratch budget')
# Both operations enter the original current stream. The snapshot must hold
# pre-mutation values even though Host does not synchronize between them.
current_stream = torch.npu.current_stream()
snap.copy_(src, non_blocking=True)
src.fill_(777)
torch.npu.synchronize()
actual = snap.cpu().reshape(-1).tolist()
need(actual == list(range(192)) and
     bool(torch.all(src == 777).item()), 'same-stream snapshot ordering')
result = {
    'status': 'isolated_npu_descriptor_and_same_stream_snapshot_pass',
    'scope': 'Device0-only toy allocation. One final synchronize after queued copy+mutation. Descriptor APIs and FIFO snapshot behavior available; no actual DSpark KV or cache generation/read-domain proof.',
    'torch_version': torch.__version__,
    'torch_npu_version': torch_npu.__version__,
    'current_stream_id': current_stream.npu_stream,
    'scratch_bytes': scratch_bytes,
    'descriptors': {'source': src_d, 'snapshot': snap_d,
                    'cache': cache_d, 'cache_view': view_d},
    'first_snapshot_values': actual[:8],
    'last_snapshot_values': actual[-8:],
}
OUT.mkdir(parents=True, exist_ok=True)
(OUT / 'snapshot_api_smoke.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({'status': result['status'], 'scratch_bytes': scratch_bytes}))
