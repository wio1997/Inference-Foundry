#!/usr/bin/env python3
"""One-device environment/operation check. No model weights; not a performance measurement."""
import json
import os
import torch
import torch_npu


def main():
    torch.set_num_threads(1)
    torch.npu.set_device(0)
    x = torch.arange(1024, dtype=torch.int32, device='npu:0')
    y = (x + 3).cpu()
    torch.npu.synchronize()
    assert torch.equal(y, torch.arange(1024, dtype=torch.int32) + 3)
    print(json.dumps({'torch': torch.__version__, 'torch_npu': torch_npu.__version__,
        'visible_requested': os.environ.get('ASCEND_RT_VISIBLE_DEVICES'),
        'device_count': torch.npu.device_count(), 'device_name': torch.npu.get_device_name(0),
        'memory_free_total_bytes': torch.npu.mem_get_info(0), 'correctness': 'pass',
        'model_loaded': False, 'performance_claim': False}))


if __name__ == '__main__':
    main()
