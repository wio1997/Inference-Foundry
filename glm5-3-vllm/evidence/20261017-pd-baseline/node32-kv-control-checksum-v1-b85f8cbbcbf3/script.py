#!/usr/bin/env python3
"""Validate NPU integer checksums; return only checksum control values to CPU."""
import json
from pathlib import Path

import numpy as np
import torch
import torch_npu


def _glm53_control_checksum_tensor(value):
    data = value.contiguous().view(torch.uint8).reshape(-1).to(torch.int32)
    position = torch.arange(data.numel(), dtype=torch.int32, device=data.device)
    weight1 = position.remainder(65521)+1
    weight2 = (position.remainder(65519)*17).remainder(65519)+1
    return torch.stack((data.sum(dtype=torch.int64),
                        (data*weight1).sum(dtype=torch.int64),
                        (data*weight2).sum(dtype=torch.int64),
                        (data*data).sum(dtype=torch.int64)))


def main():
    assert torch.npu.device_count() == 1
    torch.npu.set_device(0)
    size = 1048576
    source = (torch.arange(size, dtype=torch.int32, device='npu:0') % 251).to(torch.int8)
    torch.npu.synchronize()
    stream = torch.npu.Stream()
    profiler = torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler('/out/checksum-profile'))
    profiler.start()
    with torch.npu.stream(stream):
        result = _glm53_control_checksum_tensor(source)
        actual = result.cpu().tolist()
    profiler.stop()
    p = np.arange(size, dtype=np.int64)
    v = p % 251
    expected = [int(v.sum()), int((v*((p % 65521)+1)).sum()),
                int((v*(((p % 65519)*17) % 65519+1)).sum()), int((v*v).sum())]
    report = {'input_bytes': size, 'control_bytes_returned_to_host': result.numel()*result.element_size(),
              'actual': actual, 'expected': expected, 'equal': actual == expected,
              'result_device': str(result.device),
              'scope': 'Four position/value-sensitive integer moments, not cryptographic SHA or proof of byte identity. Bulk input stays on NPU; only checksum metadata copied toCPU. Existing operators only; no performance claim.'}
    Path('/out/result.json').write_text(json.dumps(report, indent=2)+'\n')
    print('GLM53_RESULT_JSON_BEGIN\n'+json.dumps(report, indent=2)+'\nGLM53_RESULT_JSON_END')
    assert report['equal'] and result.device.type == 'npu'


if __name__ == '__main__':
    main()
