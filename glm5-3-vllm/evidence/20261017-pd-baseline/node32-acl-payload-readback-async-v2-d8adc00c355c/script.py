#!/usr/bin/env python3
"""Validate the installed synchronous ACL checksum readback before model hooks."""
import ctypes
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch_npu
import acl


def main():
    assert torch.npu.device_count() == 1
    torch.npu.set_device(0)
    count = 1048576
    source = (torch.arange(count, device='npu:0', dtype=torch.int32) % 251).to(torch.int8)
    torch.npu.synchronize()
    pointer, allocate_rc = acl.rt.malloc_host(count)
    assert allocate_rc == 0
    profiler = torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler('/out/readback-profile'))
    stream, stream_rc = acl.rt.create_stream()
    assert stream_rc == 0
    profiler.start()
    with torch.autograd.profiler.record_function('GLM53_ACL_AUDIT_READBACK_D2H'):
        copy_rc = acl.rt.memcpy_async(pointer, count, source.data_ptr(), count, 2, stream)
        assert acl.rt.synchronize_stream(stream) == 0
    profiler.stop()
    assert acl.rt.destroy_stream(stream) == 0
    assert copy_rc == 0
    actual = hashlib.sha256(ctypes.string_at(pointer, count)).hexdigest()
    expected = hashlib.sha256((np.arange(count, dtype=np.int32) % 251).astype(np.int8).tobytes()).hexdigest()
    free_rc = acl.rt.free_host(pointer)
    report = {'bytes': count, 'allocate_rc': allocate_rc, 'copy_rc': copy_rc, 'free_rc': free_rc,
              'actual_sha256': actual, 'expected_sha256': expected, 'equal': actual == expected,
              'source_format': torch_npu.get_npu_format(source),
              'scope': 'Diagnostic D2H checksum instrumentation, not model/PD or performance; same installed ACL API. Transport trace retained for classification.'}
    Path('/out/result.json').write_text(json.dumps(report, indent=2)+'\n')
    print('GLM53_RESULT_JSON_BEGIN\n'+json.dumps(report, indent=2)+'\nGLM53_RESULT_JSON_END')
    assert free_rc == 0 and report['equal']


if __name__ == '__main__':
    main()
