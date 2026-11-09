"""Enqueue native sampling validity checks for the original post-sync phase.

PTA's aten::_assert_async reads back synchronously and cannot be captured.
Recorded sampling accumulates the same predicate on device; an owned pinned
snapshot is checked before the original SamplerOutput post-processing.
"""
from contextlib import contextmanager
from contextvars import ContextVar

import torch

_active_status = ContextVar('deepinfer_native_sampling_status', default=None)
_MESSAGE = 'Cannot sample from an all-zero tensor'


def assert_nonempty_sampling(total):
    condition = (total > 0).all()
    status = _active_status.get()
    if status is None:
        torch._assert_async(condition, _MESSAGE)
    else:
        status.copy_(torch.maximum(status, (~condition).to(torch.int32)))


class SamplingValidation:
    def __init__(self, device):
        self.status = torch.zeros((), dtype=torch.int32, device=device)
        self.pending = []

    def reset(self):
        self.status.zero_()

    @contextmanager
    def record(self):
        token = _active_status.set(self.status)
        try:
            yield
        finally:
            _active_status.reset(token)

    def enqueue(self):
        host = torch.empty((), dtype=torch.int32, device='cpu', pin_memory=True)
        host.copy_(self.status, non_blocking=True)
        done = torch.npu.Event()
        done.record()

        def check():
            done.synchronize()
            if host.item():
                raise RuntimeError(_MESSAGE)
        self.pending.append(check)

    def take(self):
        result, self.pending = self.pending, []
        return result

    def check_now(self):
        self.enqueue()
        for check in self.take():
            check()
